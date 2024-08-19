import torch
import os
import time
import random
import csv
import math
import bitsandbytes as bnb


from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset, DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils.other import fsdp_auto_wrap_policy


from experiments.data.get_dataset import get_dataset
from cats_exp.cats import CatsModel, CatsConfig, get_cats_model, model_class_to_pkg_str
from utils.constants import MISTRAL_7B, REFINED_WEB, LLAMA_7B


def print_gpu_memory():
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    else:
        print("CUDA is not available")


def run_evaluation(
    model,
    eval_dataloader,
    total,
    max_lengths,
    tokenizer,
    with_kernel_injection=False,
):
    start_time = time.time()
    count = 0

    if with_kernel_injection:
        print("Running with kernel injection...")
    else:
        print("Running without kernel injection...")

    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
            size = batch["input_ids"].shape
            if step > total:
                break
            print(f"step: {step}/ {size}")
            count += 1
            input_ids = batch["input_ids"].to(model.device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                min_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )

    elapsed_time = time.time() - start_time
    print(
        f"{'With' if with_kernel_injection else 'Without'} kernel injection: {elapsed_time}"
    )
    return elapsed_time


def record_perplexity(
    target_sparsity: float,
    base_perplexity: float,
    initial_perplexity: float,
    final_perplexity: float,
):
    csv_file = "perplexity_measurements.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Target Sparsity",
                    "Base Perplexity",
                    "Initial CATS Perplexity",
                    "Final Perplexity",
                ]
            )

        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                target_sparsity,
                base_perplexity,
                initial_perplexity,
                final_perplexity,
            ]
        )


def record_timing(elapsed_time, with_kernel_injection, target_sparsity: float = 0.5):
    csv_file = "timing_measurements.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "With Kernel Injection",
                    "Elapsed Time",
                    "Target Sparsity",
                ]
            )

        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                with_kernel_injection,
                elapsed_time,
                target_sparsity,
            ]
        )


def calculate_perplexity(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for idx, batch in tqdm(
            enumerate(eval_dataloader), desc="Calculating Perplexity"
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            print(logits.shape)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="sum")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            total_loss += loss.item()
            total_tokens += torch.sum(attention_mask).item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def ce_loss(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).mean()
    return loss


import torch
import os
import time
import random
import csv
import math

from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Subset, DataLoader

from experiments.data.get_dataset import get_dataset
from cats_exp.cats import CatsModel, CatsConfig, get_cats_model, model_class_to_pkg_str
from utils.constants import MISTRAL_7B, REFINED_WEB, LLAMA_7B


def train_model(model, train_dataloader, num_epochs, max_steps, learning_rate, device):
    model.train()

    # Set up 8-bit AdamW optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)

    # Set up scheduler
    total_steps = min(len(train_dataloader) * num_epochs, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    steps = 0
    log_steps = 5
    should_break = False
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            steps += 1
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if steps >= max_steps:
                should_break = True
                break

            if steps % log_steps == 0:
                print(f"Step {steps} Loss: {loss.item()}")

        if should_break:
            break

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

    print(f"Training stopped after {steps} steps")

    return model, optimizer


def sample_cats_zero_shot(
    model_dir=MISTRAL_7B,
    wrapped_model_class_name="MistralForCausalLM",
    save_dir="cats_exp/mistral_cats_zero_shot",
    model_name="CATS",
    target_sparsity: float = 0.75,
):
    save_dir = (
        model_name + "_" + model_dir + "_" + f"sparsity_{int(target_sparsity*100)}p"
    )
    print(save_dir)
    # Define Base model and cats model
    base_model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_7B, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = get_cats_model(
        base_model, target_sparsity=target_sparsity, is_share_params=False
    )

    model.to(torch.bfloat16)
    cats_config = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print(model)
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            print(name, param.dtype)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")

    dataset = get_dataset(
        REFINED_WEB,
        tokenizer,
        model_type="Cats_Mistral",
        max_seq_length=1024,
    )  # TODO: take size of dataset as an input
    data_collator = dataset.get_data_collator()
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=data_collator
    )
    print_gpu_memory()
    accelerator = Accelerator()
    base_model, train_dataloader, eval_dataloader = accelerator.prepare(
        base_model, train_dataloader, eval_dataloader
    )
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # if getattr(accelerator.state, "fsdp_plugin", None) is not None:
    #     fsdp_plugin = accelerator.state.fsdp_plugin
    #     fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    print_gpu_memory()

    total = 5
    count = 0
    max_lengths = 3000
    # start_time = time.time()
    # model.eval()
    # with torch.no_grad():
    #     for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
    #         size = batch["input_ids"].shape
    #         print(f"step: {step}/ {size}")
    #         if step > total:
    #             break
    #         count += 1
    #         input_ids = batch["input_ids"].to(
    #             model.device
    #         )  # Move input tensors to the device
    #         outputs = model.generate(
    #             input_ids=input_ids,
    #             max_length=max_lengths,
    #             min_length=max_lengths,
    #             pad_token_id=tokenizer.eos_token_id,
    #             use_cache=True,
    #             do_sample=True,
    #             temperature=0.7,
    #             num_beams=1,
    #         )
    # elapsed_time = time.time() - start_time
    # record_timing(elapsed_time, with_kernel_injection="Dense", target_sparsity=0)
    # print(f"Dense model generation: {elapsed_time}")

    model.enable_collect_stats()
    for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
        # print(batch["input_ids"].shape)
        if step > 1:
            break
        print(f"step: {step}")
        _ = model(torch.tensor(batch["input_ids"]).to("cuda"))
    model.set_thresholds()
    model.init_stats()
    for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
        if step > 1:
            break
        print(f"step: {step}")
        _ = model(torch.tensor(batch["input_ids"]).to("cuda"))

    # Print sparsities
    sparsities = model.get_sparsity()
    for layer_idx, sparsity in sparsities.items():
        print(f"layer {layer_idx} activation sparsity: {sparsity}")

    # Calculate perplexity for the base model
    base_perplexity = calculate_perplexity(
        base_model, eval_dataloader, accelerator.device
    )
    print(f"Base model perplexity: {base_perplexity}")

    # Calculate perplexity for the CATS model
    model.disable_collect_stats()
    initial_cats_perplexity = calculate_perplexity(
        model, eval_dataloader, accelerator.device
    )
    print(f"Initial CATS model perplexity: {initial_cats_perplexity}")

    # Train the model
    model, optimizer = train_model(
        model,
        train_dataloader,
        1,
        500,
        5e-5,
        accelerator.device,
    )

    # Calculate final perplexity
    final_perplexity = calculate_perplexity(model, eval_dataloader, accelerator.device)
    print(f"Final model perplexity: {final_perplexity}")

    # Record perplexity measurements
    record_perplexity(
        target_sparsity=target_sparsity,
        base_perplexity=base_perplexity,
        initial_perplexity=initial_cats_perplexity,
        final_perplexity=final_perplexity,
    )

    # Model generation
    model.eval()
    model.disable_collect_stats()
    total = 5
    # Run without kernel injection
    elapsed_time_without = run_evaluation(
        model,
        eval_dataloader,
        total,
        max_lengths,
        tokenizer,
        with_kernel_injection=False,
    )
    record_timing(
        elapsed_time_without,
        with_kernel_injection=False,
        target_sparsity=target_sparsity,
    )

    # Compile for kernel injection
    print("Compiling....")
    model.inject_kernel()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
            size = batch["input_ids"].shape
            if step > 1:
                break
            print(f"step: {step}/ {size}")
            input_ids = batch["input_ids"].to(model.device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )
            print("Compiling finished!")

    # Run with kernel injection
    elapsed_time_with = run_evaluation(
        model,
        eval_dataloader,
        total,
        max_lengths,
        tokenizer,
        with_kernel_injection=True,
    )
    record_timing(
        elapsed_time_with, with_kernel_injection=True, target_sparsity=target_sparsity
    )
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(os.path.join(save_dir, model_name))
    cats_config.save_pretrained(os.path.join(save_dir, model_name))
    tokenizer.save_pretrained(os.path.join(save_dir, model_name))
    cats_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(save_dir, model_name), trust_remote_code=True
    )

    # Print sparsities
    sparsities = model.get_sparsity()
    for layer_idx, sparsity in sparsities.items():
        print(f"layer {layer_idx} activation sparsity: {sparsity}")


if __name__ == "__main__":
    target_sparsites = [0.5, 0.7, 0.9, 0.95]
    for _ in range(1):
        for sparsity in target_sparsites:
            sample_cats_zero_shot(target_sparsity=sparsity)
    # save_dir = "cats_model_exp"
    # model = AutoModelForCausalLM.from_pretrained(
    #     MISTRAL_7B,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # )
    # cats_model = get_cats_model(model)
    # CatsConfig.register_for_auto_class()
    # CatsModel.register_for_auto_class("AutoModelForCausalLM")
    # cats_model.save_pretrained(save_dir)
    # print(cats_model)
    # cats_model = AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)
    # print(cats_model)
