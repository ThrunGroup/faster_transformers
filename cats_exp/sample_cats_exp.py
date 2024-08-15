import torch
import os
import time

from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from experiments.data.get_dataset import get_dataset
from cats_exp.cats import CatsModel, CatsConfig, get_cats_model, model_class_to_pkg_str
from utils.constants import MISTRAL_7B, REFINED_WEB, LLAMA_7B


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


def sample_cats_zero_shot(
    model_dir=MISTRAL_7B,
    wrapped_model_class_name="MistralForCausalLM",
    save_dir="cats_exp/mistral_cats_zero_shot",
):
    # Define model
    config = AutoConfig.from_pretrained(model_dir)

    pretrained_kwargs = {
        "pretrained_model_name_or_path": MISTRAL_7B,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    cats_config = CatsConfig(config, wrapped_model_class_name="MistralForCausalLM")
    model: CatsModel = CatsModel(cats_config, pretrained_kwargs=pretrained_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print(model)

    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")

    dataset = get_dataset(
        REFINED_WEB,
        tokenizer,
        model_type="Cats_Mistral",
        max_seq_length=512,
    )
    data_collator = dataset.get_data_collator()
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()

    eval_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, collate_fn=data_collator
    )

    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.wrapped_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_collect_stats()

    total = 5
    count = 0
    max_lengths = 2000
    start_time = time.time()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
            size = batch["input_ids"].shape
            print(f"step: {step}/ {size}")
            if step > total:
                break
            count += 1
            input_ids = batch["input_ids"].to(
                model.device
            )  # Move input tensors to the device
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )
    print(f"Dense model generation: {time.time()-start_time}")

    for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
        # print(batch["input_ids"].shape)
        if step > 1:
            break
        print(f"step: {step}")
        _ = model(torch.tensor(batch["input_ids"]))
    model.set_thresholds()
    model.init_stats()
    for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
        if step > 1:
            break
        print(f"step: {step}")
        _ = model(torch.tensor(batch["input_ids"]))

    # Model generation
    model.eval()
    total = 5
    count = 0
    max_lengths = 2000
    start_time = time.time()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
            size = batch["input_ids"].shape
            print(f"step: {step}/ {size}")
            if step > total:
                break
            count += 1
            input_ids = batch["input_ids"].to(
                model.device
            )  # Move input tensors to the device
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )
    print(f"Without kernel injection: {time.time()-start_time}")

    model.inject_kernel()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
            size = batch["input_ids"].shape
            print(f"step: {step}/ {size}")
            if step > total:
                break
            count += 1
            input_ids = batch["input_ids"].to(
                model.device
            )  # Move input tensors to the device
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )
    count = 0
    max_lengths = 2000
    start_time = time.time()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=total):
            size = batch["input_ids"].shape
            print(f"step: {step}/ {size}")
            if step > total:
                break
            count += 1
            input_ids = batch["input_ids"].to(
                model.device
            )  # Move input tensors to the device
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
            )
    print(f"With kernel injection: {time.time()-start_time}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(save_dir)
    cats_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Print sparsities
    sparsities = model.get_sparsity()
    print(sparsities)
    print("WASSUP")
    for layer_idx, sparsity in sparsities.items():
        print(f"layer {layer_idx} activation sparsity: {sparsity}")


if __name__ == "__main__":
    sample_cats_zero_shot()
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
