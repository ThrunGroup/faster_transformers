import torch
import os
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from experiments.data.get_dataset import get_dataset
from cats_exp.cats import CatsModel, CatsConfig
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


def sample_cats_zero_shot(model_dir=MISTRAL_7B, wrapped_model_class_name="MistralForCausalLM", save_dir="cats/mistral_cats_zero_shot"):
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
        val_dataset, batch_size=2, shuffle=True, collate_fn=data_collator
    )

    accelerator = Accelerator()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.wrapped_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_collect_stats()

    for step, batch in tqdm(enumerate(eval_dataloader, start=1), total=1):
        _ = model(torch.tensor(batch["input_ids"]))
    model.set_thresholds()

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(save_dir)
    cats_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Print sparsities
    sparsities = model.get_sparsity()
    for layer_idx, sparsity in sparsities.items():
        print(f"layer {layer_idx} activation sparsity: {sparsity}")

if __name__ == "__main__":
    sample_cats_zero_shot()
