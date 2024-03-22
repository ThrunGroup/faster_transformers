from experiments.models.sparse_mistral.sparsification_sftt import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
    get_sparse_mistral_config,
    print_dead_neuron_stats,
    activate_stats,
)
from experiments.mistral_instruct_tuning import prepare_sparse_model
from torch.utils.data import DataLoader

import os
import torch

from transformers import (
    MistralConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    MistralForCausalLM,
    Trainer,
    TrainingArguments,
)
from data.get_dataset import get_dataset
from utils.constants import MISTRAL_7B, REFINED_WEB


def sparse_inference():
    path = "thrunlab/sparse_mistral_50p_no_adapter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset_type = REFINED_WEB

    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=1024)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset(
        train_size=50,
    )
    data_collator = dataset.get_data_collator()

    trainer_config = TrainingArguments(
        output_dir="/scr/jay/hello",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        push_to_hub=True,
        hub_model_id="thrunlab/sparse_mistral_50p_no_adapter",
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    # trainer.push_to_hub()
    activate_stats(model, False)
    print(trainer.evaluate())
    print_dead_neuron_stats(model,)

    dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    counts = 0
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(
                device
            )  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500
            print(max_lengths)

            attention_mask = batch["attention_mask"].to(
                device
            )  # Move attention mask to the device
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode generated tokens to texts
            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            counts += 1
            if counts <= 1:
                for idx in range(len(input_ids)):
                    ids = input_ids[idx]
                    # Decode the input_ids to a string
                    tokens_str = tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"Tokens before generation: {tokens_str}")
                    generated_text = generated_texts[idx]
                    print(f"Generated: {generated_text}\n")


if __name__ == "__main__":
    sparse_inference()
