from experiments.models.sparse_mistral.sparsification_sftt import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
    get_sparse_mistral_config,
)
from experiments.mistral_instruct_tuning import prepare_sparse_model
from torch.utils.data import DataLoader

import argparse
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


def sparse_inference(B, bm, gen_len):
    path = "thrunlab/sparse_mistral_50p_no_adapter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # low_cpu_mem_usage=True,
        device_map='auto',
    ).to('cuda')
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset_type = REFINED_WEB

    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=1000)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset(
        train_size=50,
    )
    data_collator = dataset.get_data_collator()

    for m in model.model.layers:
        m.mlp.activate_stats(False)
        m.mlp.agg_path = f"/sailhome/zgh23/kernel/faster_transformers/experiments/agg_data/agg_sparsity_{B}_{bm}_tmp.csv"
        m.mlp.dead_path = f"/sailhome/zgh23/kernel/faster_transformers/experiments/agg_data/dead_percentage_{B}_{bm}_tmp.csv"
        m.mlp.agg_snapshot = f"/sailhome/zgh23/kernel/faster_transformers/experiments/agg_data/agg_sparsity_{B}_{bm}.csv"
        m.mlp.dead_snapshot = f"/sailhome/zgh23/kernel/faster_transformers/experiments/agg_data/dead_percentage_{B}_{bm}.csv"
        print(m.mlp.gate_proj.weight.dtype)


    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)

    counts = 0
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    print(len(dataloader))
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(
                device
            )  # Move input tensors to the device
            max_lengths = input_ids.size(1) + gen_len
            print(max_lengths)

            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True, 
                temperature=0.7, 
                num_beams=bm
            )

            # Decode generated tokens to texts
            # generated_texts = [
            #     tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            # ]
            # counts += 1
            # if counts <= 1:
            #     for idx in range(len(input_ids)):
            #         ids = input_ids[idx]
            #         # Decode the input_ids to a string
            #         tokens_str = tokenizer.decode(ids, skip_special_tokens=True)
            #         print(f"Tokens before generation: \n {tokens_str}")
            #         generated_text = generated_texts[idx]
            #         print(f"Generated: \n {generated_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--bm", type=int, default=8)
    parser.add_argument("--gen", type=int, default=500)
    args = parser.parse_args()
    sparse_inference(args.B, args.bm, args.gen)
