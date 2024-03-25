from experiments.models.sparse_mistral.sparsification_sftt import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
    get_sparse_mistral_config,
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
import scipy.stats.mstats as mstats
from transformers.generation import utils

def benchmark_sentence(model, input_ids, max_lengths, pad_token_id, temperature, num_beams, warmup, repeat):
    for _ in range(warmup):
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_lengths,
            pad_token_id=pad_token_id,
            use_cache=True,
            do_sample=True, 
            temperature=temperature, 
            num_beams=num_beams
        )
    elapsed_times = []
    throughputs = []
    gen_lengths = []
    for _ in range(repeat):
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
        cache.zero_()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_lengths,
            pad_token_id=pad_token_id,
            use_cache=True,
            do_sample=True, 
            temperature=temperature, 
            num_beams=num_beams
        )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        gen_length = output_ids.shape[-1] - input_ids.shape[1]
        elapsed_times.append(elapsed_time_ms)
        throughputs.append(gen_length / (elapsed_time_ms / 1000.0))
        gen_lengths.append(gen_length)
    return throughputs, elapsed_times, gen_lengths

def benchmark_decode():
    # Replace ~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils
    # with ./time_stop_util.py before running the code!!
    path = "thrunlab/sparse_mistral_50p_no_adapter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto',
    ).to('cuda')
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset_type = REFINED_WEB

    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset(
        train_size=50,
    )
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

    for m in model.model.layers:
        m.mlp.is_profile = True
        # m.mlp.use_flash_gemv = True
        m.mlp.use_flash_gemv = False
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500   

            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True, 
                temperature=0.7, 
                num_beams=1
            ) 

def benchmark_generation():
    # Replace ~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils
    # with ./ori_util.py before running the code!!
    path = "thrunlab/sparse_mistral_50p_no_adapter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto',
    ).to('cuda')
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset_type = REFINED_WEB

    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset(
        train_size=50,
    )
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    warmup = 5
    repeat = 10
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500

            throughputs, elapsed_times, gen_lengths = benchmark_sentence(model, input_ids, max_lengths, tokenizer.eos_token_id, 0.7, 8, warmup, repeat)
            for (tp, et, gl) in zip(throughputs, elapsed_times, gen_lengths):
                print("Throughput: {:.2f} tokens/s; Elapsed time: {} ms; Generation Length: {}".format(tp, et, gl))
            print("Average Throughput: {:.2f} tokens/s".format(mstats.gmean(throughputs)))

if __name__ == "__main__":
    # benchmark_generation()
    benchmark_decode()