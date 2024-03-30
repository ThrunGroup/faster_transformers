from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaConfig,
)
from torch.utils.data import DataLoader
from utils.constants import LLAMA, REFINED_WEB
from experiments.models.sparse_silu.ugly_utils import *
from experiments.data.get_dataset import get_dataset
import argparse
from scipy.stats.mstats import gmean

def prepare_sparse_model():
    base_model_name = "/matx/u/zgh23/ckpt/sparse_llama_7b_hf2_refined_web_50p_no_adapter_2600steps"
    # base_model_name = "meta-llama/Llama-2-7b"
    BaseConfig = LlamaConfig
    SparseConfig = SparseLlamaConfig
    SparseCausalLM = SparseLlamaForCausalLM
    SparseConfig.register_for_auto_class()
    SparseCausalLM.register_for_auto_class("AutoModelForCausalLM")

    config = BaseConfig.from_pretrained(base_model_name)
    config = get_sparse_config(config)
    config.use_cache = True
    model = SparseCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.float32,
        # attn_implementation="flash_attention_2",
        device_map='auto',
    )
    model.config_class = SparseConfig
    apply_sparse_silu_mlp(model, model.config, use_sparse_regularization=False)
    enable_sparse_silu(model)
    model.config.use_sparse_predictor = False
    model.config.use_relu = False
    model.config.use_sparse_model = True
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer

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

def benchmark_decode(B, bm, method, gen_len):
    # Replace ~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils
    # with ./time_stop_util.py before running the code!!
    act_hist_path = "/matx/u/zgh23/act_hist/sparse_llama_7b_hf2/refined_web.pt"
    device = torch.device("cuda")
    dataset_type = REFINED_WEB
    model_type = LLAMA
    model, tokenizer = prepare_sparse_model()
    activate_stats(model)
    load_act_hist(model, act_hist_path)
    set_sparse_threshold(model, 0.5)
    deactivate_stats(model)
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)

    for m in model.model.layers:
        m.mlp.is_profile = True
        # m.mlp.use_flash_gemv = True
        if method == 0:
            m.mlp.sp_method = 0
            print(m.mlp.down_proj.weight.dtype)
        elif method == 1:
            m.mlp.sp_method = 1
        elif method == 2:
            m.mlp.sp_method = 2
            m.mlp.wdown_t = m.mlp.down_proj.weight.t().contiguous()
            print(m.mlp.down_proj.weight.dtype)
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + gen_len   
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=max_lengths,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=True, 
                    temperature=0.7, 
                    num_beams=bm
                ) 
            except RuntimeError as _:
                print("Time interval is too short!")
                continue


def benchmark_generation(B, bm, method):
    # Replace ~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils
    # with ./ori_util.py before running the code!!
    act_hist_path = "/matx/u/zgh23/act_hist/sparse_llama_7b_hf2/refined_web.pt"
    device = torch.device("cuda")
    dataset_type = REFINED_WEB
    model_type = LLAMA
    model, tokenizer = prepare_sparse_model()
    activate_stats(model)
    load_act_hist(model, act_hist_path)
    set_sparse_threshold(model, 0.5)
    deactivate_stats(model)
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()

    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)
    for m in model.model.layers:
        m.mlp.is_profile = True
        # m.mlp.use_flash_gemv = True
        if method == 0:
            m.mlp.sp_method = 0
            print(m.mlp.down_proj.weight.dtype)
        elif method == 1:
            m.mlp.sp_method = 1
        elif method == 2:
            m.mlp.sp_method = 2
            m.mlp.wdown_t = m.mlp.down_proj.weight.t().contiguous()
            print(m.mlp.down_proj.weight.dtype)
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    warmup = 5
    repeat = 10
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500

            throughputs, elapsed_times, gen_lengths = benchmark_sentence(model, input_ids, max_lengths, tokenizer.eos_token_id, 0.7, bm, warmup, repeat)
            for (tp, et, gl) in zip(throughputs, elapsed_times, gen_lengths):
                print("Throughput: {:.2f} tokens/s; Elapsed time: {} ms; Generation Length: {}".format(tp, et, gl))
            print("Average Throughput: {:.2f} tokens/s".format(gmean(throughputs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--bm", type=int, default=1)
    parser.add_argument("--gen", type=int, default=501)
    parser.add_argument("--method", type=int, default=0)
    args = parser.parse_args()
    # benchmark_generation()
    benchmark_decode(args.B, args.bm, args.method, args.gen)