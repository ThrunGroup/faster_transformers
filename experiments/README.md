# Replace the cached model file with the latest model file

/sailhome/zgh23/.cache/huggingface/modules/transformers_modules/thrunlab/sparse_mistral_50p_no_adapter/3948f9d5a40579888c347b0ba6ad60c65df5833a

# Benchmark the decoding phase
Only Support B=1, bm=1 NOW!!!
1. Change the output filenames in `time_stop_util.py`
2. Replace the `~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils.py` with `time_stop_util.py`
3. Run `bench_sparse_inference.py`
- Method 0: Dense: ~ 33 tokens/s
- Method 1: CUDA (sparse Wup + dense Wdown): ~ 35 tokens/s
- Method 2: Triton (sparse Wup + sparse Wdown): ~ 39 tokens/s
4. Replace the `~/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/generation/utils.py` with `ori_util.py`

# Benchmark the MLP
1. Go to `/sailhome/zgh23/flash_gemv/bench`
2. `./final_profile.sh`
3. WIP: Tune the CUDA's parameter

# Get the aggregated sparsity
Run `stats_sparse_inference.py`
1. Across batch 
2. Across beam

# Get the latency ratio of MHA vs. FFN
1. Replace `/sailhome/zgh23/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/models/mistral/modeling_mistral.py` with `time_modeling_mistral.py`

2. Run `bench_sparse_inference.py` with Method 0

3. Replace `/sailhome/zgh23/miniconda3/envs/svd/lib/python3.11/site-packages/transformers/models/mistral/modeling_mistral.py` with `ori_modeling_mistral.py`

Result (Batch = 1, Beam = 1, Dense + FlashAttention2):
- Prefill (~100 tokens): 0.36 ms | 0.64 ms
- Decode (1 token): 0.5 ms | 0.3 ms