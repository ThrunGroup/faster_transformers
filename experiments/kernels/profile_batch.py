import argparse
import torch.nn as nn
import torch
from utils import benchmark
from kernels import mistral_mlp_sparse_direct_flag_batch, mistral_mlp_sparse_direct_flag
from flash_gemv import fuse_gemv_flag_batch, fuse_gemv_flag, fuse_gemv_flag_local, fuse_flag_gemv_local, atomic_gemv, flag_gemv_gemv_atomic, flag_gemv_gemv

def mistral_direct_mlp_optimal(x, Wgate, Wup, Wdownt, threshold, act_fn):
    return torch.matmul((act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup)), Wdownt)

def mistral_direct_mlp_cuda_all(x, Wgate, Wup, Wdownt, value, flag):
    # value = torch.zeros((B, 1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    # flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    output = torch.zeros((B, 1, hidden_size), dtype=torch.float16, device=torch.device('cuda:0'))
    # fuse_gemv_flag_local(x, Wgate, value, flag, THS)
    # fuse_flag_gemv_local(x, Wup, value, flag)
    # atomic_gemv(value, Wdownt, output, flag)
    # return output.half()
    # flag_gemv_gemv_atomic(x, Wgate, Wup, Wdownt, output, THS)
    flag_gemv_gemv(x, Wgate, Wup, Wdownt, output, THS)
    return output

def mistral_direct_mlp_cuda(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((B, 1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag_local(x, Wgate, value, flag, THS)
    return mistral_mlp_sparse_direct_flag_batch(x, value, Wup, Wdownt, flag)

def mistral_direct_mlp_cuda_single(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag_local(x, Wgate, value, flag, THS)
    return mistral_mlp_sparse_direct_flag(x, value, Wup, Wdownt, flag)

def bench_flag(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((B, 1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag_batch(x, Wgate, value, flag, THS)

def bench_flag_local(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((B, 1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag_local(x, Wgate, value, flag, THS)

def bench_flag_gemv_local(x, Wgate, Wup, Wdownt, value, flag):
    fuse_flag_gemv_local(x, Wup, value, flag)

def bench_atomic_gemv_local(x, Wgate, Wup, Wdownt, value, flag):
    output = torch.zeros((B, 1, hidden_size), dtype=torch.float32, device=torch.device('cuda:0'))
    # output = torch.zeros((B, 1, hidden_size), dtype=torch.float16, device=torch.device('cuda:0'))
    atomic_gemv(value, Wdownt, output, flag)
    # return output.half()
    return output.half()

def bench_single_flag(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag(x, Wgate, value, flag, THS)

def bench_single_flag_local(x, Wgate, Wup, Wdownt, value, flag):
    value = torch.zeros((1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    fuse_gemv_flag_local(x, Wgate, value, flag, THS)

def get_k_mistral_direct_mlp(x, Wgate, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    idx = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=False)
    return idx.shape[0]

def get_k_mistral_direct_mlp_cuda(x, Wgate, value, flag):
    # fuse_gemv_flag_batch(x, Wgate, value, flag, THS)
    fuse_gemv_flag_local(x, Wgate, value, flag, THS)
    idx = torch.nonzero(flag, as_tuple=False)
    # print(idx)
    return idx.shape[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ths", type=float, default=0.1)
    parser.add_argument("--cuths", type=float, default=0.1)
    parser.add_argument("--filename", type=str, default="mistral_direct.csv")

    args = parser.parse_args()

    torch.manual_seed(0)
    THS = args.cuths
    hidden_size = 4096
    intermediate_size = 14336
    B = 2
    input = torch.rand((B, 1, hidden_size)).half().cuda() - 0.5
    gate = torch.rand(hidden_size, intermediate_size).half().cuda() - 0.5
    up = torch.rand(hidden_size, intermediate_size).half().cuda() - 0.5
    down = torch.rand(intermediate_size, hidden_size).half().cuda() - 0.5
    gate_t = gate.t().contiguous()
    up_t = up.t().contiguous()
    value = torch.zeros((B, 1, intermediate_size)).half().cuda()
    flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool).cuda()

    k_fp32 = get_k_mistral_direct_mlp(input, gate, args.ths, nn.SiLU())
    k_fp16 = get_k_mistral_direct_mlp_cuda(input, gate_t, value, flag)

    time_direct_cuda_all = benchmark(mistral_direct_mlp_cuda_all, input, gate_t, up_t, down, value, flag)
    time_direct_cuda = benchmark(mistral_direct_mlp_cuda, input, gate_t, up_t, down, value, flag)
    
    time_direct_cuda_single = benchmark(mistral_direct_mlp_cuda_single, input[0], gate_t, up_t, down, value[0], flag[0])
    time_flag_batch = benchmark(bench_flag, input, gate_t, up_t, down, value, flag)
    time_flag_single = benchmark(bench_single_flag, input[0], gate_t, up_t, down, value[0], flag[0])

    k_mean = k_fp32 // B
    gate = torch.rand(hidden_size, k_mean).half().cuda() - 0.5
    up = torch.rand(hidden_size, k_mean).half().cuda() - 0.5
    down = torch.rand(k_mean, 4096).half().cuda() - 0.5
    time_direct_optimal = benchmark(mistral_direct_mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())
    time_direct_optimal_single = benchmark(mistral_direct_mlp_optimal, input[0], gate, up, down, args.ths, nn.SiLU())
    print(f"[{k_fp32},{k_fp16}] Optimal: {time_direct_optimal}, Optimal single: {time_direct_optimal_single}")
    print(f"Batch flag: {time_flag_batch}, Single flag: {time_flag_single}, Local Batch all: {time_direct_cuda}, Local Single all: {time_direct_cuda_single}, Batch all w/o triton: {time_direct_cuda_all}")
    # print(f"Batch compute: {time_direct_cuda - time_flag_batch}, Single compute: {time_direct_cuda_single - time_flag_single}")
        
    time_flag_batch = benchmark(bench_flag_local, input, gate_t, up_t, down, value, flag)
    time_flag_single = benchmark(bench_single_flag_local, input[0], gate_t, up_t, down, value[0], flag[0])
    value = torch.zeros((B, 1, intermediate_size)).half().cuda()
    flag = torch.zeros((B, 1, intermediate_size), dtype=torch.bool).cuda()
    fuse_gemv_flag_local(input, gate_t, value, flag, THS)
    time_flag_gemv_batch = benchmark(bench_flag_gemv_local, input, gate_t, up_t, down, value, flag)
    time_atomic_batch = benchmark(bench_atomic_gemv_local, input, gate_t, up_t, down, value, flag)
    print(f"Local Batch flag: {time_flag_batch}, Local Single flag: {time_flag_single}")
    # print(f"Local Batch compute: {time_direct_cuda - time_flag_batch}, Local Single compute: {time_direct_cuda_single - time_flag_single}")
    print(f"Local Batch gemv: {time_flag_gemv_batch}, Local Batch atomic: {time_atomic_batch}")