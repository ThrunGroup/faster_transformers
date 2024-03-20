import argparse
import torch.nn as nn
import torch
from utils import benchmark
from kernels import mistral_mlp_sparse_direct_flag, mistral_mlp_sparse_direct_index_2d, gemv_triton, mistral_mlp_sparse_direct_noflag
from dejavu import mistral_mlp_partial_sparse
import flash_gemv

def mistral_direct_mlp_optimal(x, Wgate, Wup, Wdownt, threshold, act_fn):
    return torch.matmul((act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup)), Wdownt)

def mistral_direct_baseline(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_partial_sparse(x, x_1[0][idx], Wup, Wdownt, idx)

def mistral_direct_mlp(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    return mistral_mlp_sparse_direct_index_2d(x, x_1[0][idx], Wup, Wdownt, idx)

def mistral_direct_mlp_triton(x, Wgate, Wup, Wdownt, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    return mistral_mlp_sparse_direct_flag(x, x_1, Wup, Wdownt, torch.abs(x_1) > threshold)

def mistral_direct_mlp_cuda(x, Wgate, Wup, Wdown, value, flag):
    value = torch.zeros((1, intermediate_size), dtype=torch.float16, device=torch.device('cuda:0'))
    flag = torch.zeros((intermediate_size), dtype=torch.bool, device=torch.device('cuda:0'))
    # flash_gemv.fuse_gemv_flag_local(x, Wgate, value, flag, THS)
    # flash_gemv.fuse_flag_gemv_local(x, Wup, value, flag)
    # return gemv_triton(x, Wdown)
    flash_gemv.fuse_gemv_flag(x, Wgate, value, flag, THS)
    return mistral_mlp_sparse_direct_noflag(x, value, Wup, Wdown, flag)
    

def mistral_direct_mlp_allfuse(x, Wgate, Wup, Wdownt, threshold, THS):
    return flash_gemv.flag_gemv_gemv_inner(x, Wgate, Wup, Wdownt, threshold)

def get_k_mistral_direct_mlp(x, Wgate, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=True)
    # print(idx)
    # print(idx.shape)
    # print(x_1.shape)
    # print(x_1[0][idx])
    return idx.shape[-1]

def get_k_mistral_direct_mlp_cuda(x, Wgate, value, flag):
    flash_gemv.fuse_gemv_flag(x, Wgate, value, flag, THS)
    (idx,) = torch.nonzero(flag, as_tuple=True)
    # print(idx)
    return idx.shape[-1]


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
    rank = 128
    K = 3584
    B = 1
    hidden_act = "swish"
    svd_act = "swish"
    input = torch.rand((B, hidden_size)).half().cuda() - 0.5
 
    gate = torch.rand(4096, 14336).half().cuda() - 0.5
    up = torch.rand(4096, 14336).half().cuda() - 0.5
    down = torch.rand(14336, 4096).half().cuda() - 0.5
    gate_t = gate.t().contiguous()
    up_t = up.t().contiguous()
    down_t = down.t().contiguous()
    value = torch.zeros((B, intermediate_size)).half().cuda()
    flag = torch.zeros((intermediate_size), dtype=torch.bool).cuda()
    
    k_fp32 = get_k_mistral_direct_mlp(input, gate, args.ths, nn.SiLU())
    k_fp16 = get_k_mistral_direct_mlp_cuda(input, gate_t, value, flag)
    # torch.cuda.synchronize()
    
    # mistral_direct_mlp(input, gate, up_t, down, args.ths, hidden_act)
    time_baseline = benchmark(mistral_direct_baseline, input, gate, up_t, down, args.ths, nn.SiLU())
    time_direct = benchmark(mistral_direct_mlp, input, gate, up_t, down, args.ths, nn.SiLU())
    time_direct_triton = benchmark(mistral_direct_mlp_triton, input, gate, up_t, down, args.ths, nn.SiLU())
    time_all_fuse = benchmark(mistral_direct_mlp_allfuse, input, gate_t, up_t, down_t, args.ths, THS)
    time_direct_cuda = benchmark(mistral_direct_mlp_cuda, input, gate_t, up_t, down_t, value, flag)
    time_dense_mlp = benchmark(mistral_direct_mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())

    gate = torch.rand(4096, k_fp32).half().cuda() - 0.5
    up = torch.rand(4096, k_fp32).half().cuda() - 0.5
    down = torch.rand(k_fp32, 4096).half().cuda() - 0.5
    time_direct_optimal = benchmark(mistral_direct_mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())
    print(f"[{k_fp32},{k_fp16}]Dense: {time_dense_mlp}, Baseline: {time_baseline}, Direct: {time_direct}, Triton: {time_direct_triton}, CUDA: {time_direct_cuda}, AllFuse: {time_all_fuse}, Optimal: {time_direct_optimal}")
    # with open(args.filename, "a") as f:
    #     print(f"{k_fp32},{k_fp16},{time_dense_mlp},{time_baseline},{time_direct},{time_direct_triton},{time_direct_cuda},{time_all_fuse},{time_direct_optimal}", file=f)

