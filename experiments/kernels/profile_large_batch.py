import argparse
import torch.nn as nn
import torch
from utils import benchmark
import flash_gemv

def mlp_optimal(x, Wgate, Wup, Wdown, threshold, act_fn):
    return torch.matmul((act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup)), Wdown)

def cuda_batch(x, Wgate, Wup, Wdown, threshold, act_fn):
    return flash_gemv.flag_gemv_gemv_inner(x, Wgate, Wup, Wdown, threshold)

def get_k_mistral_direct_mlp(x, Wgate, threshold, act_fn):
    x_1 = act_fn(torch.matmul(x, Wgate))
    idx = torch.nonzero(torch.abs(x_1) > threshold, as_tuple=False)
    return idx.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ths", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    hidden_size = 4096
    intermediate_size = 14336
    B = args.batch
    input = torch.rand((B, 1, hidden_size)).half().cuda() - 0.5
    gate = torch.rand(hidden_size, intermediate_size).half().cuda() - 0.5
    up = torch.rand(hidden_size, intermediate_size).half().cuda() - 0.5
    down = torch.rand(intermediate_size, hidden_size).half().cuda() - 0.5
    gate_t = gate.t().contiguous()
    up_t = up.t().contiguous()
    down_t = down.t().contiguous()
    mlp_optimal(input, gate, up, down, args.ths, nn.SiLU())
    cuda_batch(input, gate_t, up_t, down_t, args.ths, nn.SiLU())
 #   time_direct_optimal = benchmark(mlp_optimal, input, gate, up, down, args.ths, nn.SiLU())
 #   time_cuda_batch = benchmark(cuda_batch, input, gate_t, up_t, down_t, args.ths, nn.SiLU())

#    print(f"Batch: {B}; Dense: {time_direct_optimal}; Sparse: {time_cuda_batch}")
