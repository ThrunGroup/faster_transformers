import argparse
import time
import torch
from compare_svd_mlp import benchmark_pred, mistral_pred_select_torch, mistral_pred_sparse, benchmark_lrank, lrank_mlp, benchmark_topk, single_topk
from compute_mare import Config, MistralSparseMLP, degrad_tensor, benchmark, mistral_mlp_torch, mistral_mlp_select_torch
from dejavu import mistral_mlp_sparse

def profile_pred(args):
    hidden_size = 4096
    intermediate_size = 14336
    rank = args.rank
    K = int(intermediate_size // args.compress)
    config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish', 'relu')
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    B = 1
    input = torch.randn((B, config.hidden_size), dtype=torch.float16).cuda()
    mlp(input)
    input = degrad_tensor(input)
    gate_proj = degrad_tensor(mlp.gate_proj.weight)
    up_proj = degrad_tensor(mlp.up_proj.weight)
    down_proj = degrad_tensor(mlp.down_proj.weight)
    u_proj = degrad_tensor(mlp.u_proj.weight)
    sv_proj = degrad_tensor(mlp.sv_proj.weight)
    sv_bias = degrad_tensor(mlp.sv_proj.bias)
    perm = torch.randperm(intermediate_size)[:K]
    idx, _ = torch.sort(perm)
    topk_idcs = idx.cuda()
    
    u_proj_t = u_proj.T
    sv_proj_t = sv_proj.T
    gate_proj_t = gate_proj.T
    up_proj_t = up_proj.T
    down_proj_t = down_proj.T

    mistral_pred_select = torch.compile(mistral_pred_select_torch)
    mistral_mlp = torch.compile(mistral_mlp_torch)
    mistral_mlp_select = torch.compile(mistral_mlp_select_torch)
    lrank_mlp_p = torch.compile(lrank_mlp)
    single_topk_p = torch.compile(single_topk)
    time.sleep(2)
    time_torch = benchmark_pred(mistral_pred_select, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj_t, up_proj_t, down_proj_t, config.hidden_act, K)
    time_triton = benchmark_pred(mistral_pred_sparse, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj, up_proj, down_proj_t, config.hidden_act, K)
    time_dense_torch = benchmark(mistral_mlp, input, gate_proj_t, up_proj_t, down_proj_t, topk_idcs, config.hidden_act)
    time_lrank = benchmark_lrank(lrank_mlp_p, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act)
    time_select_torch = benchmark(mistral_mlp_select, input, gate_proj_t, up_proj_t, down_proj_t, topk_idcs, config.hidden_act)
    time_select_triton = benchmark(mistral_mlp_sparse, input, gate_proj, up_proj, down_proj_t, topk_idcs, config.hidden_act)
    time_topk = benchmark_topk(single_topk_p, mlp.probs, K)
    with open(args.filename, "a") as f:
        if args.record:
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(rank, K, time_torch, time_triton, time_dense_torch, time_lrank, time_select_torch, time_select_triton, time_topk))
        print("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(rank, K, time_torch, time_triton, time_dense_torch, time_lrank, time_select_torch, time_select_triton, time_topk))

# def profile_lrank_mlp(args):
#     hidden_size = 4096
#     intermediate_size = 14336
#     rank = args.rank
#     K = int(intermediate_size // args.compress)
#     config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish', 'relu')
#     mlp = MistralSparseMLP(config).to(torch.float16).cuda()
#     B = 1
#     input = torch.randn((B, config.hidden_size), dtype=torch.float16).cuda()
#     mlp(input)
#     input = degrad_tensor(input)
#     u_proj = degrad_tensor(mlp.u_proj.weight)
#     sv_proj = degrad_tensor(mlp.sv_proj.weight)
#     sv_bias = degrad_tensor(mlp.sv_proj.bias)
#     u_proj_t = u_proj.T
#     sv_proj_t = sv_proj.T
#     time.sleep(2)
#     time_torch = benchmark_lrank(lrank_mlp, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile mare')
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--compress', type=int, default=4)
    parser.add_argument('--filename', type=str, default="/home/zhongming/genghan/mare/perfs/mares-fuse.csv")
    parser.add_argument('--record', type=bool)
    args = parser.parse_args()

    profile_pred(args)
    # print("Sparse Torch: {:.2f} ms".format(time_torch))