import torch
import argparse
import time
from utils import benchmark, ACT2FN, benchmark_pred, Config, MistralSparseMLP, degrad_tensor
from dejavu import mistral_mlp_sparse, mistral_mlp_sparse_fuse, mistral_mlp_partial_sparse
from kernels import matmul_triton, matmul_triton_bias, matmul_triton_bias_where, mistral_mlp_sparse_flag, matmul_triton_bias_fp32, mistral_mlp_sparse_direct_flag
import torch_index_scatter
# @torch.compile
# def svd_where(x, Wu, Wsv, Bsv, svd_activation, threshold, K):
#     svd_act_fn = torch.nn.SiLU()
#     return (svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv)[0] >= threshold).nonzero().squeeze(1)[:K]


def mistral_mlp_pred_where_idx(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, idx):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    eps = 7e-5
    threshold = THS + eps
    x_1 = matmul_triton(x, Wu)
    x_2 = matmul_triton_bias_where(x_1, Wsv, Bsv, threshold, svd_activation)
    output = mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx, activation) 
    return output

def mistral_mlp_pred_where_idx_torch(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, idx):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    threshold = THS
    svd_act_fn = ACT2FN[svd_activation]
    x_1 = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv) > threshold
    output = mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx, activation) 
    return output

def mistral_mlp_pred_where(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    eps = 7e-5
    threshold = THS + eps
    # svd_act_fn = ACT2FN[svd_activation]
    x_1 = matmul_triton(x, Wu)
    # print(x_1.shape)
    # x_2 = svd_act_fn((matmul_triton(x_1, Wsv) + Bsv)) >= threshold
    x_2 = matmul_triton_bias_where(x_1, Wsv, Bsv, threshold, svd_activation)
    # x_2 = matmul_triton_bias(x_1, Wsv, Bsv, svd_activation) > threshold
    # idx = (x_2.squeeze(0) >= threshold).nonzero() # The output of torch.where is a tuple
    # print(x_2)
    # (_, idx) = torch.nonzero(x_2, as_tuple=True)
    # print(idx.shape)
    # return mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx[:K], activation) 
    output = mistral_mlp_sparse_flag(x, Wgate, Wup, Wdownt, x_2, activation)
    # output = mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx[:K], activation) 
    # print(output)
    return output

def mistral_mlp_pred_where_torch(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    threshold = THS
    svd_act_fn = ACT2FN[svd_activation]
    x_1 = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv) > threshold
    # x_1 = x > threshold
    (_, idx) = torch.nonzero(x_1, as_tuple=True)
    output = mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx, activation) 
    # print(output)
    return output

def mistral_mlp_pred_sparse(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    svd_act_fn = ACT2FN[svd_activation]
    probs = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv)[0]
    topk_idcs = torch.argsort(probs)[:K]
    topk_idcs, _ = torch.sort(topk_idcs)
    return mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx, activation)

def mistral_mlp_torch(x, Wgate, Wup, Wdown, idx, activation="relu"):
    """
    x: (B, hidden_size)
    Wgate: (hidden_size, intermediate_size)
    Wup: (hidden_size, intermediate_size)
    Wdown: (intermediate_size, hidden_size)
    """
    act_fn = ACT2FN[activation]
    return torch.matmul(act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup), Wdown)

def mistral_mlp_thrust(x, Wu, Wsv, Bsv, orange, Wgate, Wup, Wdownt, idx, K):
    svd_activation = "swish"
    threshold = 0.1
    feature = matmul_triton_bias_fp32(matmul_triton(input, Wu), Wsv, svd_activation, Bsv).squeeze(0)
    threshold_tensor = torch.Tensor([threshold]).to(torch.float32)
    nzid = torch_index_scatter.fast_nonzero(feature, orange, idx, threshold_tensor)
    output = mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx[:nzid], svd_activation) 
    return output

def mistral_direct_mlp(x, Wgate, Wup, Wdownt, threshold, activation):
    act_fn = ACT2FN[activation]
    x_1 = act_fn(torch.matmul(x, Wgate))
    (_,idx) = torch.nonzero(x_1 > threshold, as_tuple=True)
    return mistral_mlp_partial_sparse(x, x_1[:,idx], Wup, Wdownt, idx)

def mistral_direct_mlp_triton(x, Wgate, Wup, Wdownt, threshold, activation):
    act_fn = ACT2FN[activation]
    x_1 = act_fn(torch.matmul(x, Wgate))
    return mistral_mlp_sparse_direct_flag(x, x_1, Wup, Wdownt, x_1 > threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile mare upperbound')
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--compress', type=float, default=4)
    parser.add_argument('--filename', type=str, default="./perfs/nosvd.csv")
    parser.add_argument('--record', type=bool)
    parser.add_argument('--ths', type=float, default=-0.1)
    parser.add_argument('--fths', type=float, default=-0.1)
    args = parser.parse_args()

    torch.manual_seed(0)
    THS = args.ths
    full_ths = args.fths
    hidden_size = 4096
    intermediate_size = 14336
    rank = args.rank
    K = int(intermediate_size / args.compress)
    # K = 10209 # Cannot control the actual nonzero in the real svd-mlp
    B = 1
    hidden_act = "swish"
    svd_act = "swish"
    input = torch.randn((B, hidden_size), dtype=torch.float16).cuda()

    # gate_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    # gate_t = gate_full.T.contiguous()
    # up_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    # up_t = up_full.T.contiguous()
    # down_full = torch.randn((intermediate_size, hidden_size), dtype=torch.float16).cuda()
    # u_proj = torch.randn((hidden_size, rank), dtype=torch.float16).cuda()
    # sv_proj = torch.randn((rank, intermediate_size), dtype=torch.float16).cuda()
    # sv_bias = torch.randn((intermediate_size), dtype=torch.float16).cuda()

    config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish', 'swish')
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    # mlp(input)
    input = degrad_tensor(input)
    gate_t = degrad_tensor(mlp.gate_proj.weight)
    up_t = degrad_tensor(mlp.up_proj.weight)
    # down_full = degrad_tensor(mlp.down_proj.weight).T
    down_full = degrad_tensor(mlp.down_proj.weight).T.contiguous()
    u_proj = degrad_tensor(mlp.u_proj.weight).T.contiguous()
    sv_proj = degrad_tensor(mlp.sv_proj.weight).T.contiguous()
    sv_bias = degrad_tensor(mlp.sv_proj.bias)

    svd_act_fn = ACT2FN[svd_act]
    flag = svd_act_fn(torch.matmul(torch.matmul(input, u_proj),sv_proj) + sv_bias) > THS
    (_, idx) = torch.nonzero(flag, as_tuple=True)
    print(idx.shape)
    K = idx.shape[0]
    gate = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    up = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    down = torch.randn((K, hidden_size), dtype=torch.float16).cuda()
    gate_full = gate_t.T.contiguous()
    up_full = up_t.T.contiguous()

    flag_real = svd_act_fn(torch.matmul(input, gate_full)) > full_ths
    (_, idx_real) = torch.nonzero(flag_real, as_tuple=True)
    print(idx_real.shape)
    k_real = idx_real.shape[0]

    # perm = torch.randperm(intermediate_size)[:K]
    # idx, _ = torch.sort(perm)
    # idx = idx.cuda()

    # output = mistral_direct_mlp(input, gate_full, up_t, down_full, full_ths, hidden_act)
    # print(output)
    # triton_output = mistral_direct_mlp_triton(input, gate_full, up_t, down_full, full_ths, hidden_act)
    # print(output)
    # diff_indices = torch.nonzero(torch.abs(output - triton_output) > 0.0005)
    # print(diff_indices)
    # print(output[0][448])
    # print(triton_output[0][448])
    # output = mistral_mlp_pred_where_torch(input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    # triton_output = mistral_mlp_pred_where(input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    # print(torch.nonzero(torch.abs(output - triton_output) > 0.0001))

    time.sleep(2)

    time_dense_torch = benchmark(mistral_mlp_torch, input, gate_full, up_full, down_full, idx, hidden_act)
    time_torch = benchmark(mistral_mlp_torch, input, gate, up, down, idx, hidden_act)
    time_dejavu = benchmark(mistral_mlp_sparse_fuse, input, gate_t, up_t, down_full, idx, hidden_act)
    time_pred_dejavu = benchmark_pred(mistral_mlp_pred_sparse, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    time_pred_where_torch = benchmark_pred(mistral_mlp_pred_where_torch, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    time_pred_where = benchmark_pred(mistral_mlp_pred_where, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    time_pred_where_idx_torch = benchmark_pred(mistral_mlp_pred_where_idx_torch, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, idx)
    time_pred_where_idx = benchmark_pred(mistral_mlp_pred_where_idx, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, idx)
    idx = torch.zeros(intermediate_size, dtype=torch.int64).cuda()
    orange = torch.arange(intermediate_size, dtype=torch.int64).cuda()
    time_thrust = benchmark_pred(mistral_mlp_thrust, input, u_proj, sv_proj, sv_bias, orange, gate_t, up_t, down_full, idx, K)
    time_direct = benchmark(mistral_direct_mlp, input, gate_full, up_t, down_full, full_ths, hidden_act)
    time_direct_triton = benchmark(mistral_direct_mlp_triton, input, gate_full, up_t, down_full, full_ths, hidden_act)
    with open(args.filename, "a") as f:
        if args.record:
            f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(K, k_real, time_dense_torch, time_torch, time_dejavu, time_pred_dejavu, time_pred_where_torch, time_pred_where, time_pred_where_idx_torch, time_pred_where_idx, time_thrust, time_direct, time_direct_triton))
        print("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(K, k_real, time_dense_torch, time_torch, time_dejavu, time_pred_dejavu, time_pred_where_torch, time_pred_where, time_pred_where_idx_torch, time_pred_where_idx, time_thrust, time_direct, time_direct_triton))
