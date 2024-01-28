import argparse
import torch
from utils import MistralSparseMLP, ACT2FN, Config, degrad_tensor
from dejavu import mistral_mlp_sparse, gather_gemv, gather_transposed_gemv
from kernels import matmul_triton, matmul_triton_bias_where, mistral_mlp_sparse_flag, gather_gemv2_flag, gather_transposed_gemv_flag, matmul_triton_bias_fp32
import torch_index_scatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile mare upperbound')
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--compress', type=float, default=4)
    parser.add_argument('--filename', type=str, default="./perfs/dejavu.csv")
    parser.add_argument('--record', type=bool)
    args = parser.parse_args()

    torch.manual_seed(0)
    hidden_size = 4096
    intermediate_size = 14336
    rank = args.rank
    K = int(intermediate_size / args.compress)
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
    mlp(input)
    input = degrad_tensor(input)
    gate_t = degrad_tensor(mlp.gate_proj.weight)
    up_t = degrad_tensor(mlp.up_proj.weight)
    # down_full = degrad_tensor(mlp.down_proj.weight).T
    down_full = degrad_tensor(mlp.down_proj.weight).T.contiguous()
    u_proj = degrad_tensor(mlp.u_proj.weight).T.contiguous()
    sv_proj = degrad_tensor(mlp.sv_proj.weight).T.contiguous()
    sv_bias = degrad_tensor(mlp.sv_proj.bias)

    gate = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    up = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    down = torch.randn((K, hidden_size), dtype=torch.float16).cuda()    
    perm = torch.randperm(intermediate_size)[:K]
    idx, _ = torch.sort(perm)
    idx = idx.cuda()
    threshold = -0.1

    feature = matmul_triton_bias_fp32(matmul_triton(input, u_proj), sv_proj, svd_act, sv_bias).squeeze(0)
    threshold_tensor = torch.Tensor([threshold]).to(torch.float32)
    flags = torch.zeros(feature.shape, dtype=torch.int64).cuda()
    print(feature.shape)
    print(feature.dtype)
    nzid = torch_index_scatter.fast_nonzero(feature, flags, threshold_tensor)
    print(flags.shape)
    
    
    svd_act_fn = ACT2FN[svd_act]
    x_1 = svd_act_fn(torch.matmul(torch.matmul(input, u_proj),sv_proj) + sv_bias) > threshold
    (_, idx) = torch.nonzero(x_1, as_tuple=True)
    print(idx.shape)
    x1 = gather_gemv(input, gate_t, idx, activation=hidden_act)
    x2 = gather_gemv(input, up_t, idx)
    x = x1 * x2
    print(x)
    output_1 = gather_transposed_gemv(x, down_full, idx)
    # print(output_1)
    assert torch.allclose(idx, flags[:nzid])


    # eps = 7e-5
    # threshold = -0.1 + eps
    # flags = x_1
    # x_fuse = gather_gemv2_flag(input, gate_t, up_t, flags, hidden_act)
    # print(x_fuse)
    # print(x_fuse.shape)
    # output_2 = gather_transposed_gemv_flag(x_fuse, down_full, flags)
    # print(output_2)
    # Assert flags and x_1 are equal, which are two Bool tensors
    # print(torch.nonzero(torch.abs(output_1 - output_2) > 0.0005))




