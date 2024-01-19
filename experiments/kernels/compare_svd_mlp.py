import time
import torch
from compute_mare import mistral_mlp_sparse_torch, ACT2FN, mistral_mlp_torch, Config, MistralSparseMLP, benchmark, degrad_tensor
from dejavu import mistral_mlp_sparse

def mistral_pred_sparse(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
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
    intermediate_size, _ = Wgate.shape
    svd_act_fn = ACT2FN[svd_activation]
    probs = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv)[0]
    _, topk_idcs = torch.topk(probs, k=K, dim=-1)
    topk_idcs, _ = torch.sort(topk_idcs)
    return mistral_mlp_sparse(x, Wgate, Wup, Wdownt, topk_idcs, activation)

def mistral_pred_sparse_torch(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
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
    intermediate_size, _ = Wgate.shape
    svd_act_fn = ACT2FN[svd_activation]
    probs = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv)[0]
    _, topk_idcs = torch.topk(probs, k=K, dim=-1)
    return mistral_mlp_sparse_torch(x, Wgate, Wup, Wdownt, topk_idcs, activation)

def benchmark_pred(fn, input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K, warmup=20, rep=80, quantiles=None, fast_flush=True, return_mode="mean"):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    torch.cuda.synchronize()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn(input, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdown, activation, K)
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()

if __name__ == "__main__":
    hidden_size = 4096
    intermediate_size = 14336
    rank = 64
    K = int(intermediate_size // 16)
    config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish', 'relu')
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    # B = 16
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
    
    u_proj_t = u_proj.T
    sv_proj_t = sv_proj.T
    gate_proj_t = gate_proj.T
    up_proj_t = up_proj.T
    down_proj_t = down_proj.T
    time.sleep(2)
    mistral_pred_sparse_torch = torch.compile(mistral_pred_sparse_torch)
    mistral_mlp_torch = torch.compile(mistral_mlp_torch)
    output = mistral_pred_sparse_torch(input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj_t, up_proj_t, down_proj_t, config.hidden_act, K)
    # print(output)
    test_output = mistral_pred_sparse(input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj, up_proj, down_proj_t, config.hidden_act, K)
    # print(test_output)
    # assert torch.allclose(output, test_output, rtol=0.01)
    print(torch.nonzero(torch.abs(output - test_output) > 0.0005))
    time_torch = benchmark_pred(mistral_pred_sparse_torch, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj_t, up_proj_t, down_proj_t, config.hidden_act, K)
    print("Sparse Torch: {:.2f} ms".format(time_torch))
    
    time_triton = benchmark_pred(mistral_pred_sparse, input, u_proj_t, sv_proj_t, sv_bias, config.svd_act, gate_proj, up_proj, down_proj_t, config.hidden_act, K)
    # time.sleep(2)
    print("Sparse Triton: {:.2f} ms".format(time_triton))
    time_dense_torch = benchmark(mistral_mlp_torch, input, gate_proj_t, up_proj_t, down_proj_t, mlp.topk_idcs, config.hidden_act)
    print("Dense Torch: {:.2f} ms".format(time_dense_torch))