import torch
import torch.nn as nn

from dejavu import mistral_mlp_sparse

def dram(hidden_size, intermediate_size, rank, K, dsize):
    ori = hidden_size * intermediate_size * 3 * dsize
    svd_mat = hidden_size * rank + rank * intermediate_size + intermediate_size
    tran_mat = hidden_size * K * 3
    mare = (svd_mat + tran_mat) * dsize
    return ori, mare

def flops(BN, hidden_size, intermediate_size, rank, K, dsize):
    mm1 = BN * hidden_size * rank * 2
    mm2 = BN * rank * intermediate_size * 2 + BN * intermediate_size 
    mmup = BN * hidden_size * K * 2 + BN * hidden_size * K * 2 + BN * K
    mmdown = BN * intermediate_size * K 
    mare = mm1 + mm2 + mmup + mmdown

    mmup = BN * hidden_size * intermediate_size * 2 + BN * hidden_size * intermediate_size * 2 + BN * hidden_size
    mmdown = BN * hidden_size * intermediate_size
    ori = mmup + mmdown
    return ori, mare

def op_intensity():
    pass

ACT2FN = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "swish": nn.SiLU(),
}

class Config:
    def __init__(self, hidden_size, intermediate_size, rank, K, dtype, hidden_act='swish', svd_act='relu'):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.K = K
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.svd_act = svd_act

class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        l1 = self.gate_proj(x)
        print(l1.dtype)
        return self.down_proj(self.act_fn(l1) * self.up_proj(x))

class MistralSparseMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.rank
        self.K = config.K
        self.u_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.sv_proj = nn.Linear(self.rank, self.intermediate_size, bias=True)
        self.svd_act_fn = ACT2FN[config.svd_act]

    def forward(self, x):
        probs = self.svd_act_fn(self.sv_proj(self.u_proj(x)))
        self.probs = probs[0]
        _, topk_idcs = torch.topk(self.probs, k=self.K, dim=-1)
        self.topk_idcs, _ = torch.sort(topk_idcs) # [0] to make i compatible with deja vu
        mask = torch.zeros(
            self.intermediate_size,
            dtype=probs.dtype,
            device=probs.device,
        ).scatter(-1, self.topk_idcs, 1)
        masked_up = (self.act_fn(self.gate_proj(x)) * self.up_proj(x) * mask)
        return self.down_proj(masked_up)
def mistral_mlp_mask_torch(x, Wgate, Wup, Wdown, idx, activation="relu"):
    """
    x: (B, hidden_size)
    Wgate: (hidden_size, intermediate_size)
    Wup: (hidden_size, intermediate_size)
    Wdown: (intermediate_size, hidden_size)
    idx: (K,)
    """
    _, intermediate_size = Wgate.shape
    mask = torch.zeros(
            intermediate_size,
            dtype=x.dtype,
            device=x.device,
        ).scatter(-1, idx, 1)
    act_fn = ACT2FN[activation]
    return torch.matmul(act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup) * mask, Wdown)

def mistral_mlp_select_torch(x, Wgate, Wup, Wdown, idx, activation="relu"):
    """
    x: (B, hidden_size)
    Wgate: (hidden_size, intermediate_size)
    Wup: (hidden_size, intermediate_size)
    Wdown: (intermediate_size, hidden_size)
    idx: (K,)
    """
    act_fn = ACT2FN[activation]
    return torch.matmul(act_fn(torch.matmul(x, torch.index_select(Wgate, -1, idx))) * 
                        torch.matmul(x, torch.index_select(Wup, -1, idx)), 
                        torch.index_select(Wdown, -2, idx))

def mistral_mlp_torch(x, Wgate, Wup, Wdown, idx, activation="relu"):
    """
    x: (B, hidden_size)
    Wgate: (hidden_size, intermediate_size)
    Wup: (hidden_size, intermediate_size)
    Wdown: (intermediate_size, hidden_size)
    """
    act_fn = ACT2FN[activation]
    return torch.matmul(act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup), Wdown)

def degrad_tensor(x):
    x.grad = None
    x.requires_grad = False
    return x

def benchmark(fn, input, Wgate, Wup, Wdown, idx, activation="swish", warmup=20, rep=80, quantiles=None, fast_flush=True, return_mode="mean"):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn(input, Wgate, Wup, Wdown, idx, activation)
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
        fn(input, Wgate, Wup, Wdown, idx, activation)
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
        fn(input, Wgate, Wup, Wdown, idx, activation)
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn(input, Wgate, Wup, Wdown, idx, activation)
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
    K = int(intermediate_size // 64)
    config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish')
    # config = Config(4, 14, 2, 7, torch.float16, 'swish', 'relu')
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    # B = 16
    B = 1
    input = torch.randn((B, config.hidden_size), dtype=torch.float16).cuda()
    mlp(input)
    input = degrad_tensor(input)
    gate_proj = degrad_tensor(mlp.gate_proj.weight)
    up_proj = degrad_tensor(mlp.up_proj.weight)
    down_proj = degrad_tensor(mlp.down_proj.weight)
    output = mistral_mlp_select_torch(input, gate_proj.T, up_proj.T, down_proj.T, mlp.topk_idcs, config.hidden_act)
    # print(output)
    test_output = mistral_mlp_sparse(input, gate_proj, up_proj, down_proj.T, mlp.topk_idcs, config.hidden_act)
    # print(test_output)
    # assert torch.allclose(output, test_output, rtol=0.01)
    print(torch.nonzero(torch.abs(output - test_output) > 0.0005))
    time_torch = benchmark(mistral_mlp_select_torch, input, gate_proj.T, up_proj.T, down_proj.T, mlp.topk_idcs, config.hidden_act)
    time_triton = benchmark(mistral_mlp_sparse, input, gate_proj, up_proj, down_proj.T, mlp.topk_idcs, config.hidden_act)
    print("Sparse Torch: {:.2f} ms".format(time_torch))
    print("Sparse Triton: {:.2f} ms".format(time_triton))
    time_dense_torch = benchmark(mistral_mlp_torch, input, gate_proj.T, up_proj.T, down_proj.T, mlp.topk_idcs, config.hidden_act)
    print("Dense Torch: {:.2f} ms".format(time_dense_torch))