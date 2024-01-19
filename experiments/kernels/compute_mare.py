import torch
import torch.nn as nn

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
        print(probs)
        _, topk_idcs = torch.topk(probs, k=self.K, dim=-1)
        mask = torch.zeros(
            x.shape[:-1] + (self.intermediate_size,),
            dtype=probs.dtype,
            device=probs.device,
        ).scatter(-1, topk_idcs, 1)
        print(mask)
        masked_up = (self.act_fn(self.gate_proj(x)) * self.up_proj(x) * mask)
        print(masked_up)
        return self.down_proj(masked_up)


if __name__ == "__main__":
    # config = Config(4096, 14336, 64, 16, torch.float16, 'swish')
    config = Config(4, 14, 2, 7, torch.float16, 'swish', 'relu')
    # mlp = MistralMLP(config).to(torch.float16).cuda()
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    # B = 16
    B = 1
    N = 2
    input = torch.randn((B, N, config.hidden_size), dtype=torch.float16).cuda()
    output = mlp(input)
