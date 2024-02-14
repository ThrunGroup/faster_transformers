import torch
import torch.nn as nn
import flash_gemv

def test():
    input = torch.randn(1, 1, 4096).half().cuda()
    weight = torch.randn(4096, 14336).half().cuda()
    weight_t = weight.t().contiguous()
    threshold = 0.1
    idx = torch.zeros(14336, dtype=torch.int64).cuda()
    value = torch.zeros(1, 1, 14336).half().cuda()
    N = flash_gemv.fuse_gemv_cmp(input, weight_t, value, idx, threshold)
    torch.cuda.synchronize()
    print("finish")
    act_fn = nn.SiLU()
    output = torch.abs(act_fn(torch.matmul(input, weight)))
    (_,_,golden_idx) = torch.nonzero(output > threshold, as_tuple=True)
    print(golden_idx.shape)
    # print(value.shape)
    # print(value[0])
    print(N)
    idx = idx[:N]
    # Sort idx and value together

    idx_sorted, indices = torch.sort(idx[:N])
    value_sorted = value[0][0][indices]
    print(value_sorted)
    print(output[0][0][golden_idx])
    print(idx_sorted)
    print(golden_idx)

    # assert torch.allclose(value[0][0][idx_sorted], golden)

if __name__ == '__main__':
    test()