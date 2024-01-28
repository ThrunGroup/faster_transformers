import torch
import torch_index_scatter

def test_fast_nonzero():
    len = 10000
    x = torch.randn(len).cuda()
    y = torch.zeros(len, dtype=torch.int64).cuda()
    orange = torch.arange(len, dtype=torch.int64).cuda()
    threshold = torch.Tensor([0.1]).to(torch.float32)
    print(threshold)
    print("Test begin")
    N = torch_index_scatter.fast_nonzero(x, orange, y, threshold)
    golden = torch.nonzero(x > 0.1, as_tuple=False).squeeze(1)
    # print(golden.shape)
    # print(y)
    assert torch.allclose(y[:N], golden)

if __name__ == '__main__':
    test_fast_nonzero()