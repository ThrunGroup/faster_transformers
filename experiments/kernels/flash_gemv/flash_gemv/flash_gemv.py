import torch

def fuse_gemv_cmp(input, weight, output, idx, threshold):
    return torch.ops.flash_gemv.fuse_gemv_cmp(input, weight, output, idx, threshold)