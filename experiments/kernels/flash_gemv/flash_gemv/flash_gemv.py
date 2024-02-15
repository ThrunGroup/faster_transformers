import torch

def fuse_gemv_cmp(input, weight, output, idx, threshold):
    return torch.ops.flash_gemv.fuse_gemv_cmp(input, weight, output, idx, threshold)

def fuse_gemv_flag(input, weight, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag(input, weight, output, flag, threshold)