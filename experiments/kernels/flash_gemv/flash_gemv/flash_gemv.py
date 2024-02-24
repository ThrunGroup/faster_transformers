import torch

def fuse_gemv_cmp(input, weight, output, idx, threshold):
    return torch.ops.flash_gemv.fuse_gemv_cmp(input, weight, output, idx, threshold)

def fuse_gemv_flag(input, weight, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag(input, weight, output, flag, threshold)

def fuse_gemv_flag_gemv(input, wgate, wup, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_gemv(input, wgate, wup, output, flag, threshold)

def fuse_gemv_flag_gemv_gemv(input, wgate, wup, wdown, output, flag, threshold):
    return torch.ops.flash_gemv.fuse_gemv_flag_gemv_gemv(input, wgate, wup, wdown, output, flag, threshold)

def fuse_gemv_gemv_gemv(input, wgate, wup, wdown, output, threshold):
    return torch.ops.flash_gemv.fuse_gemv_gemv_gemv(input, wgate, wup, wdown, output, threshold)