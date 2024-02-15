#pragma once

#include <torch/torch.h>
int64_t fuse_gemv_cmp_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, double threshold);
void fuse_gemv_flag_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold);