#include "../reduceutils.h"
#include "index_scatter_cuda.h"
#include "index_scatter_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace at::native;

// policy listed in template
template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void segscan_sr_sorted(const at::Tensor &index, const at::Tensor &src,
                       const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  thrust::device_vector<int64_t> d_indices(indices, indices + nnz);
  thrust::device_vector<int> dC(thrust::count_if(d_indices.begin(), d_indices.end(),[] __device__ (const int& x){
    return x < 5;
  }));
  printf("dC.size() = %d\n", dC.size());

  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(N, NThreadX * NPerThread),
               CEIL(nnz, NnzThreadY * NnzPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_sr_sorted_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                           NnzThreadY>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void segscan_pr_sorted(const at::Tensor &index, const at::Tensor &src,
                       const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_pr_sorted_kernel<scalar_t, NPerThread, NThreadY, NnzPerThread, RNum,
                           RSync>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto keys = dst.numel() / N;
  int avg_key_len = nnz / keys;
  if (N >= 1 && N <= 4) {
    segscan_pr_sorted<scalar_t, 1, 1, 2, 4, 32>(index, src, dst);
  } else if (N > 4 && N < 32) {
    segscan_pr_sorted<scalar_t, 2, 2, 2, 4, 32>(index, src, dst);
  } else if (N >= 32 && N < 64) {
    if (avg_key_len < 16) {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 1>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 2>(index, src, dst);
    } else {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 4>(index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 32, 32, 1>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 32, 32, 2>(index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 32, 32, 4>(index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 64, 32, 1>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 64, 32, 2>(index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 64, 32, 4>(index, src, dst);
    }
  }
}

void index_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst,
                                   const ReductionType &reduction) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_sorted", [&] {
    DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      index_scatter_sorted_wrapper<scalar_t, reduce>(index, src, dst);
    });
  });
}

at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce,
                              const bool sorted) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)

  if (sorted) {
    index_scatter_sorted_dispatch(index, src, dst, reduce_type);
  } else {
    TORCH_CHECK(false, "unsorted index is not supported yet");
  }

  return dst;
}


template <typename scalar_t>
int64_t fast_nonzero_cuda_kernel(const at::Tensor &value, const at::Tensor &orange, at::Tensor &result, const at::Tensor &threshold) {
    // printf("fast_nonzero_cuda_kernel\n");
    const auto nnz = value.numel();
    auto threshold_data = threshold.data_ptr<scalar_t>();
    scalar_t ths = static_cast<scalar_t>(threshold_data[0]);
    // printf("ths = %f\n", ths);
    auto value_data = value.data_ptr<scalar_t>();
    auto result_data = result.data_ptr<int64_t>();
    auto indices_data = orange.data_ptr<int64_t>();
    
    int64_t N = thrust::count_if(thrust::device, value_data, value_data + nnz, [=] __device__ (const scalar_t& x){
        return x > ths;
    });

    thrust::copy_if(thrust::device, indices_data, indices_data + nnz, value_data, result_data, [=] __device__ (const scalar_t& x){
        return x > ths;
    });
    return N;
}

int64_t fast_nonzero_dispatch(const at::Tensor &value, const at::Tensor &orange, at::Tensor &result, const at::Tensor &threshold) {
    int64_t N;
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "fast_nonzero", [&] {
        N = fast_nonzero_cuda_kernel<scalar_t>(value, orange, result, threshold);
    });
    return N;
}
