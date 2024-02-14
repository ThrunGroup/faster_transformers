#include "fuse_gemv_cmp_cuda.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>



#define N 4096
#define M 14336

using namespace at::native;
using namespace cooperative_groups;

constexpr float log2e = 1.44269504088896340736f;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }

__device__ __managed__ int length = 0;

__global__ void fuse_gemv_cmp_kernel(half* input, half* weight, half* output, int64_t* idx, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    half2 w_0 = make_half2(weight[m * N + warp_id * 128 + lane_id], weight[m * N + warp_id * 128 + lane_id + 32]);
    half2 w_1 = make_half2(weight[m * N + warp_id * 128 + lane_id + 64], weight[m * N + warp_id * 128 + lane_id + 96]);
    half2 x_0 = make_half2(input[warp_id * 128 + lane_id], input[warp_id * 128 + lane_id + 32]);
    half2 x_1 = make_half2(input[warp_id * 128 + lane_id + 64], input[warp_id * 128 + lane_id + 96]);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum_0 = __hmul2(w_0, x_0);
    half2 sum_1 = __hmul2(w_1, x_1);
    for (int i = 16; i > 0; i /= 2) {
        sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
        sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_0;
        temp_result[warp_id + 32] = sum_1;
    }
    __syncthreads();
    if (warp_id == 0) {
        half2 sum_0 = temp_result[lane_id];
        half2 sum_1 = temp_result[lane_id + 32];
        for (int i = 16; i > 0; i /= 2) {
            sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
            sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
        }
        if (lane_id == 0) {
            half2 sum = __hadd2(sum_0, sum_1);
            float sum_f = __half2float(__hadd(sum.x, sum.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            if (fabsf(sum_f) > threshold) {
                int offset = atomicAdd(&length, 1);
                idx[offset] = (int64_t)m;
                output[offset] = __float2half_rn(sum_f);
            }
        }
    } 

}

int64_t fuse_gemv_cmp_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    int64_t* idx_data = reinterpret_cast<int64_t*>(idx.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,32,1);

    fuse_gemv_cmp_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, idx_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    return length;
}