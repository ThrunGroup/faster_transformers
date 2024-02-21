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

__global__ void fuse_gemv_flag_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
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
                flag[m] = true;
                output[m] = __float2half_rn(sum_f);
            }
        }
    } 
}

__forceinline__ __device__ uint64_t load_4half_to_uint64(const half* src) {
    half combined[4] = {src[0], src[32], src[64], src[96]};
    return *(uint64_t*)combined;
}

__forceinline__ __device__ uint64_t load_2half2_to_uint64(const half2* src) {
    half2 combined[2] = {src[0], src[32]};
    return *(uint64_t*)combined; 
}

__forceinline__ __device__ void store_uint64_to_2half2(uint64_t src, half2* dst) {
    half2* combined = (half2*)&src;
    dst[0] = combined[0];
    dst[32] = combined[1];
}

__forceinline__ __device__ uint64_t add_2half2_uint64(const uint64_t a, const uint64_t b) {
    half2 a_half[2];
    half2 b_half[2];
    half2 c_half[2];
    *(uint64_t*)a_half = a;
    *(uint64_t*)b_half = b;
    c_half[0] = __hadd2(a_half[0], b_half[0]);
    c_half[1] = __hadd2(a_half[1], b_half[1]);
    return *(uint64_t*)c_half;
}

__forceinline__ __device__ uint64_t mul_2half2_uint64(const uint64_t a, const uint64_t b) {
    half2 a_half[2];
    half2 b_half[2];
    half2 c_half[2];
    *(uint64_t*)a_half = a;
    *(uint64_t*)b_half = b;
    c_half[0] = __hmul2(a_half[0], b_half[0]);
    c_half[1] = __hmul2(a_half[1], b_half[1]);
    return *(uint64_t*)c_half;
}

__forceinline__ __device__ half2 reduce_2half2_uint64(const uint64_t a) {
    half2 a_half[2];
    *(uint64_t*)a_half = a;
    half2 sum = __hadd2(a_half[0], a_half[1]);
    return sum;
}

__global__ void fuse_gemv_flag_opt_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    uint64_t w = load_4half_to_uint64(weight + m * N + warp_id * 128 + lane_id);
    uint64_t x = load_4half_to_uint64(input + warp_id * 128 + lane_id);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        store_uint64_to_2half2(sum, temp_result + warp_id);
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = load_2half2_to_uint64(temp_result + lane_id);
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__global__ void fuse_gemv_flag_mem_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 16 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 16 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_inter_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ uint64_t temp_result[32];
    uint64_t w = load_4half_to_uint64(weight + m * N + warp_id * 128 + lane_id);
    uint64_t x = load_4half_to_uint64(input + warp_id * 128 + lane_id);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = temp_result[lane_id];
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__forceinline__ __device__ uint64_t load_cont4half_to_uint64(const half* src) {
    half combined[4] = {src[0], src[1], src[2], src[3]};
    return *(uint64_t*)combined;
}

__global__ void fuse_gemv_flag_cont_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    uint64_t w = load_cont4half_to_uint64(weight + m * N + warp_id * 128 + lane_id * 4);
    uint64_t x = load_cont4half_to_uint64(input + warp_id * 128 + lane_id * 4);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        store_uint64_to_2half2(sum, temp_result + warp_id);
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = load_2half2_to_uint64(temp_result + lane_id);
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__global__ void fuse_gemv_flag_mem1024_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[32];
    half2* local_weight_ptr = (half2*)(weight + m * N + 128 * warp_id + 4 * lane_id);
    half2* local_input_ptr = (half2*)(input + 128 * warp_id + 4 * lane_id);

    
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = temp_result[lane_id];
        #pragma unroll
        for (int i = 16; i > 0; i /= 2) {
            sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, i));
        }
        if (lane_id == 0) {
            float sum_f = __half2float(__hadd(sum.x, sum.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
        }
    }
}

__global__ void fuse_gemv_flag_mem128_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[4];
    half2* local_weight_ptr = (half2*)(weight + m * N + 1024 * warp_id + 32 * lane_id);
    half2* local_input_ptr = (half2*)(input + 1024 * warp_id + 32 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_fp32_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        float2 w = __half22float2(local_weight_ptr[i]);
        float2 x = __half22float2(local_input_ptr[i]);
        sum = sum + w.y * x.y + w.x * x.x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum) > threshold;
        output[m] = __float2half_rn(sum);
    }
}

__global__ void fuse_gemv_flag_mem256_2_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int mz = threadIdx.z;
    int m = blockIdx.x * 2 + mz;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[16];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id + mz * 8] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i + mz * 8]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_4_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int mz = threadIdx.z;
    int m = blockIdx.x * 4 + mz;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[32];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id + mz * 8] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i + mz * 8]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem256_kernel1(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ half2 input_shared[2048];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
        input_shared[warp_id * 256 + lane_id + i] = x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = input_shared + warp_id * 256 + lane_id;
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem128_kernel(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[4];
    __shared__ bool flag_shared[1];
    __shared__ half2 input_shared[2048];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 1024 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 1024 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 512; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
        input_shared[warp_id * 512 + lane_id + i] = x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 1024 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = input_shared + warp_id * 512 + lane_id;
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 512; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = true;
        value_shared[0] = sum_f;
        // output[blockIdx.x] = __float2half_rn(sum_f);
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    // half2* global_output_ptr = (half2*)(output + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    half2 sum_down = __float2half2_rn(value_shared[0]);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = __hmul2(local_weight_ptr_down[i], sum_down);
        // atomicAdd(global_output_ptr + i, __hmul2(w, sum_down));
        // atomicAdd(global_output_ptr + i * 2, __float2half2_rn(1.0f).x);
        // atomicAdd(global_output_ptr + i * 2 + 1, __float2half2_rn(1.0f).y);
        // atomicAdd(global_output_ptr + i * 2, w.x);
        // atomicAdd(global_output_ptr + i * 2 + 1, w.y);
        // atomicAdd(global_output_ptr + i * 2, __float2half_rn(0.000001f * (float)blockIdx.x));
        // atomicAdd(global_output_ptr + i * 2, 0.000001f * (float)blockIdx.x);
        atomicAdd(global_output_ptr + i * 2, __half2float(w.x));
        atomicAdd(global_output_ptr + i * 2 + 1, __half2float(w.y));
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        value_shared[0] = sum_f;
        // output[blockIdx.x] = __float2half_rn(sum_f);
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    // half2* global_output_ptr = (half2*)(output + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    half2 sum_down = __float2half2_rn(value_shared[0]);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = __hmul2(local_weight_ptr_down[i], sum_down);
        atomicAdd(global_output_ptr + i * 2, __half2float(w.x));
        atomicAdd(global_output_ptr + i * 2 + 1, __half2float(w.y));
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem256_fp32_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr[i]);
        // float2 x = __half22float2(local_input_ptr[i]);
        // sum = fmaf(w.x, x.x, sum);
        // sum = fmaf(w.y, x.y, sum);
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag_shared[0] = fabsf(sum) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    float sum_up = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr_up[i]);
        // float2 x = __half22float2(local_input_ptr_up[i]);
        // sum_up = fmaf(w.x, x.x, sum_up);
        // sum_up = fmaf(w.y, x.y, sum_up);
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum_up += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up += group.shfl_down(sum_up, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up += temp_result[i];
        }
        // sum *= sum_up;
        // value_shared[0] = __hmul(__float2half(sum), __float2half(sum_up));
        value_shared[0] = sum * sum_up;
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    float sum_down = value_shared[0];
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // half2 w = local_weight_ptr_down[i];
        // atomicAdd(global_output_ptr + i * 2, __half2float(__hmul(w.x, sum_down)));
        // atomicAdd(global_output_ptr + i * 2 + 1, __half2float(__hmul(w.y, sum_down)));
        float2 w = __half22float2(local_weight_ptr_down[i]);
        atomicAdd(global_output_ptr + i * 2, w.x * sum_down);
        atomicAdd(global_output_ptr + i * 2 + 1, w.y * sum_down);
    }
    
}

int64_t fuse_gemv_cmp_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, const double threshold) {
    length = 0;
    // printf("threshold: %f\n", threshold);
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    int64_t* idx_data = reinterpret_cast<int64_t*>(idx.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,32,1);

    fuse_gemv_cmp_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, idx_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    return length;
}

void fuse_gemv_flag_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    // dim3 blockDim(32,4,1);
    // fuse_gemv_flag_mem128_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    dim3 blockDim(32,8,1);
    // fuse_gemv_flag_mem_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_mem256_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    fuse_gemv_flag_mem256_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // dim3 blockDim(32,32,1);
    // fuse_gemv_flag_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_opt_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_inter_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_cont_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_mem1024_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    
    // dim3 gridDim(M/2,1,1);
    // dim3 blockDim(32,8,2);
    // fuse_gemv_flag_mem256_2_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // dim3 gridDim(M/4,1,1);
    // dim3 blockDim(32,8,4);
    // fuse_gemv_flag_mem256_4_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}

void fuse_gemv_flag_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    // dim3 blockDim(32,8,1);
    // fuse_gemv_flag_gemv_mem256_kernel1<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, output_data, flag_data, static_cast<float>(threshold));
    dim3 blockDim(32,4,1);
    fuse_gemv_flag_gemv_mem128_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, output_data, flag_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}

void fuse_gemv_flag_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,8,1);
    fuse_gemv_flag_gemv_gemv_mem256_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, flag_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void fuse_gemv_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,8,1);
    fuse_gemv_gemv_gemv_mem256_fp32_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, static_cast<float>(threshold));
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}