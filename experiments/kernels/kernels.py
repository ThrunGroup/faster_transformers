import torch
import triton
import triton.language as tl

from k_activations import  silu, cramp
from dejavu import apply_activation

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "swish":
        accumulator = silu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)




# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul_triton(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,
    )
    return c

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_bias(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn).to(tl.float32)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if HAS_BIAS:
        accumulator += bias[None, :]
    if ACTIVATION == "swish":
        accumulator = silu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)




# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul_triton_bias(a, b, bias, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bias[grid](
        a, b, c, bias,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        HAS_BIAS=1,
        ACTIVATION=activation,
    )
    return c

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_bias_where(
        # Pointers to matrices
        a_ptr, b_ptr, bias_ptr, indices_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        threshold: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn).to(tl.float32)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if HAS_BIAS:
        accumulator += bias[None, :]
    if ACTIVATION == "swish":
        accumulator = silu(accumulator)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    indices_ptrs = indices_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    i_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    accumulator = cramp(accumulator, threshold)
    tl.store(indices_ptrs, accumulator, mask=i_mask)

# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul_triton_bias_where(a, b, bias, threshold: float, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    indices = torch.empty((M, N), device=a.device, dtype=torch.int32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bias_where[grid](
        a, b, bias, indices,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        indices.stride(0), indices.stride(1),  #
        threshold=threshold,
        HAS_BIAS=1,
        ACTIVATION=activation,
    )
    return indices

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)

@triton.jit
def gather_gemv2_flag_kernel(
    Y,  # Pointers to matrices
    A,
    B,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Kernel for computing Y = ACTIVATION(A[IDX, :] @ X) * (B[IDX, :] @ X), where A and B are 
    dense matrices with M rows and N columns.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - A and B have shape (M, N)
    - IDX has shape (M), where M is the flag for non-zero rows in A
    - Output has shape (BATCHSIZE, M)
    """
    # EVEN_N is asserted to be true
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A and B
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    # count = 0
    # for i in tl.static_range(0,BLOCK_M,1):
    #     if idx[i] > 0:
    #         idx[count] = start_m * BLOCK_M + i
    #         count += 1

    # A = A + (idx[:count, None] * stride_am + rn[None, :])
    # B = B + (idx[:count, None] * stride_am + rn[None, :])
    A = A + (rm[:, None] * stride_am + rn[None, :])
    B = B + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn

    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        # acc0 = tl.zeros((count,), dtype=tl.float32)
        # acc1 = tl.zeros((count,), dtype=tl.float32)
        i_mask = idx[:, None] > 0
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A, mask=i_mask, other=0.0)
            b = tl.load(B, mask=i_mask, other=0.0)
            # a = tl.load(A)
            # b = tl.load(B)
            x0 = tl.load(X)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(b.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            B += BLOCK_N
            X += BLOCK_N
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        
    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    acc = acc0 * acc1
    tl.store(Y, acc, mask=rm < M)

def gather_gemv2_flag(
    x: torch.Tensor,
    wgate: torch.Tensor,
    wup: torch.Tensor,
    idx: torch.Tensor,
    activation: str = "relu",
) -> torch.Tensor:
    """
    Compute y = activation(x @ wgate[idx, :].T) * (x @ wup[idx, :].T).
    :param x: input tensor, (batch, N)
    :param wgate: gate weight matrix, (Z, N)
    :param wup: up weigth matrix, (Z, N)
    :param idx: indices, (Z,)
    :return: result tensor, (batch, N)
    """
    Z, N = wgate.shape
    batch, _ = x.shape
    assert x.shape == (batch, N)
    assert batch == 1
    x = x.contiguous()
    if wgate.stride(1) > 1:
        wgate = wgate.contiguous()
    assert (
        x.dtype == wgate.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {wgate.dtype}"
    if wup.stride(1) > 1:
        wup = wup.contiguous()
    assert (
        x.dtype == wup.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {wup.dtype}"

    output = torch.empty(batch, Z, device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(Z, META["BLOCK_M"]),)  # noqa

    gather_gemv2_flag_kernel[grid](
        output,  # data ptrs
        wgate,
        wup,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        wgate.stride(0),  # strides
        batch,  # Can't use kwargs because auto-tuner requires args
        ACTIVATION=activation,
    )

    return output

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_flag_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    a = tl.load(A, mask=idx[:, None] > 0, other=0.0)
    if BATCHSIZE == 1:
        x0 = tl.load(X, mask=idx>0, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)

def gather_transposed_gemv_flag(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    batch, _ = x.shape
    assert x.shape == (batch,Z)
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"

    output = torch.empty(
        batch,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(Z, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    kernel = gather_transposed_gemv_flag_atomicadd_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        batch,  # can't use kwargs because auto-tuner requires args
    )

    return output.to(dtype=x.dtype)

def mistral_mlp_sparse_flag(x, Wgate, Wup, Wdownt, flags, activation):
    x = gather_gemv2_flag(x, Wgate, Wup, flags, activation)
    return gather_transposed_gemv_flag(x, Wdownt, flags)