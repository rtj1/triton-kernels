"""
Matrix Multiplication Kernel - Optimized for T4 GPU
====================================================

T4 GPU Characteristics:
- 320 Tensor Cores (Turing architecture)
- 16 GB GDDR6 memory
- 300 GB/s memory bandwidth
- 65 TFLOPS FP16 Tensor Core peak
- Smaller shared memory than A100 (64KB vs 164KB)

Key optimizations for T4:
1. Smaller block sizes (64x64 or 128x64 instead of 128x256)
2. Fewer pipeline stages (2-3 instead of 4-5)
3. Fewer warps (4 instead of 8)
4. Careful tuning for Turing architecture
"""

import torch
import triton
import triton.language as tl
import time


# Autotuning configs optimized for T4 GPU
@triton.autotune(
    configs=[
        # T4-optimized configs with smaller blocks
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Additional small configs for smaller matrices
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_t4_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B with T4-optimized tiling.

    Uses L2 cache grouping for better data reuse.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator in fp32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load with bounds checking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # Matrix multiply and accumulate
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert to output dtype
    c = accumulator.to(tl.float16)

    # Store with bounds checking
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_t4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication optimized for T4 GPU.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)

    Returns:
        c: Output matrix C of shape (M, N)
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_t4_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================================
# Non-autotuned version with fixed good config for T4
# ============================================================================

@triton.jit
def matmul_t4_fixed_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fixed config matmul kernel (no autotuning overhead)."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_t4_fixed(a: torch.Tensor, b: torch.Tensor, block_m=64, block_n=64, block_k=32) -> torch.Tensor:
    """Matrix multiplication with fixed block sizes (no autotuning)."""
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matmul_t4_fixed_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    return c


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_matmul_t4(sizes=[512, 1024, 2048, 4096]):
    """Benchmark T4-optimized matmul."""
    print("=" * 80)
    print("Matrix Multiplication - T4 Optimized")
    print("=" * 80)

    results = []

    for size in sizes:
        a = torch.randn((size, size), device='cuda', dtype=torch.float16)
        b = torch.randn((size, size), device='cuda', dtype=torch.float16)

        # Correctness check
        triton_out = matmul_t4(a, b)
        torch_out = torch.matmul(a, b)
        is_correct = torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)

        # Warmup
        for _ in range(20):
            _ = matmul_t4(a, b)
            _ = torch.matmul(a, b)

        torch.cuda.synchronize()

        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(50):
            _ = matmul_t4(a, b)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 50 * 1000

        # Benchmark cuBLAS
        start = time.perf_counter()
        for _ in range(50):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        cublas_time = (time.perf_counter() - start) / 50 * 1000

        # Calculate TFLOPS
        flops = 2 * size * size * size
        triton_tflops = flops / (triton_time * 1e-3) / 1e12
        cublas_tflops = flops / (cublas_time * 1e-3) / 1e12

        results.append({
            'size': size,
            'correct': is_correct,
            'triton_ms': triton_time,
            'cublas_ms': cublas_time,
            'triton_tflops': triton_tflops,
            'cublas_tflops': cublas_tflops,
            'efficiency': triton_tflops / cublas_tflops * 100,
        })

        status = "✓" if is_correct else "✗"
        print(f"Size {size}: {status} | Triton: {triton_time:.3f}ms ({triton_tflops:.1f} TFLOPS) | "
              f"cuBLAS: {cublas_time:.3f}ms ({cublas_tflops:.1f} TFLOPS) | Efficiency: {results[-1]['efficiency']:.0f}%")

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    benchmark_matmul_t4()
