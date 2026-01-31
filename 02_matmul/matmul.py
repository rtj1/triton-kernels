"""
Matrix Multiplication Kernel in Triton
======================================

This implements a high-performance GEMM (General Matrix Multiply): C = A @ B

Key Concepts:
1. **2D Tiling**: Divide output matrix into tiles, each processed by one program
2. **Block-level parallelism**: Each program computes a BLOCK_M x BLOCK_N tile of C
3. **K-dimension reduction**: Iterate over K in chunks of BLOCK_K
4. **Memory coalescing**: Access patterns optimized for GPU memory hierarchy
5. **Register blocking**: Accumulate partial results in registers

Performance Targets:
- Should match ~80-90% of cuBLAS performance for large matrices
- Key bottleneck: memory bandwidth for loading A and B tiles

Memory Access Pattern:
```
        K                    N
    ┌────────┐          ┌────────┐
  M │   A    │   x    K │   B    │
    └────────┘          └────────┘
         ↓
         ┌────────┐
       M │   C    │  N
         └────────┘

Each program computes one tile of C:
- Loads BLOCK_M x BLOCK_K tiles from A
- Loads BLOCK_K x BLOCK_N tiles from B
- Accumulates into BLOCK_M x BLOCK_N tile of C
```
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (elements to skip to get to next row/column)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)

    Each program instance computes a BLOCK_M x BLOCK_N tile of C.
    """
    # =========================================================================
    # Program ID and Tile Assignment
    # =========================================================================
    # Get program ID
    pid = tl.program_id(axis=0)

    # Calculate number of tiles in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Group programs for better L2 cache reuse
    # Programs in same group work on nearby tiles
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Calculate which tile this program computes
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # =========================================================================
    # Pointer Arithmetic
    # =========================================================================
    # Calculate starting row/column offsets for this tile
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B for this program
    # A tile: rows [pid_m*BLOCK_M : (pid_m+1)*BLOCK_M], cols [0 : BLOCK_K]
    # B tile: rows [0 : BLOCK_K], cols [pid_n*BLOCK_N : (pid_n+1)*BLOCK_N]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # =========================================================================
    # Main Loop - Accumulate over K dimension
    # =========================================================================
    # Initialize accumulator for this tile
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in chunks of BLOCK_K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles of A and B
        # Mask out-of-bounds accesses
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # Matrix multiply and accumulate: accumulator += A @ B
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert accumulator to output dtype
    c = accumulator.to(tl.float16)

    # =========================================================================
    # Store Output Tile
    # =========================================================================
    # Calculate output pointers
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Create mask for boundary tiles
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Store the result
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix multiplication C = A @ B using Triton.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)

    Returns:
        c: Output matrix C of shape (M, N)
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape[1] == b.shape[0], f"Inner dimensions must match: {a.shape[1]} vs {b.shape[0]}"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Calculate grid
    # grid = (num_tiles_m * num_tiles_n,)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================================
# Non-autotuned version for educational purposes
# ============================================================================

@triton.jit
def matmul_kernel_simple(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Simplified matmul kernel without grouping (for educational purposes).
    """
    # Get 2D program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Simple matmul without autotuning."""
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matmul_kernel_simple[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return c


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_matmul(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """
    Benchmark Triton matmul vs PyTorch/cuBLAS.

    Returns dict with timing and TFLOPS metrics.
    """
    # Create inputs in float16 for better performance
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = matmul_triton(a, b)
        _ = torch.matmul(a, b)

    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = matmul_triton(a, b)
        end.record()

        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    # Benchmark PyTorch (cuBLAS)
    torch_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = torch.matmul(a, b)
        end.record()

        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))

    # Calculate metrics
    triton_avg = sum(triton_times) / len(triton_times)
    torch_avg = sum(torch_times) / len(torch_times)

    # FLOPS = 2 * M * N * K (multiply-add)
    flops = 2 * M * N * K
    triton_tflops = flops / (triton_avg * 1e-3) / 1e12
    torch_tflops = flops / (torch_avg * 1e-3) / 1e12

    return {
        'M': M, 'N': N, 'K': K,
        'triton_ms': triton_avg,
        'torch_ms': torch_avg,
        'speedup': torch_avg / triton_avg,
        'triton_tflops': triton_tflops,
        'torch_tflops': torch_tflops,
        'efficiency': triton_tflops / torch_tflops * 100,
    }


def verify_correctness(M: int = 512, N: int = 512, K: int = 512) -> bool:
    """Verify Triton matmul produces correct results."""
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_result = matmul_triton(a, b)
    torch_result = torch.matmul(a, b)

    # Use larger tolerance for fp16
    is_correct = torch.allclose(triton_result, torch_result, rtol=1e-2, atol=1e-2)

    if is_correct:
        print(f"[PASS] Correctness verified for M={M}, N={N}, K={K}")
    else:
        max_diff = (triton_result - torch_result).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}")

    return is_correct


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Matrix Multiplication Kernel - Triton Implementation")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. This code requires a GPU.")
        exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Verify correctness
    print("\n" + "-" * 80)
    print("Correctness Tests")
    print("-" * 80)

    for size in [128, 256, 512, 1024]:
        verify_correctness(M=size, N=size, K=size)

    # Benchmark
    print("\n" + "-" * 80)
    print("Performance Benchmarks (Square Matrices)")
    print("-" * 80)

    print(f"\n{'Size':>8} | {'Triton (ms)':>12} | {'cuBLAS (ms)':>12} | {'Triton TFLOPS':>14} | {'cuBLAS TFLOPS':>14} | {'Efficiency':>10}")
    print("-" * 90)

    sizes = [512, 1024, 2048, 4096, 8192]

    for size in sizes:
        try:
            results = benchmark_matmul(M=size, N=size, K=size)
            print(f"{size:>8} | {results['triton_ms']:>12.3f} | {results['torch_ms']:>12.3f} | "
                  f"{results['triton_tflops']:>14.2f} | {results['torch_tflops']:>14.2f} | "
                  f"{results['efficiency']:>9.1f}%")
        except Exception as e:
            print(f"{size:>8} | Error: {e}")

    # Non-square matrices
    print("\n" + "-" * 80)
    print("Performance Benchmarks (Non-Square Matrices)")
    print("-" * 80)

    print(f"\n{'(M, N, K)':>20} | {'Triton (ms)':>12} | {'cuBLAS (ms)':>12} | {'Efficiency':>10}")
    print("-" * 70)

    shapes = [
        (8192, 8192, 1024),   # Wide A
        (1024, 8192, 8192),   # Wide B
        (4096, 4096, 4096),   # Square
        (16384, 16384, 128),  # Very wide with small K
    ]

    for M, N, K in shapes:
        try:
            results = benchmark_matmul(M=M, N=N, K=K)
            print(f"({M}, {N}, {K}):".ljust(20) + f" | {results['triton_ms']:>12.3f} | {results['torch_ms']:>12.3f} | "
                  f"{results['efficiency']:>9.1f}%")
        except Exception as e:
            print(f"({M}, {N}, {K}):".ljust(20) + f" | Error: {e}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- TFLOPS = 2 * M * N * K / time (multiply-add counted as 2 FLOPS)")
    print("- Efficiency = Triton TFLOPS / cuBLAS TFLOPS * 100")
    print("- cuBLAS is highly optimized; 80-90% efficiency is excellent")
    print("- Autotuning selects best block sizes for each matrix shape")
    print("=" * 80)
