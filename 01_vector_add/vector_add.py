"""
Vector Addition Kernel in Triton
================================

This is the "Hello World" of GPU programming. It demonstrates:
1. How to write a Triton kernel
2. How to launch kernels with a grid
3. How to handle memory pointers and offsets
4. How to use masks for boundary conditions

The kernel computes: C = A + B (element-wise)

Key Concepts:
- program_id: Each kernel instance gets a unique ID
- BLOCK_SIZE: Number of elements processed per kernel instance
- tl.load/tl.store: Memory operations with optional masking
- Grid: How many kernel instances to launch
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def vector_add_kernel(
    # Pointers to tensors
    a_ptr,
    b_ptr,
    c_ptr,
    # Size of the vector
    n_elements,
    # Block size (number of elements each program instance processes)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise vector addition.

    Each program instance processes BLOCK_SIZE elements.
    Total elements = grid_size * BLOCK_SIZE

    Memory Layout:
    - Each program gets a contiguous chunk of BLOCK_SIZE elements
    - program_id(0) processes elements [0, BLOCK_SIZE)
    - program_id(1) processes elements [BLOCK_SIZE, 2*BLOCK_SIZE)
    - etc.
    """
    # Get the program ID (which chunk we're processing)
    pid = tl.program_id(axis=0)

    # Calculate the starting offset for this program
    # Each program handles BLOCK_SIZE consecutive elements
    block_start = pid * BLOCK_SIZE

    # Create offset array for this block
    # offsets = [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary conditions
    # Some elements might be out of bounds if n_elements is not divisible by BLOCK_SIZE
    mask = offsets < n_elements

    # Load data from global memory
    # masked load returns 0.0 for out-of-bounds elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Perform the addition
    c = a + b

    # Store the result back to global memory
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Wrapper function to call the Triton vector addition kernel.

    Args:
        a: First input tensor (1D, on CUDA)
        b: Second input tensor (1D, on CUDA, same shape as a)
        block_size: Number of elements per thread block

    Returns:
        c: Result tensor (a + b)
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
    assert a.shape == b.shape, "Input shapes must match"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"

    # Allocate output tensor
    c = torch.empty_like(a)

    n_elements = a.numel()

    # Calculate grid size (number of kernel instances to launch)
    # We need ceil(n_elements / block_size) blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    vector_add_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return c


def vector_add_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch native vector addition for comparison."""
    return a + b


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_vector_add(
    size: int = 1_000_000,
    block_size: int = 1024,
    warmup: int = 10,
    rep: int = 100,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Benchmark Triton vs PyTorch vector addition.

    Args:
        size: Number of elements in vectors
        block_size: Triton block size
        warmup: Number of warmup iterations
        rep: Number of timed iterations
        dtype: Data type for tensors

    Returns:
        Dictionary with benchmark results
    """
    # Create random input tensors
    a = torch.randn(size, device='cuda', dtype=dtype)
    b = torch.randn(size, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = vector_add_triton(a, b, block_size)
        _ = vector_add_torch(a, b)

    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = vector_add_triton(a, b, block_size)
        end.record()

        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    # Benchmark PyTorch
    torch_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = vector_add_torch(a, b)
        end.record()

        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))

    # Calculate statistics
    triton_avg = sum(triton_times) / len(triton_times)
    torch_avg = sum(torch_times) / len(torch_times)

    # Calculate bandwidth (GB/s)
    # We read 2 vectors and write 1 vector
    bytes_transferred = 3 * size * a.element_size()
    triton_bandwidth = bytes_transferred / (triton_avg * 1e-3) / 1e9  # GB/s
    torch_bandwidth = bytes_transferred / (torch_avg * 1e-3) / 1e9  # GB/s

    return {
        'size': size,
        'dtype': str(dtype),
        'triton_ms': triton_avg,
        'torch_ms': torch_avg,
        'speedup': torch_avg / triton_avg,
        'triton_bandwidth_gbps': triton_bandwidth,
        'torch_bandwidth_gbps': torch_bandwidth,
    }


def verify_correctness(size: int = 10000, dtype: torch.dtype = torch.float32) -> bool:
    """Verify that Triton kernel produces correct results."""
    a = torch.randn(size, device='cuda', dtype=dtype)
    b = torch.randn(size, device='cuda', dtype=dtype)

    triton_result = vector_add_triton(a, b)
    torch_result = vector_add_torch(a, b)

    is_correct = torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)

    if is_correct:
        print(f"[PASS] Correctness verified for size={size}, dtype={dtype}")
    else:
        max_diff = (triton_result - torch_result).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}")

    return is_correct


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Vector Addition Kernel - Triton Implementation")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. This code requires a GPU.")
        exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Verify correctness
    print("\n" + "-" * 70)
    print("Correctness Tests")
    print("-" * 70)

    for dtype in [torch.float32, torch.float16]:
        verify_correctness(size=100000, dtype=dtype)

    # Run benchmarks
    print("\n" + "-" * 70)
    print("Performance Benchmarks")
    print("-" * 70)

    sizes = [1024, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    print(f"\n{'Size':>12} | {'Triton (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8} | {'Triton BW':>10} | {'PyTorch BW':>10}")
    print("-" * 80)

    for size in sizes:
        try:
            results = benchmark_vector_add(size=size)
            print(f"{results['size']:>12,} | {results['triton_ms']:>12.4f} | {results['torch_ms']:>12.4f} | "
                  f"{results['speedup']:>8.2f}x | {results['triton_bandwidth_gbps']:>9.1f} GB/s | "
                  f"{results['torch_bandwidth_gbps']:>9.1f} GB/s")
        except Exception as e:
            print(f"{size:>12,} | Error: {e}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- Bandwidth = 3 * size * element_size / time (read A, B; write C)")
    print("- For small sizes, kernel launch overhead dominates")
    print("- For large sizes, memory bandwidth is the bottleneck")
    print("=" * 70)
