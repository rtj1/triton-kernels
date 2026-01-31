"""
Fused Softmax Kernel in Triton
==============================

Implements numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Key Concepts:
1. **Numerical Stability**: Subtract max before exp to prevent overflow
2. **Online Algorithm**: Compute max and sum in single pass where possible
3. **Fusion**: Combine multiple operations into single kernel
4. **Row-wise Operation**: Each program handles one row (common in attention)

Why Fused Softmax Matters:
- Naive softmax: 3 kernel launches (max, subtract+exp, sum+divide)
- Fused softmax: 1 kernel launch
- Reduces memory bandwidth by 3x (each element read once, not three times)

Memory Pattern:
```
Input:  [batch, seq_len]  - Each row is independent
Output: [batch, seq_len]  - softmax(row)

Program i processes row i entirely in registers/shared memory
```
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel - each program processes one row.

    Algorithm (numerically stable):
    1. Find max value in row (for numerical stability)
    2. Compute exp(x - max) for each element
    3. Sum all exp values
    4. Divide each exp value by sum

    This is done in a single pass through memory when possible,
    or with minimal passes for very long rows.
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Calculate starting pointer for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Create pointers to input elements
    input_ptrs = row_start_ptr + col_offsets

    # Create mask for valid columns
    mask = col_offsets < n_cols

    # =========================================================================
    # Step 1: Load row and find max (for numerical stability)
    # =========================================================================
    # Load with -inf for masked elements (won't affect max)
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Find max value in row
    row_max = tl.max(row, axis=0)

    # =========================================================================
    # Step 2: Compute exp(x - max) - numerically stable
    # =========================================================================
    numerator = tl.exp(row - row_max)

    # =========================================================================
    # Step 3: Compute sum of exp values
    # =========================================================================
    denominator = tl.sum(numerator, axis=0)

    # =========================================================================
    # Step 4: Compute softmax = exp(x - max) / sum
    # =========================================================================
    softmax_output = numerator / denominator

    # =========================================================================
    # Store result
    # =========================================================================
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def softmax_kernel_online(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax kernel for rows longer than BLOCK_SIZE.

    Uses the online softmax algorithm:
    - Process row in chunks
    - Maintain running max and sum with correction factor
    - Final pass to compute outputs

    This is the algorithm used in FlashAttention!
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Initialize online softmax state
    m_i = -float('inf')  # Running max
    l_i = 0.0            # Running sum (with correction)

    # =========================================================================
    # First pass: Compute max and sum using online algorithm
    # =========================================================================
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load block
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        # Online softmax update
        m_ij = tl.max(x, axis=0)  # Max in this block
        m_new = tl.maximum(m_i, m_ij)  # New running max

        # Correction factor for previous sum
        alpha = tl.exp(m_i - m_new)
        # Sum of exp in this block (shifted to new max)
        beta = tl.exp(m_ij - m_new)

        # Update running sum with correction
        l_i = l_i * alpha + tl.sum(tl.exp(x - m_ij), axis=0) * beta

        # Update running max
        m_i = m_new

    # =========================================================================
    # Second pass: Compute and store outputs
    # =========================================================================
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load block
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        # Compute softmax output
        output = tl.exp(x - m_i) / l_i

        # Store
        tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


def softmax_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax along specified dimension using Triton.

    Args:
        x: Input tensor
        dim: Dimension to compute softmax over (default: -1, last dimension)

    Returns:
        Softmax output (same shape as input)
    """
    assert x.is_cuda, "Input must be on CUDA"

    # Handle dimension
    if dim < 0:
        dim = x.ndim + dim
    assert dim == x.ndim - 1, "Currently only supports softmax over last dimension"

    # Reshape to 2D: [batch, n_cols]
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    # Allocate output
    output = torch.empty_like(x_2d)

    # Choose block size (must be power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Cap block size for memory reasons
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Choose kernel based on row length
    if n_cols <= BLOCK_SIZE:
        # Single-pass kernel (fits in one block)
        grid = (n_rows,)
        softmax_kernel[grid](
            x_2d, output,
            x_2d.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Multi-pass online kernel (for very long rows)
        grid = (n_rows,)
        softmax_kernel_online[grid](
            x_2d, output,
            x_2d.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output.view(original_shape)


# ============================================================================
# Scaled Softmax (for attention)
# ============================================================================

@triton.jit
def scaled_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scaled softmax: softmax(x * scale)

    Used in attention: softmax(Q @ K^T / sqrt(d_k))
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # Load and scale
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row = row * scale

    # Softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def scaled_softmax_triton(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Scaled softmax for attention."""
    assert x.is_cuda and x.ndim >= 2

    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    grid = (n_rows,)
    scaled_softmax_kernel[grid](
        x_2d, output,
        x_2d.stride(0), output.stride(0),
        n_cols, scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(original_shape)


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_softmax(
    batch_size: int = 32,
    seq_len: int = 1024,
    warmup: int = 25,
    rep: int = 100,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Benchmark Triton softmax vs PyTorch."""
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = softmax_triton(x)
        _ = torch.softmax(x, dim=-1)

    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = softmax_triton(x)
        end.record()

        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    # Benchmark PyTorch
    torch_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = torch.softmax(x, dim=-1)
        end.record()

        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))

    triton_avg = sum(triton_times) / len(triton_times)
    torch_avg = sum(torch_times) / len(torch_times)

    # Calculate bandwidth
    # Naive: 3 passes (max, exp, sum/div) = 6 * elements * bytes
    # Fused: 1 pass = 2 * elements * bytes (read + write)
    elements = batch_size * seq_len
    bytes_triton = 2 * elements * x.element_size()  # Fused: 1 read, 1 write
    triton_bandwidth = bytes_triton / (triton_avg * 1e-3) / 1e9

    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'triton_ms': triton_avg,
        'torch_ms': torch_avg,
        'speedup': torch_avg / triton_avg,
        'triton_bandwidth_gbps': triton_bandwidth,
    }


def verify_correctness(batch_size: int = 32, seq_len: int = 1024, dtype: torch.dtype = torch.float32) -> bool:
    """Verify Triton softmax produces correct results."""
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=dtype)

    triton_result = softmax_triton(x)
    torch_result = torch.softmax(x, dim=-1)

    # Check sum to 1
    sums = triton_result.sum(dim=-1)
    sums_correct = torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)

    # Check values match
    values_correct = torch.allclose(triton_result, torch_result, rtol=1e-4, atol=1e-4)

    is_correct = sums_correct and values_correct

    if is_correct:
        print(f"[PASS] Correctness verified for batch={batch_size}, seq_len={seq_len}, dtype={dtype}")
    else:
        max_diff = (triton_result - torch_result).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}, Sums correct: {sums_correct}")

    return is_correct


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Fused Softmax Kernel - Triton Implementation")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. This code requires a GPU.")
        exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Verify correctness
    print("\n" + "-" * 80)
    print("Correctness Tests")
    print("-" * 80)

    for seq_len in [128, 512, 1024, 2048, 4096]:
        verify_correctness(batch_size=32, seq_len=seq_len)

    # Test online kernel for long sequences
    print("\nTesting online kernel for long sequences:")
    verify_correctness(batch_size=8, seq_len=16384)

    # Benchmark
    print("\n" + "-" * 80)
    print("Performance Benchmarks")
    print("-" * 80)

    print(f"\n{'(Batch, SeqLen)':>20} | {'Triton (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8} | {'Bandwidth':>12}")
    print("-" * 80)

    configs = [
        (32, 128),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
        (64, 2048),
        (128, 1024),
        (8, 8192),
        (4, 16384),
    ]

    for batch, seq_len in configs:
        try:
            results = benchmark_softmax(batch_size=batch, seq_len=seq_len)
            print(f"({batch}, {seq_len}):".rjust(20) + f" | {results['triton_ms']:>12.4f} | {results['torch_ms']:>12.4f} | "
                  f"{results['speedup']:>7.2f}x | {results['triton_bandwidth_gbps']:>10.1f} GB/s")
        except Exception as e:
            print(f"({batch}, {seq_len}):".rjust(20) + f" | Error: {e}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- Fused softmax reads/writes each element once (vs 3x for naive)")
    print("- Online algorithm used for sequences > BLOCK_SIZE")
    print("- This is the same algorithm used in FlashAttention!")
    print("=" * 80)
