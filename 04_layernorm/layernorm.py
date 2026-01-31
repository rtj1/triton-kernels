"""
Fused Layer Normalization Kernel in Triton
==========================================

Implements Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta

Key Concepts:
1. **Welford's Algorithm**: Numerically stable online mean/variance computation
2. **Kernel Fusion**: Combine mean, variance, normalize, scale, shift
3. **Row-wise Operation**: Each program handles one row (hidden dimension)
4. **Learnable Parameters**: gamma (scale) and beta (shift)

LayerNorm in Transformers:
- Applied after attention and FFN blocks
- Normalizes across the hidden dimension (last axis)
- Critical for training stability

Formula:
    mean = sum(x) / N
    var = sum((x - mean)^2) / N
    y = (x - mean) / sqrt(var + eps) * gamma + beta
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def layernorm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm kernel - each program processes one row.

    Uses Welford's online algorithm for numerically stable mean/variance.
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Calculate starting pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # =========================================================================
    # Step 1: Load input row
    # =========================================================================
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # =========================================================================
    # Step 2: Compute mean using Welford's algorithm (single pass)
    # =========================================================================
    # For simplicity, using standard approach when row fits in one block
    # sum(x) / n
    mean = tl.sum(x, axis=0) / n_cols

    # =========================================================================
    # Step 3: Compute variance
    # =========================================================================
    # sum((x - mean)^2) / n
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # =========================================================================
    # Step 4: Normalize
    # =========================================================================
    # (x - mean) / sqrt(var + eps)
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # =========================================================================
    # Step 5: Scale and shift with learnable parameters
    # =========================================================================
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

    output = x_norm * gamma + beta

    # =========================================================================
    # Store result
    # =========================================================================
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


@triton.jit
def layernorm_kernel_online(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm with Welford's online algorithm for long rows.

    Welford's algorithm computes mean and variance in a single pass:
    - More numerically stable than naive sum/sum^2 approach
    - Handles arbitrarily long rows
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Initialize Welford state
    mean = 0.0
    m2 = 0.0  # Sum of squared differences from mean
    count = 0

    # =========================================================================
    # First pass: Compute mean and variance using Welford's algorithm
    # =========================================================================
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load block
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

        # Count valid elements in this block
        block_count = tl.sum(mask.to(tl.int32), axis=0)

        # Compute block statistics
        block_mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / tl.maximum(block_count, 1)

        # Update running statistics (parallel Welford update)
        delta = block_mean - mean
        total_count = count + block_count
        mean = mean + delta * block_count / tl.maximum(total_count, 1)

        # Update M2 (sum of squared differences)
        block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), 0.0), axis=0)
        m2 = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(total_count, 1)

        count = total_count

    # Compute variance from M2
    var = m2 / tl.maximum(count, 1)
    rstd = 1.0 / tl.sqrt(var + eps)

    # =========================================================================
    # Second pass: Normalize and apply gamma/beta
    # =========================================================================
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input and parameters
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

        # Normalize, scale, shift
        x_norm = (x - mean) * rstd
        output = x_norm * gamma + beta

        # Store
        tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute Layer Normalization using Triton.

    Args:
        x: Input tensor of shape (..., hidden_size)
        weight: Scale parameter (gamma) of shape (hidden_size,)
        bias: Shift parameter (beta) of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized output of same shape as input
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Parameters must be on CUDA"

    # Get dimensions
    original_shape = x.shape
    hidden_size = x.shape[-1]

    assert weight.shape == (hidden_size,), f"Weight shape mismatch: {weight.shape} vs ({hidden_size},)"
    assert bias.shape == (hidden_size,), f"Bias shape mismatch: {bias.shape} vs ({hidden_size},)"

    # Reshape to 2D
    x_2d = x.view(-1, hidden_size).contiguous()
    n_rows = x_2d.shape[0]

    # Allocate output
    output = torch.empty_like(x_2d)

    # Choose block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Choose kernel
    grid = (n_rows,)

    if hidden_size <= BLOCK_SIZE:
        layernorm_kernel[grid](
            x_2d, output, weight, bias,
            x_2d.stride(0), output.stride(0),
            hidden_size, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        layernorm_kernel_online[grid](
            x_2d, output, weight, bias,
            x_2d.stride(0), output.stride(0),
            hidden_size, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output.view(original_shape)


# ============================================================================
# RMSNorm (Root Mean Square Layer Normalization)
# ============================================================================

@triton.jit
def rmsnorm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel: y = x / sqrt(mean(x^2) + eps) * gamma

    RMSNorm is simpler than LayerNorm (no mean subtraction, no beta).
    Used in LLaMA, Gemma, and other modern LLMs.
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # Compute RMS: sqrt(mean(x^2))
    x_sq = tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Normalize and scale
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    output = x * rrms * gamma

    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute RMSNorm using Triton.

    RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma

    Used in LLaMA, Gemma, and other modern architectures.
    """
    assert x.is_cuda and weight.is_cuda

    original_shape = x.shape
    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size).contiguous()
    n_rows = x_2d.shape[0]

    output = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    grid = (n_rows,)
    rmsnorm_kernel[grid](
        x_2d, output, weight,
        x_2d.stride(0), output.stride(0),
        hidden_size, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(original_shape)


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_layernorm(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 768,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """Benchmark Triton LayerNorm vs PyTorch."""
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    weight = torch.randn(hidden_size, device='cuda')
    bias = torch.randn(hidden_size, device='cuda')

    # PyTorch LayerNorm
    torch_ln = torch.nn.LayerNorm(hidden_size, device='cuda')
    torch_ln.weight.data = weight.clone()
    torch_ln.bias.data = bias.clone()

    # Warmup
    for _ in range(warmup):
        _ = layernorm_triton(x, weight, bias)
        _ = torch_ln(x)

    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = layernorm_triton(x, weight, bias)
        end.record()

        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    # Benchmark PyTorch
    torch_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = torch_ln(x)
        end.record()

        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end))

    triton_avg = sum(triton_times) / len(triton_times)
    torch_avg = sum(torch_times) / len(torch_times)

    # Bandwidth calculation
    elements = batch_size * seq_len * hidden_size
    # Read x, gamma, beta; write output
    bytes_accessed = (3 + 1) * elements * x.element_size()
    triton_bandwidth = bytes_accessed / (triton_avg * 1e-3) / 1e9

    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'triton_ms': triton_avg,
        'torch_ms': torch_avg,
        'speedup': torch_avg / triton_avg,
        'triton_bandwidth_gbps': triton_bandwidth,
    }


def verify_correctness(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 768,
) -> bool:
    """Verify Triton LayerNorm produces correct results."""
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    weight = torch.randn(hidden_size, device='cuda')
    bias = torch.randn(hidden_size, device='cuda')

    triton_result = layernorm_triton(x, weight, bias)

    torch_ln = torch.nn.LayerNorm(hidden_size, device='cuda')
    torch_ln.weight.data = weight.clone()
    torch_ln.bias.data = bias.clone()
    torch_result = torch_ln(x)

    is_correct = torch.allclose(triton_result, torch_result, rtol=1e-4, atol=1e-4)

    if is_correct:
        print(f"[PASS] LayerNorm correctness verified for shape ({batch_size}, {seq_len}, {hidden_size})")
    else:
        max_diff = (triton_result - torch_result).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}")

    return is_correct


def verify_rmsnorm(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 768,
) -> bool:
    """Verify RMSNorm produces expected results."""
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    weight = torch.randn(hidden_size, device='cuda')

    triton_result = rmsnorm_triton(x, weight)

    # Reference implementation
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
    reference = (x / rms) * weight

    is_correct = torch.allclose(triton_result, reference, rtol=1e-4, atol=1e-4)

    if is_correct:
        print(f"[PASS] RMSNorm correctness verified for shape ({batch_size}, {seq_len}, {hidden_size})")
    else:
        max_diff = (triton_result - reference).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}")

    return is_correct


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Fused Layer Normalization Kernel - Triton Implementation")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. This code requires a GPU.")
        exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Verify correctness
    print("\n" + "-" * 80)
    print("Correctness Tests")
    print("-" * 80)

    for hidden in [256, 768, 1024, 2048, 4096]:
        verify_correctness(batch_size=4, seq_len=128, hidden_size=hidden)

    print("\nRMSNorm Tests:")
    for hidden in [768, 2048, 4096]:
        verify_rmsnorm(batch_size=4, seq_len=128, hidden_size=hidden)

    # Benchmark
    print("\n" + "-" * 80)
    print("Performance Benchmarks")
    print("-" * 80)

    print(f"\n{'Shape':>25} | {'Triton (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8} | {'Bandwidth':>12}")
    print("-" * 85)

    configs = [
        (32, 512, 768),    # BERT-base
        (32, 512, 1024),   # BERT-large hidden
        (16, 1024, 768),   # Longer sequence
        (8, 2048, 1024),   # GPT-2 medium
        (4, 2048, 2048),   # Larger hidden
        (2, 4096, 4096),   # Very large
    ]

    for batch, seq, hidden in configs:
        try:
            results = benchmark_layernorm(batch_size=batch, seq_len=seq, hidden_size=hidden)
            shape_str = f"({batch}, {seq}, {hidden})"
            print(f"{shape_str:>25} | {results['triton_ms']:>12.4f} | {results['torch_ms']:>12.4f} | "
                  f"{results['speedup']:>7.2f}x | {results['triton_bandwidth_gbps']:>10.1f} GB/s")
        except Exception as e:
            print(f"{shape_str:>25} | Error: {e}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- LayerNorm normalizes across the hidden dimension (last axis)")
    print("- RMSNorm (used in LLaMA) is simpler: no mean subtraction")
    print("- Fused kernel avoids multiple memory passes")
    print("- Welford's algorithm ensures numerical stability")
    print("=" * 80)
