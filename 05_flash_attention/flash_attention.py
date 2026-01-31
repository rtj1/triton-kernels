"""
FlashAttention Kernel in Triton
===============================

Implements FlashAttention: memory-efficient exact attention with O(N) memory instead of O(N^2).

Key Innovation:
- Standard attention materializes the full N×N attention matrix (quadratic memory)
- FlashAttention computes attention in tiles, never materializing the full matrix
- Uses online softmax to accumulate results across tiles

This is the algorithm that enables training with very long sequences (100K+ tokens).

Algorithm Overview:
```
Standard Attention:
    S = Q @ K^T           # O(N^2) memory
    P = softmax(S)        # O(N^2) memory
    O = P @ V             # O(N^2) memory

FlashAttention:
    for each Q block:
        for each K, V block:
            S_block = Q_block @ K_block^T
            Update online softmax state
            Update output accumulator
```

Memory Complexity:
- Standard: O(N^2) for attention matrix
- FlashAttention: O(N) - only block-sized tiles in SRAM

Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
       https://arxiv.org/abs/2205.14135
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def flash_attention_forward_kernel(
    # Pointers to Q, K, V, Output
    Q, K, V, Out,
    # Softmax statistics for backward pass
    L,  # logsumexp
    # Dimensions
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Sequence lengths and head dim
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    # Scaling factor
    sm_scale,
    # Causal masking
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention forward pass kernel.

    Each program computes one tile of the output matrix.
    Grid: (batch * n_heads, cdiv(N_CTX, BLOCK_M))

    The key insight is computing softmax in an "online" manner:
    - Process K,V in blocks
    - Maintain running max (m) and sum (l) for softmax normalization
    - Correct previous outputs when max changes
    """
    # =========================================================================
    # Program indices
    # =========================================================================
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Calculate batch and head indices
    off_z = off_hz // stride_qh  # Not used directly, but for understanding
    off_h = off_hz % stride_qh   # Not used directly

    # Initialize pointers to Q, K, V for this batch/head
    # Q: [batch, heads, seq, head_dim]
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    # =========================================================================
    # Load Q tile for this program
    # =========================================================================
    # Offsets for Q rows this program processes
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers to Q[start_m*BLOCK_M : (start_m+1)*BLOCK_M, :]
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # Load Q tile (stays in registers for all K,V iterations)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # =========================================================================
    # Initialize online softmax state
    # =========================================================================
    # m_i: running max for numerical stability
    # l_i: running sum of exp(x - m)
    # acc: running weighted sum (output accumulator)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # =========================================================================
    # Main loop: iterate over K, V blocks
    # =========================================================================
    # For causal attention, only process K blocks where k_idx <= q_idx
    if IS_CAUSAL:
        hi = start_m * BLOCK_M + BLOCK_M
        hi = min(hi, N_CTX)
    else:
        hi = N_CTX

    lo = 0

    # K, V block offsets
    offs_n = tl.arange(0, BLOCK_N)

    # Pointers to K and V (will be advanced in loop)
    k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    # Iterate over K, V blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -----------------------------------------------------------------
        # Load K block and compute attention scores
        # -----------------------------------------------------------------
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)

        # S = Q @ K^T, scaled
        # qk shape: [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # -----------------------------------------------------------------
        # Apply causal mask if needed
        # -----------------------------------------------------------------
        if IS_CAUSAL:
            # Mask out positions where k_idx > q_idx
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Mask out-of-bounds positions
        qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float('-inf'))

        # -----------------------------------------------------------------
        # Online softmax update
        # -----------------------------------------------------------------
        # Find max in this block
        m_ij = tl.max(qk, axis=1)

        # New running max
        m_new = tl.maximum(m_i, m_ij)

        # Correction factors
        alpha = tl.exp(m_i - m_new)  # Correction for old accumulator
        beta = tl.exp(m_ij - m_new)  # Scale for new block

        # Update running sum
        # l_new = l_old * alpha + sum(exp(qk - m_new))
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # -----------------------------------------------------------------
        # Update output accumulator
        # -----------------------------------------------------------------
        # Correct old accumulator and add new contribution
        # acc = acc * (l_i * alpha / l_new) + p @ v / l_new
        #     = (acc * l_i * alpha + p @ v) / l_new

        # Scale old accumulator
        acc = acc * (l_i[:, None] * alpha[:, None])

        # Load V block
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)

        # Add new contribution: p @ V
        acc += tl.dot(p.to(v.dtype), v)

        # Update state
        l_i = l_new
        m_i = m_new

    # =========================================================================
    # Finalize: normalize by sum
    # =========================================================================
    acc = acc / l_i[:, None]

    # =========================================================================
    # Store output
    # =========================================================================
    # Store logsumexp for backward pass
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)

    # Store output
    out_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FlashAttention forward pass.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal masking
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim)
    """
    # Validate inputs
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4, "Expected 4D tensor (batch, heads, seq, head_dim)"

    batch, n_heads, seq_len, head_dim = q.shape

    # Default scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Allocate output and logsumexp
    o = torch.empty_like(q)
    L = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim

    # Ensure head_dim is power of 2 for efficiency
    assert head_dim in [16, 32, 64, 128, 256], f"head_dim must be power of 2, got {head_dim}"

    # Grid: (num_q_blocks, batch * n_heads)
    num_warps = 4 if head_dim <= 64 else 8
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    flash_attention_forward_kernel[grid](
        q, k, v, o,
        L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=2,
    )

    return o


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard attention for comparison (O(N^2) memory).
    """
    batch, n_heads, seq_len, head_dim = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Apply causal mask
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

    # Softmax and output
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    return output


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_attention(
    batch_size: int = 4,
    n_heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    causal: bool = True,
    warmup: int = 10,
    rep: int = 50,
) -> dict:
    """Benchmark FlashAttention vs standard attention."""
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(warmup):
        _ = flash_attention_triton(q, k, v, causal=causal)
        if seq_len <= 4096:  # Standard attention runs out of memory for long sequences
            _ = standard_attention(q, k, v, causal=causal)

    torch.cuda.synchronize()

    # Benchmark FlashAttention
    flash_times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = flash_attention_triton(q, k, v, causal=causal)
        end.record()

        torch.cuda.synchronize()
        flash_times.append(start.elapsed_time(end))

    flash_avg = sum(flash_times) / len(flash_times)

    # Benchmark standard attention (if sequence isn't too long)
    torch_avg = None
    if seq_len <= 4096:
        torch_times = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = standard_attention(q, k, v, causal=causal)
            end.record()

            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))

        torch_avg = sum(torch_times) / len(torch_times)

    # Calculate memory for standard attention
    attn_memory_mb = batch_size * n_heads * seq_len * seq_len * 2 / (1024 * 1024)  # fp16

    # Calculate FLOPS
    # Attention: 2 * batch * heads * seq^2 * head_dim (Q@K^T and attn@V)
    flops = 4 * batch_size * n_heads * seq_len * seq_len * head_dim
    flash_tflops = flops / (flash_avg * 1e-3) / 1e12

    return {
        'batch_size': batch_size,
        'n_heads': n_heads,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'flash_ms': flash_avg,
        'standard_ms': torch_avg,
        'speedup': torch_avg / flash_avg if torch_avg else None,
        'flash_tflops': flash_tflops,
        'attn_memory_mb': attn_memory_mb,
    }


def verify_correctness(
    batch_size: int = 2,
    n_heads: int = 4,
    seq_len: int = 256,
    head_dim: int = 64,
    causal: bool = True,
) -> bool:
    """Verify FlashAttention produces correct results."""
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    flash_result = flash_attention_triton(q, k, v, causal=causal)
    standard_result = standard_attention(q, k, v, causal=causal)

    # Use larger tolerance for fp16
    is_correct = torch.allclose(flash_result, standard_result, rtol=1e-2, atol=1e-2)

    if is_correct:
        print(f"[PASS] FlashAttention correctness verified: batch={batch_size}, heads={n_heads}, "
              f"seq={seq_len}, head_dim={head_dim}, causal={causal}")
    else:
        max_diff = (flash_result - standard_result).abs().max().item()
        print(f"[FAIL] Max difference: {max_diff}")

    return is_correct


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print("FlashAttention Kernel - Triton Implementation")
    print("=" * 90)

    if not torch.cuda.is_available():
        print("CUDA not available. This code requires a GPU.")
        exit(1)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Verify correctness
    print("\n" + "-" * 90)
    print("Correctness Tests")
    print("-" * 90)

    for causal in [False, True]:
        for seq_len in [128, 256, 512, 1024]:
            verify_correctness(batch_size=2, n_heads=4, seq_len=seq_len, head_dim=64, causal=causal)

    # Benchmark
    print("\n" + "-" * 90)
    print("Performance Benchmarks (Causal Attention)")
    print("-" * 90)

    print(f"\n{'Config':>30} | {'Flash (ms)':>10} | {'Standard (ms)':>12} | {'Speedup':>8} | {'TFLOPS':>8} | {'Attn Mem':>10}")
    print("-" * 100)

    configs = [
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
        (4, 8, 4096, 64),
        (2, 8, 8192, 64),    # Standard attention would OOM
        (1, 8, 16384, 64),   # Only Flash can handle this
    ]

    for batch, heads, seq, dim in configs:
        try:
            results = benchmark_attention(batch_size=batch, n_heads=heads, seq_len=seq, head_dim=dim, causal=True)
            config_str = f"({batch}, {heads}, {seq}, {dim})"

            std_ms = f"{results['standard_ms']:.3f}" if results['standard_ms'] else "OOM"
            speedup = f"{results['speedup']:.2f}x" if results['speedup'] else "N/A"

            print(f"{config_str:>30} | {results['flash_ms']:>10.3f} | {std_ms:>12} | {speedup:>8} | "
                  f"{results['flash_tflops']:>7.1f} | {results['attn_memory_mb']:>9.1f} MB")
        except Exception as e:
            print(f"{config_str:>30} | Error: {e}")

    print("\n" + "=" * 90)
    print("Key Insights:")
    print("-" * 90)
    print("1. FlashAttention uses O(N) memory vs O(N^2) for standard attention")
    print("2. Enables training with very long sequences (100K+ tokens)")
    print("3. Uses online softmax algorithm to process attention in tiles")
    print("4. Speedup comes from reduced memory bandwidth, not fewer FLOPs")
    print("5. This is the algorithm behind efficient LLM training and inference")
    print("=" * 90)
