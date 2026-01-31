"""
FlashAttention Kernel in Triton - FIXED VERSION
================================================

Fixes from v1:
1. Corrected stride/offset calculation for batch*heads indexing
2. Fixed accumulator update in online softmax (removed erroneous l_i multiplication)
3. Improved numerical stability

Key Bug Fixes:
- The accumulator should be: acc = acc * alpha + P @ V (not acc * l_i * alpha)
- The l_i normalization only happens at the end
- Proper handling of batch and head dimension offsets
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def flash_attention_v2_kernel(
    Q, K, V, Out,
    L,  # logsumexp for backward
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z,  # batch size
    H,  # number of heads
    N_CTX,  # sequence length
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention forward pass - FIXED VERSION.

    Grid: (cdiv(N_CTX, BLOCK_M), batch * n_heads)

    Key fix: Accumulator update is acc = acc * alpha + P @ V
    NOT acc = acc * l_i * alpha + P @ V
    """
    # Program indices
    start_m = tl.program_id(0)  # Which Q block
    off_hz = tl.program_id(1)   # Which batch*head

    # Compute batch and head indices properly
    off_z = off_hz // H  # batch index
    off_h = off_hz % H   # head index

    # Compute base offset for this batch/head combination
    # Q/K/V shape: [batch, heads, seq, head_dim]
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # Q block pointers
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + qkv_offset + offs_d[None, :] * stride_kk  # Will add n offset in loop
    v_ptrs = V + qkv_offset + offs_d[None, :] * stride_vk  # Will add n offset in loop

    # Load Q block (stays in registers)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-6  # Small epsilon to avoid div by zero
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Determine loop bounds
    if IS_CAUSAL:
        # Only process K positions <= Q positions
        end_n = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        end_n = N_CTX

    # Main loop over K, V blocks
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block [BLOCK_N, BLOCK_DMODEL]
        k_block_ptrs = k_ptrs + (start_n + offs_n[:, None]) * stride_kn
        k = tl.load(k_block_ptrs,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)

        # Compute attention scores: S = Q @ K^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Mask out-of-bounds
        qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # Max in this block
        m_new = tl.maximum(m_i, m_ij)  # New running max

        # Compute correction factor for old values
        alpha = tl.exp(m_i - m_new)

        # Compute P = exp(qk - m_new) for this block
        p = tl.exp(qk - m_new[:, None])

        # Update running sum: l_new = l_old * alpha + sum(p)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # Update accumulator: acc = acc * alpha + P @ V
        # KEY FIX: Only multiply by alpha, not by l_i!
        acc = acc * alpha[:, None]

        # Load V block and accumulate
        v_block_ptrs = v_ptrs + (start_n + offs_n[:, None]) * stride_vn
        v = tl.load(v_block_ptrs,
                    mask=(start_n + offs_n[:, None]) < N_CTX,
                    other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        # Update state
        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store logsumexp for backward pass
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)

    # Store output
    out_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FlashAttention v2 - Fixed implementation.

    Args:
        q: Query [batch, heads, seq_len, head_dim]
        k: Key [batch, heads, seq_len, head_dim]
        v: Value [batch, heads, seq_len, head_dim]
        causal: Apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output [batch, heads, seq_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4

    batch, n_heads, seq_len, head_dim = q.shape

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Allocate output
    o = torch.empty_like(q)
    L = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)

    # Block sizes - tuned for correctness first
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (num_q_blocks, batch * n_heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    num_warps = 4
    num_stages = 2

    flash_attention_v2_kernel[grid](
        q, k, v, o, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, n_heads, seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o


def standard_attention(q, k, v, causal=False, sm_scale=None):
    """Standard O(N^2) attention for comparison."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))

    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# ============================================================================
# Testing
# ============================================================================

def test_correctness():
    """Test FlashAttention correctness."""
    print("=" * 60)
    print("FlashAttention v2 - Correctness Tests")
    print("=" * 60)

    configs = [
        (2, 4, 64, 32, False),   # Small, non-causal
        (2, 4, 64, 32, True),    # Small, causal
        (2, 4, 128, 64, False),  # Medium, non-causal
        (2, 4, 128, 64, True),   # Medium, causal
        (4, 8, 256, 64, True),   # Larger, causal
        (2, 8, 512, 64, True),   # Even larger
    ]

    all_passed = True

    for batch, heads, seq, dim, causal in configs:
        q = torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        flash_out = flash_attention_v2(q, k, v, causal=causal)
        std_out = standard_attention(q, k, v, causal=causal)

        # Check correctness
        max_diff = (flash_out - std_out).abs().max().item()
        is_correct = torch.allclose(flash_out, std_out, rtol=1e-2, atol=1e-2)

        status = "✓ PASS" if is_correct else "✗ FAIL"
        all_passed = all_passed and is_correct

        print(f"{status} | batch={batch}, heads={heads}, seq={seq}, dim={dim}, causal={causal} | max_diff={max_diff:.6f}")

    return all_passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    passed = test_correctness()

    if passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
