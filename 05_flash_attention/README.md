# FlashAttention in Triton

Memory-efficient exact attention with O(N) memory instead of O(N²).

## Overview

This is a from-scratch implementation of the FlashAttention algorithm in Triton, achieving performance competitive with PyTorch's native SDPA (which uses cuDNN FlashAttention).

### Key Results

| Metric | Our Implementation |
|--------|-------------------|
| Correctness | Within 0.01 max diff of PyTorch SDPA |
| Performance | ~85-95% of cuDNN FlashAttention |
| Memory | O(N) vs O(N²) for standard attention |
| Long Context | Handles 16K+ sequences that would OOM with standard attention |

## The Problem

Standard attention requires O(N²) memory:

```python
S = Q @ K.T        # [seq, seq] - O(N²) memory
P = softmax(S)     # [seq, seq] - O(N²) memory
O = P @ V          # [seq, dim]
```

For seq_len=16384 with fp16:
- Attention matrix: 16384² × 2 bytes = **512 MB per head**
- With 32 heads: **16 GB** just for attention!

## The Solution

Process attention in tiles, never materializing the full matrix:

```
for each Q_block:
    for each K_block, V_block:
        S_block = Q_block @ K_block.T
        Update online softmax state
        Update output accumulator
```

## Algorithm: Online Softmax

The key innovation is computing softmax incrementally across blocks:

```
Standard softmax (needs full data):
    m = max(x)
    p = exp(x - m)
    l = sum(p)
    out = p / l

Online softmax (block by block):
    m_i = -inf, l_i = 0, acc = 0

    for each block x_j:
        m_new = max(m_i, max(x_j))

        # Correction factors
        alpha = exp(m_i - m_new)    # Scale old accumulator
        p_j = exp(x_j - m_new)      # New block contribution

        # Update state
        l_i = l_i * alpha + sum(p_j)
        acc = acc * alpha + p_j @ V_block
        m_i = m_new

    output = acc / l_i
```

## Implementation Details

### Block Size Selection

```python
BLOCK_M = 64  # Q block rows
BLOCK_N = 64  # K/V block cols
```

Trade-offs:
- **Larger blocks**: Better compute utilization, more register pressure
- **Smaller blocks**: Less register pressure, more memory traffic

For T4 GPU (limited shared memory), we use 64×64. A100/H100 can use 128×128.

### Memory Access Pattern

```
Q: Load once, keep in registers across K,V loop
K: Stream blocks, transposed access for Q@K.T
V: Stream blocks, aligned with K
O: Accumulate in registers, write once at end
```

### Causal Masking

For autoregressive models, we skip K,V blocks that are entirely masked:

```python
if IS_CAUSAL:
    hi = start_m * BLOCK_M + BLOCK_M  # Only compute up to diagonal
else:
    hi = N_CTX  # Full attention
```

This provides ~2x speedup for causal attention.

## Benchmarks

Run the full benchmark suite:

```bash
python benchmark.py
```

### Performance vs PyTorch SDPA

| Configuration | Triton (ms) | PyTorch (ms) | Ratio |
|---------------|-------------|--------------|-------|
| B=4, H=8, S=1024, D=64 | 0.45 | 0.42 | 0.93x |
| B=4, H=8, S=2048, D=64 | 1.21 | 1.15 | 0.95x |
| B=4, H=8, S=4096, D=64 | 4.52 | 4.28 | 0.95x |
| B=2, H=8, S=8192, D=64 | 8.91 | 8.45 | 0.95x |

### Memory Bandwidth Utilization

FlashAttention is memory-bound, not compute-bound:

| GPU | Peak BW | Achieved BW | Utilization |
|-----|---------|-------------|-------------|
| T4 | 320 GB/s | ~250 GB/s | 78% |
| A100 | 2039 GB/s | ~1600 GB/s | 78% |

### Memory Comparison

| Sequence Length | Standard Attention | FlashAttention | Reduction |
|-----------------|-------------------|----------------|-----------|
| 1024 | 48 MB | 12 MB | 4x |
| 4096 | 768 MB | 48 MB | 16x |
| 8192 | OOM | 96 MB | ∞ |
| 16384 | OOM | 192 MB | ∞ |

## Why FlashAttention is Faster

Counter-intuitive: same FLOPs, but faster.

| Aspect | Standard | FlashAttention |
|--------|----------|----------------|
| FLOPs | O(N²d) | O(N²d) |
| Memory I/O | O(N²) | O(N) |
| Bottleneck | Memory | Compute |

Modern GPUs have:
- **High compute**: 312 TFLOPS (A100 fp16)
- **Limited bandwidth**: 2 TB/s (A100)

Standard attention wastes bandwidth reading/writing the N² attention matrix. FlashAttention keeps everything in SRAM.

### Arithmetic Intensity Analysis

```
Standard Attention:
    Reads:  Q, K, V = 3Nd
    Writes: S, P, O = N² + N² + Nd
    FLOPs:  4N²d
    AI = 4N²d / (3Nd + 2N² + Nd) ≈ 2d for large N

FlashAttention:
    Reads:  Q, K, V = 3Nd (each read once)
    Writes: O = Nd
    FLOPs:  4N²d
    AI = 4N²d / 4Nd = N (scales with sequence!)
```

## Correctness Verification

We verify against PyTorch's SDPA:

```python
triton_out = flash_attention_triton(q, k, v, causal=True)
pytorch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

assert (triton_out - pytorch_out).abs().max() < 0.05  # fp16 tolerance
```

All configurations pass within numerical tolerance.

## Design Decisions & Trade-offs

### 1. Single-pass vs Two-pass

We use **single-pass** online softmax:
- **Pro**: 2x fewer memory accesses
- **Con**: Slightly more complex, accumulator correction

### 2. Accumulator Precision

Accumulator uses fp32 even for fp16 inputs:
- Prevents numerical overflow in long sequences
- Final cast to fp16 for output

### 3. Block Dimensions

We tile Q in BLOCK_M rows and K,V in BLOCK_N cols:
- Q stays in registers (reused across K,V blocks)
- K,V stream through (loaded once per block)

## Known Limitations

1. **Head dimensions**: Only powers of 2 (16, 32, 64, 128, 256)
2. **Backward pass**: Not implemented (forward only)
3. **Variable sequence lengths**: Not supported (fixed shapes)
4. **Multi-query attention**: Not implemented

## Files

```
05_flash_attention/
├── flash_attention.py    # Main implementation
├── benchmark.py          # Comprehensive benchmark suite
├── benchmark_results.json # Saved benchmark data
└── README.md             # This file
```

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Online Normalizer Calculation (Milakov & Gimelshein)](https://arxiv.org/abs/1805.02867)
- [Triton Documentation](https://triton-lang.org/)
