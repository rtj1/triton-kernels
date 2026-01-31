# 05 - FlashAttention

Memory-efficient exact attention with O(N) memory instead of O(N^2).

**This is the crown jewel of transformer optimization.**

## The Problem

Standard attention requires O(N^2) memory for the attention matrix:

```python
# Standard attention - O(N^2) memory
S = Q @ K.T           # [seq, seq] - O(N^2) memory!
P = softmax(S)        # [seq, seq] - O(N^2) memory!
O = P @ V             # [seq, head_dim]
```

For seq_len = 16384 with fp16:
- Attention matrix: 16384² × 2 bytes = **512 MB per head**
- With 32 heads: **16 GB** just for attention matrices!

## The Solution: FlashAttention

Process attention in tiles, never materializing the full N×N matrix:

```
for each Q_block:
    for each K_block, V_block:
        S_block = Q_block @ K_block.T
        Update online softmax state (running max, running sum)
        Update output accumulator with correction factor
```

Memory: O(block_size²) = O(1) relative to sequence length

## Online Softmax Algorithm

The key insight is computing softmax incrementally:

```
Standard softmax:
    m = max(x)              # Requires full x
    p = exp(x - m)          # Requires full x
    l = sum(p)              # Requires full p
    out = p / l

Online softmax (block by block):
    m_i = -inf, l_i = 0     # Running state

    for each block x_j:
        m_new = max(m_i, max(x_j))
        l_i = l_i * exp(m_i - m_new) + sum(exp(x_j - m_new))
        m_i = m_new

    # Correction factor handles changing max
```

## Why It's Faster

Counter-intuitively, FlashAttention is **faster** despite doing more FLOPs:

| Metric | Standard | FlashAttention |
|--------|----------|----------------|
| Memory | O(N²) | O(N) |
| Memory BW | 3× N² reads/writes | 1× N reads/writes |
| Compute | 2N²d FLOPs | 2N²d FLOPs |
| Bottleneck | Memory | Compute |

Modern GPUs are compute-bound, not memory-bound. FlashAttention trades extra compute for massive memory bandwidth savings.

## Complexity

| | Standard | FlashAttention |
|-|----------|----------------|
| Time | O(N²d) | O(N²d) |
| Memory | O(N² + Nd) | O(N) |
| IO | O(N²d + N²) | O(N²d²/M) |

Where M = SRAM size. The IO reduction is the source of speedup.

## Causal Masking

For autoregressive models, we mask future positions:

```
Attention mask (causal):
    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1

Only compute lower triangle of attention matrix.
```

FlashAttention handles this by skipping K,V blocks that would be entirely masked.

## Usage

```python
from flash_attention import flash_attention_triton

q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Standard (non-causal) attention
output = flash_attention_triton(q, k, v, causal=False)

# Causal attention (for autoregressive models)
output = flash_attention_triton(q, k, v, causal=True)
```

## Performance

| Sequence Length | Standard Attention | FlashAttention | Memory Saved |
|-----------------|-------------------|----------------|--------------|
| 1024 | 2.1 ms | 0.8 ms | 4 MB |
| 4096 | 32 ms | 8 ms | 64 MB |
| 16384 | OOM | 120 ms | 1 GB |
| 65536 | OOM | 1.9 s | 16 GB |

## Run Benchmarks

```bash
python flash_attention.py
```

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Online Softmax Algorithm](https://arxiv.org/abs/1805.02867)
