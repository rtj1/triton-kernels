# 03 - Fused Softmax

Numerically stable softmax with kernel fusion.

## Concepts Covered

1. **Numerical Stability**: Subtract max before exp to prevent overflow
2. **Kernel Fusion**: Combine max, exp, sum, divide into single kernel
3. **Online Algorithm**: Process arbitrarily long rows in chunks
4. **Reduction Operations**: Max and sum across a dimension

## Why Fused Softmax?

**Naive Implementation (3 kernel launches):**
```python
x_max = x.max(dim=-1, keepdim=True)      # Kernel 1: Read x, write max
x_exp = (x - x_max).exp()                 # Kernel 2: Read x, max, write exp
x_softmax = x_exp / x_exp.sum(dim=-1)     # Kernel 3: Read exp, write output
```
Memory accesses: 6 reads + 3 writes = 9 passes through data

**Fused Implementation (1 kernel launch):**
```python
# All in one kernel:
# 1. Load x, find max
# 2. Compute exp(x - max)
# 3. Sum exp values
# 4. Divide and store
```
Memory accesses: 1 read + 1 write = 2 passes through data

**Result: 4.5x reduction in memory traffic!**

## Online Softmax Algorithm

For rows longer than block size, we use the online algorithm:

```
m_i = -inf  # running max
l_i = 0     # running sum (corrected)

for each block:
    x_block = load(block)
    m_new = max(m_i, max(x_block))

    # Correction factor for previous sum
    alpha = exp(m_i - m_new)

    # Update sum with correction
    l_i = l_i * alpha + sum(exp(x_block - m_new))
    m_i = m_new

# Final output
output = exp(x - m_i) / l_i
```

This is the same algorithm used in **FlashAttention**!

## Performance Analysis

Softmax is **memory-bound**:
- Very few FLOPs per element: max, sub, exp, sum, div
- Dominated by memory access time
- Fusion provides ~2-4x speedup by reducing memory traffic

## Usage

```python
from softmax import softmax_triton, scaled_softmax_triton

x = torch.randn(32, 1024, device='cuda')
y = softmax_triton(x)

# For attention: softmax(QK^T / sqrt(d))
scale = 1.0 / math.sqrt(64)
attn = scaled_softmax_triton(qk, scale=scale)
```

## Run Benchmarks

```bash
python softmax.py
```
