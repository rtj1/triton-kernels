# 04 - Fused Layer Normalization

Layer Normalization and RMSNorm with kernel fusion.

## Concepts Covered

1. **Welford's Algorithm**: Numerically stable mean/variance computation
2. **Kernel Fusion**: Mean, variance, normalize, scale, shift in one kernel
3. **Learnable Parameters**: Gamma (scale) and beta (shift)
4. **RMSNorm**: Simplified normalization used in modern LLMs

## LayerNorm Formula

```
mean = sum(x) / N
var = sum((x - mean)^2) / N
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

## RMSNorm Formula (LLaMA, Gemma)

```
rms = sqrt(mean(x^2) + eps)
y = x / rms * gamma
```

RMSNorm is faster (no mean computation) and works just as well in practice.

## Welford's Online Algorithm

For numerically stable variance computation:

```python
mean = 0
M2 = 0  # sum of squared differences
count = 0

for x in data:
    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean
    M2 += delta * delta2

variance = M2 / count
```

Benefits:
- Single pass through data
- Numerically stable (avoids catastrophic cancellation)
- Works for streaming data

## Why Fused LayerNorm?

**Naive Implementation:**
```python
mean = x.mean(dim=-1, keepdim=True)      # Pass 1
var = ((x - mean) ** 2).mean(dim=-1)      # Pass 2
y = (x - mean) / sqrt(var + eps)          # Pass 3
y = y * gamma + beta                       # Pass 4
```

**Fused Implementation:**
All operations in single kernel = single memory pass

## Usage

```python
from layernorm import layernorm_triton, rmsnorm_triton

x = torch.randn(32, 512, 768, device='cuda')
gamma = torch.ones(768, device='cuda')
beta = torch.zeros(768, device='cuda')

# LayerNorm
y = layernorm_triton(x, gamma, beta)

# RMSNorm (LLaMA-style)
y = rmsnorm_triton(x, gamma)
```

## Run Benchmarks

```bash
python layernorm.py
```
