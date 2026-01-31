# 02 - Matrix Multiplication (GEMM)

High-performance General Matrix Multiply using Triton.

## Concepts Covered

1. **2D Tiling**: Output matrix divided into tiles, each computed by one program
2. **K-dimension Reduction**: Iterate over inner dimension in chunks
3. **Autotuning**: `@triton.autotune` to find optimal block sizes
4. **Group Scheduling**: Programs grouped for better L2 cache reuse
5. **Accumulator Pattern**: Accumulate in fp32, store in fp16

## Algorithm

```
C[M, N] = A[M, K] @ B[K, N]

For each output tile C[m:m+BLOCK_M, n:n+BLOCK_N]:
    accumulator = zeros(BLOCK_M, BLOCK_N)
    for k in range(0, K, BLOCK_K):
        a_tile = A[m:m+BLOCK_M, k:k+BLOCK_K]
        b_tile = B[k:k+BLOCK_K, n:n+BLOCK_N]
        accumulator += a_tile @ b_tile
    C[m:m+BLOCK_M, n:n+BLOCK_N] = accumulator
```

## Tiling Visualization

```
        BLOCK_K
       ┌──────┐
BLOCK_M│  A   │────┐
       └──────┘    │
                   ▼
              ┌──────┐BLOCK_N
       BLOCK_K│  B   │
              └──────┘
                   │
                   ▼
              ┌──────┐
       BLOCK_M│  C   │BLOCK_N  (accumulated)
              └──────┘
```

## Performance Analysis

Matrix multiplication is **compute-bound** for large matrices:
- FLOPs = 2 * M * N * K
- Memory = (M*K + K*N + M*N) * element_size
- Arithmetic intensity = 2*M*N*K / ((M*K + K*N + M*N) * bytes)

For large square matrices (M=N=K=4096):
- Intensity ≈ 341 FLOP/byte
- Peak: ~312 TFLOPS (A100), ~82 TFLOPS (RTX 3090)

## Autotuning

The kernel uses `@triton.autotune` to select optimal parameters:
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K`: Tile dimensions
- `num_stages`: Software pipelining depth
- `num_warps`: Parallelism within each tile
- `GROUP_SIZE_M`: Programs per L2 cache group

## Usage

```python
from matmul import matmul_triton

a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
c = matmul_triton(a, b)
```

## Expected Performance

| Size | Triton | cuBLAS | Efficiency |
|------|--------|--------|------------|
| 1024 | ~0.15ms | ~0.12ms | ~80% |
| 4096 | ~3.5ms | ~3.2ms | ~90% |
| 8192 | ~25ms | ~23ms | ~92% |

(Actual numbers depend on GPU model)

## Run Benchmarks

```bash
python matmul.py
```
