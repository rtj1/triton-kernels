# 01 - Vector Addition

The "Hello World" of GPU programming.

## Concepts Covered

1. **Triton Kernel Structure**: `@triton.jit` decorator
2. **Program ID**: `tl.program_id(axis=0)` - unique identifier for each kernel instance
3. **Block Processing**: Each kernel instance processes `BLOCK_SIZE` elements
4. **Memory Operations**: `tl.load()` and `tl.store()` with masking
5. **Grid Launch**: Calculating how many kernel instances to launch

## Memory Access Pattern

```
Vector A:  [a0, a1, a2, a3, a4, a5, a6, a7, a8, ...]
Vector B:  [b0, b1, b2, b3, b4, b5, b6, b7, b8, ...]
           |----- Block 0 -----|---- Block 1 ----|

Program 0: Processes elements [0, BLOCK_SIZE)
Program 1: Processes elements [BLOCK_SIZE, 2*BLOCK_SIZE)
...
```

## Performance Analysis

Vector addition is **memory-bound** (not compute-bound):
- Each element requires: 2 loads + 1 store = 12 bytes (fp32)
- Only 1 FLOP per element
- Arithmetic intensity = 1/12 = 0.083 FLOP/byte

Peak performance is limited by memory bandwidth, not compute.

## Usage

```python
from vector_add import vector_add_triton

a = torch.randn(1000000, device='cuda')
b = torch.randn(1000000, device='cuda')
c = vector_add_triton(a, b)
```

## Run Benchmarks

```bash
python vector_add.py
```
