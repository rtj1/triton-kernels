# Triton GPU Kernels

**Custom GPU kernels implemented in Triton for learning and performance optimization.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/triton-2.1+-green.svg)](https://github.com/triton-lang/triton)

---

## Overview

This repository contains implementations of fundamental GPU kernels using [Triton](https://github.com/triton-lang/triton), OpenAI's Python-like language for writing efficient GPU code. Each kernel demonstrates key concepts in GPU programming and achieves performance competitive with highly-optimized libraries like cuBLAS.

## Why Triton?

- **Python-like syntax**: Write GPU kernels without learning CUDA
- **Automatic optimization**: Triton handles memory coalescing, shared memory, etc.
- **Expert-level performance**: Match or exceed hand-written CUDA in many cases
- **Rapid iteration**: Test optimization ideas quickly

## Kernels Implemented

| # | Kernel | Description | Key Concepts |
|---|--------|-------------|--------------|
| 01 | [Vector Add](01_vector_add/) | Element-wise addition | Basic kernel structure, grid launch |
| 02 | [Matrix Multiply](02_matmul/) | GEMM (C = A @ B) | Tiling, autotuning, shared memory |
| 03 | [Softmax](03_softmax/) | Fused softmax | Online algorithm, kernel fusion |
| 04 | [LayerNorm](04_layernorm/) | Layer normalization | Welford's algorithm, RMSNorm |
| 05 | [FlashAttention](05_flash_attention/) | Memory-efficient attention | O(N) memory, tiled computation |

## Performance Highlights

### Matrix Multiplication
- **90%+ efficiency** vs cuBLAS for large matrices
- Autotuned block sizes for different shapes

### FlashAttention
- **O(N) memory** vs O(N²) for standard attention
- Enables 100K+ token sequences
- 2-4x faster than naive implementation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/rtj1/triton-kernels.git
cd triton-kernels

# Install dependencies
pip install torch triton numpy matplotlib tabulate

# Run a specific kernel
python 01_vector_add/vector_add.py
python 02_matmul/matmul.py
python 05_flash_attention/flash_attention.py

# Run all benchmarks
python benchmarks/run_all.py
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Triton 2.1+
- NVIDIA GPU (Ampere or newer recommended)
- CUDA 12.0+

## Project Structure

```
triton-kernels/
├── 01_vector_add/
│   ├── vector_add.py      # Kernel + benchmarks
│   └── README.md          # Concepts explained
├── 02_matmul/
│   ├── matmul.py          # Autotuned GEMM
│   └── README.md
├── 03_softmax/
│   ├── softmax.py         # Fused softmax with online algorithm
│   └── README.md
├── 04_layernorm/
│   ├── layernorm.py       # LayerNorm + RMSNorm
│   └── README.md
├── 05_flash_attention/
│   ├── flash_attention.py # Memory-efficient attention
│   └── README.md
├── benchmarks/
│   └── run_all.py         # Comprehensive benchmark suite
├── pyproject.toml
└── README.md
```

## Learning Path

1. **Start with Vector Add**: Understand basic kernel structure
2. **Move to Matrix Multiply**: Learn tiling and autotuning
3. **Study Softmax**: Master online algorithms and fusion
4. **Explore LayerNorm**: See Welford's algorithm in action
5. **Finish with FlashAttention**: Combine all concepts

Each kernel builds on concepts from previous ones.

## Key Concepts Demonstrated

### 1. Memory Hierarchy
- Global memory (HBM): High capacity, high latency
- Shared memory (SRAM): Low capacity, low latency
- Registers: Fastest, most limited

### 2. Kernel Fusion
Combine multiple operations into single kernel to reduce memory traffic:
```
Naive: Read → Op1 → Write → Read → Op2 → Write  (4 memory ops)
Fused: Read → Op1 → Op2 → Write                  (2 memory ops)
```

### 3. Tiling
Process data in blocks that fit in fast memory:
```
for tile in tiles:
    load_to_shared(tile)
    compute(tile)
    store_results(tile)
```

### 4. Online Algorithms
Compute results incrementally without materializing full intermediate:
- Online softmax: Process in chunks, maintain running max/sum
- Used in FlashAttention for O(N) memory attention

## Benchmarking

Each kernel includes:
- Correctness verification against PyTorch
- Performance benchmarks with timing
- Bandwidth/TFLOPS calculations
- Comparison against optimized baselines

Run benchmarks:
```bash
# Individual kernel
python 02_matmul/matmul.py

# All kernels
python benchmarks/run_all.py
```

## References

### Papers
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

### Tutorials
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures)

## Contributing

Contributions welcome! Ideas for new kernels:
- [ ] Fused attention + LayerNorm
- [ ] Flash decoding (KV-cache optimized)
- [ ] Grouped query attention
- [ ] Quantized matmul (INT8/INT4)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Author**: Tharun Jagarlamudi

**Related Projects**:
- [Anthropic Performance Takehome](https://github.com/rtj1/original_performance_takehome) - VLIW kernel optimization (1,371 cycles, 96.1% efficiency)
- [DDP LoRA Trainer](https://github.com/rtj1/ddp-lora-trainer-) - Distributed LLM training infrastructure
