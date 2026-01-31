"""
Triton GPU Kernels
==================

Custom GPU kernels implemented in Triton for learning and performance optimization.

Kernels:
    01_vector_add: Element-wise vector addition
    02_matmul: High-performance matrix multiplication (GEMM)
    03_softmax: Fused numerically-stable softmax
    04_layernorm: Fused Layer Normalization and RMSNorm
    05_flash_attention: Memory-efficient attention (FlashAttention)

Usage:
    from triton_kernels.vector_add import vector_add_triton
    from triton_kernels.matmul import matmul_triton
    from triton_kernels.softmax import softmax_triton
    from triton_kernels.layernorm import layernorm_triton
    from triton_kernels.flash_attention import flash_attention_triton
"""

__version__ = "0.1.0"
__author__ = "Tharun Jagarlamudi"
