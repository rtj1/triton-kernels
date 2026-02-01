#!/usr/bin/env python3
"""
Triton Kernels Test Suite for Google Colab
==========================================

Run this in Colab with a GPU runtime:
1. Runtime -> Change runtime type -> GPU
2. Upload the kernel files or clone repo
3. Run this script

Usage:
    !git clone https://github.com/rtj1/triton-kernels.git
    %cd triton-kernels
    !pip install triton
    !python test_all_kernels_colab.py
"""

import torch
import torch.nn.functional as F
import sys
import time
from typing import Tuple, List

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Please use a GPU runtime.")
    sys.exit(1)

print("="*70)
print("TRITON KERNELS TEST SUITE")
print("="*70)
print(f"Device: {torch.cuda.get_device_name()}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")

# Import kernels
sys.path.insert(0, '.')

results = []

def run_test(name: str, test_fn) -> bool:
    """Run a test and record result."""
    try:
        test_fn()
        print(f"[PASS] {name}")
        results.append((name, True, None))
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        results.append((name, False, str(e)))
        return False

def run_benchmark(name: str, fn, warmup: int = 10, iters: int = 100) -> float:
    """Benchmark a function and return avg time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()

    avg_ms = (time.perf_counter() - start) / iters * 1000
    return avg_ms


# =============================================================================
# Test 1: Vector Add
# =============================================================================
print("\n" + "="*70)
print("1. VECTOR ADDITION KERNEL")
print("="*70)

try:
    from vector_add.vector_add import vector_add_triton, vector_add_torch

    def test_vector_add_correctness():
        for size in [1024, 10000, 100000, 1000000]:
            a = torch.randn(size, device='cuda')
            b = torch.randn(size, device='cuda')

            result = vector_add_triton(a, b)
            expected = vector_add_torch(a, b)

            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), \
                f"Mismatch at size {size}"

    def test_vector_add_fp16():
        a = torch.randn(100000, device='cuda', dtype=torch.float16)
        b = torch.randn(100000, device='cuda', dtype=torch.float16)

        result = vector_add_triton(a, b)
        expected = a + b

        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_vector_add_non_divisible():
        # Size not divisible by block size
        a = torch.randn(1023, device='cuda')
        b = torch.randn(1023, device='cuda')

        result = vector_add_triton(a, b)
        expected = a + b

        assert torch.allclose(result, expected)

    run_test("Vector Add - Correctness (multiple sizes)", test_vector_add_correctness)
    run_test("Vector Add - FP16 support", test_vector_add_fp16)
    run_test("Vector Add - Non-divisible size", test_vector_add_non_divisible)

    # Benchmark
    a = torch.randn(10_000_000, device='cuda')
    b = torch.randn(10_000_000, device='cuda')

    triton_ms = run_benchmark("triton", lambda: vector_add_triton(a, b))
    torch_ms = run_benchmark("torch", lambda: a + b)

    print(f"\nBenchmark (10M elements):")
    print(f"  Triton: {triton_ms:.3f} ms")
    print(f"  PyTorch: {torch_ms:.3f} ms")
    print(f"  Speedup: {torch_ms/triton_ms:.2f}x")

except ImportError as e:
    print(f"[SKIP] Vector Add - Import error: {e}")
except Exception as e:
    print(f"[ERROR] Vector Add: {e}")


# =============================================================================
# Test 2: Matrix Multiplication
# =============================================================================
print("\n" + "="*70)
print("2. MATRIX MULTIPLICATION KERNEL")
print("="*70)

try:
    from matmul.matmul import matmul_triton

    def test_matmul_correctness():
        for M, N, K in [(128, 128, 128), (256, 512, 256), (1024, 1024, 1024)]:
            a = torch.randn(M, K, device='cuda', dtype=torch.float16)
            b = torch.randn(K, N, device='cuda', dtype=torch.float16)

            result = matmul_triton(a, b)
            expected = torch.matmul(a, b)

            assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), \
                f"Mismatch at {M}x{K}x{N}"

    def test_matmul_non_square():
        a = torch.randn(256, 512, device='cuda', dtype=torch.float16)
        b = torch.randn(512, 128, device='cuda', dtype=torch.float16)

        result = matmul_triton(a, b)
        expected = torch.matmul(a, b)

        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)

    run_test("MatMul - Correctness (multiple sizes)", test_matmul_correctness)
    run_test("MatMul - Non-square matrices", test_matmul_non_square)

    # Benchmark
    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    triton_ms = run_benchmark("triton", lambda: matmul_triton(a, b))
    torch_ms = run_benchmark("torch", lambda: torch.matmul(a, b))

    flops = 2 * M * N * K
    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    print(f"\nBenchmark ({M}x{K}x{N}):")
    print(f"  Triton: {triton_ms:.3f} ms ({triton_tflops:.1f} TFLOPS)")
    print(f"  PyTorch: {torch_ms:.3f} ms ({torch_tflops:.1f} TFLOPS)")

except ImportError as e:
    print(f"[SKIP] MatMul - Import error: {e}")
except Exception as e:
    print(f"[ERROR] MatMul: {e}")


# =============================================================================
# Test 3: Softmax
# =============================================================================
print("\n" + "="*70)
print("3. SOFTMAX KERNEL")
print("="*70)

try:
    from softmax.softmax import softmax_triton

    def test_softmax_correctness():
        for size in [(128, 1024), (256, 2048), (512, 4096)]:
            x = torch.randn(size, device='cuda')

            result = softmax_triton(x)
            expected = F.softmax(x, dim=-1)

            assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
                f"Mismatch at size {size}"

    def test_softmax_numerical_stability():
        # Large values that could cause overflow
        x = torch.randn(128, 1024, device='cuda') * 100

        result = softmax_triton(x)
        expected = F.softmax(x, dim=-1)

        assert not torch.isnan(result).any(), "NaN in result"
        assert not torch.isinf(result).any(), "Inf in result"
        assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)

    def test_softmax_sums_to_one():
        x = torch.randn(64, 2048, device='cuda')
        result = softmax_triton(x)
        row_sums = result.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4)

    run_test("Softmax - Correctness (multiple sizes)", test_softmax_correctness)
    run_test("Softmax - Numerical stability", test_softmax_numerical_stability)
    run_test("Softmax - Rows sum to 1", test_softmax_sums_to_one)

    # Benchmark
    x = torch.randn(256, 4096, device='cuda')

    triton_ms = run_benchmark("triton", lambda: softmax_triton(x))
    torch_ms = run_benchmark("torch", lambda: F.softmax(x, dim=-1))

    print(f"\nBenchmark (256x4096):")
    print(f"  Triton: {triton_ms:.3f} ms")
    print(f"  PyTorch: {torch_ms:.3f} ms")

except ImportError as e:
    print(f"[SKIP] Softmax - Import error: {e}")
except Exception as e:
    print(f"[ERROR] Softmax: {e}")


# =============================================================================
# Test 4: LayerNorm
# =============================================================================
print("\n" + "="*70)
print("4. LAYER NORMALIZATION KERNEL")
print("="*70)

try:
    from layernorm.layernorm import layernorm_triton

    def test_layernorm_correctness():
        for size in [(128, 768), (256, 1024), (512, 2048)]:
            x = torch.randn(size, device='cuda')
            weight = torch.ones(size[-1], device='cuda')
            bias = torch.zeros(size[-1], device='cuda')

            result = layernorm_triton(x, weight, bias)
            expected = F.layer_norm(x, (size[-1],), weight, bias)

            assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
                f"Mismatch at size {size}"

    def test_layernorm_zero_mean_unit_var():
        x = torch.randn(64, 768, device='cuda')
        weight = torch.ones(768, device='cuda')
        bias = torch.zeros(768, device='cuda')

        result = layernorm_triton(x, weight, bias)

        # Check mean ~= 0, std ~= 1 along last dim
        mean = result.mean(dim=-1)
        std = result.std(dim=-1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    run_test("LayerNorm - Correctness (multiple sizes)", test_layernorm_correctness)
    run_test("LayerNorm - Zero mean, unit variance", test_layernorm_zero_mean_unit_var)

    # Benchmark
    x = torch.randn(256, 1024, device='cuda')
    weight = torch.ones(1024, device='cuda')
    bias = torch.zeros(1024, device='cuda')

    triton_ms = run_benchmark("triton", lambda: layernorm_triton(x, weight, bias))
    torch_ms = run_benchmark("torch", lambda: F.layer_norm(x, (1024,), weight, bias))

    print(f"\nBenchmark (256x1024):")
    print(f"  Triton: {triton_ms:.3f} ms")
    print(f"  PyTorch: {torch_ms:.3f} ms")

except ImportError as e:
    print(f"[SKIP] LayerNorm - Import error: {e}")
except Exception as e:
    print(f"[ERROR] LayerNorm: {e}")


# =============================================================================
# Test 5: FlashAttention
# =============================================================================
print("\n" + "="*70)
print("5. FLASH ATTENTION KERNEL")
print("="*70)

try:
    from flash_attention.flash_attention import flash_attention_triton, standard_attention

    def test_flash_attention_correctness():
        for seq_len in [64, 128, 256, 512]:
            batch, heads, head_dim = 2, 4, 64

            q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            result = flash_attention_triton(q, k, v, causal=False)
            expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)

            assert torch.allclose(result, expected, rtol=0.05, atol=0.05), \
                f"Mismatch at seq_len {seq_len}"

    def test_flash_attention_causal():
        batch, heads, seq_len, head_dim = 2, 4, 256, 64

        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        result = flash_attention_triton(q, k, v, causal=True)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert torch.allclose(result, expected, rtol=0.05, atol=0.05)

    def test_flash_attention_memory():
        """Test that FlashAttention uses less memory."""
        torch.cuda.reset_peak_memory_stats()

        batch, heads, seq_len, head_dim = 2, 8, 2048, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        _ = flash_attention_triton(q, k, v, causal=True)
        flash_mem = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()

        # Standard attention would allocate N^2 matrix
        attn_matrix_size = batch * heads * seq_len * seq_len * 2 / 1e6  # fp16

        # FlashAttention should use much less
        assert flash_mem < attn_matrix_size * 3, \
            f"FlashAttention using too much memory: {flash_mem:.1f}MB vs {attn_matrix_size:.1f}MB attn matrix"

    run_test("FlashAttention - Correctness (multiple seq_lens)", test_flash_attention_correctness)
    run_test("FlashAttention - Causal masking", test_flash_attention_causal)
    run_test("FlashAttention - Memory efficiency", test_flash_attention_memory)

    # Benchmark
    batch, heads, seq_len, head_dim = 4, 8, 2048, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    flash_ms = run_benchmark("flash", lambda: flash_attention_triton(q, k, v, causal=True))
    sdpa_ms = run_benchmark("sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    flash_tflops = flops / (flash_ms * 1e-3) / 1e12

    print(f"\nBenchmark (B={batch}, H={heads}, S={seq_len}, D={head_dim}):")
    print(f"  FlashAttention: {flash_ms:.3f} ms ({flash_tflops:.1f} TFLOPS)")
    print(f"  PyTorch SDPA: {sdpa_ms:.3f} ms")
    print(f"  Ratio: {sdpa_ms/flash_ms:.2f}x")

    # Long sequence test
    print("\nLong sequence test (8192 tokens):")
    q = torch.randn(1, 8, 8192, 64, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    flash_ms = run_benchmark("flash", lambda: flash_attention_triton(q, k, v, causal=True), warmup=5, iters=20)
    print(f"  FlashAttention: {flash_ms:.3f} ms")

except ImportError as e:
    print(f"[SKIP] FlashAttention - Import error: {e}")
except Exception as e:
    print(f"[ERROR] FlashAttention: {e}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)

print(f"\nTotal: {len(results)} tests")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed > 0:
    print("\nFailed tests:")
    for name, passed, error in results:
        if not passed:
            print(f"  - {name}: {error}")

print("\n" + "="*70)
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"SOME TESTS FAILED ({failed}/{len(results)})")
print("="*70)
