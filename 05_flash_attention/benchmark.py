#!/usr/bin/env python3
"""
Comprehensive FlashAttention Benchmarks
========================================

Compares our Triton FlashAttention implementation against:
1. PyTorch's native scaled_dot_product_attention (uses cuDNN FlashAttention)
2. Standard O(N^2) attention implementation

Metrics measured:
- Latency (ms)
- Throughput (TFLOPS)
- Memory bandwidth utilization (GB/s)
- Peak memory usage

Run: python benchmark.py
"""

import torch
import torch.nn.functional as F
import math
import json
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict
from flash_attention import flash_attention_triton, standard_attention


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    config: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int

    # Timing (ms)
    triton_ms: float
    pytorch_sdpa_ms: Optional[float]
    standard_ms: Optional[float]

    # Throughput
    triton_tflops: float
    pytorch_tflops: Optional[float]

    # Memory bandwidth (GB/s)
    triton_bandwidth_gbs: float
    pytorch_bandwidth_gbs: Optional[float]

    # Memory
    triton_peak_mem_mb: float
    standard_peak_mem_mb: Optional[float]

    # Speedup
    speedup_vs_pytorch: Optional[float]
    speedup_vs_standard: Optional[float]


def get_theoretical_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """
    Calculate theoretical FLOPs for attention.

    Attention: Q @ K^T (2*N^2*d) + softmax (5*N^2) + attn @ V (2*N^2*d)
    Simplified: ~4 * batch * heads * seq^2 * head_dim
    """
    return 4 * batch * heads * seq_len * seq_len * head_dim


def get_memory_io_bytes(batch: int, heads: int, seq_len: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """
    Calculate memory I/O for FlashAttention.

    FlashAttention reads Q, K, V once and writes O once.
    Total: 4 * batch * heads * seq_len * head_dim * dtype_bytes
    """
    return 4 * batch * heads * seq_len * head_dim * dtype_bytes


def benchmark_single_config(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = True,
    warmup: int = 25,
    iterations: int = 100,
) -> BenchmarkResult:
    """Run comprehensive benchmark for a single configuration."""

    config = f"B={batch}, H={heads}, S={seq_len}, D={head_dim}"
    dtype = torch.float16
    device = 'cuda'

    # Create inputs
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Calculate theoretical metrics
    flops = get_theoretical_flops(batch, heads, seq_len, head_dim)
    io_bytes = get_memory_io_bytes(batch, heads, seq_len, head_dim, dtype_bytes=2)

    # =========================================================================
    # Benchmark Triton FlashAttention
    # =========================================================================
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        _ = flash_attention_triton(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # Benchmark
    triton_times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = flash_attention_triton(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    triton_ms = sum(triton_times) / len(triton_times)
    triton_peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    triton_tflops = flops / (triton_ms * 1e-3) / 1e12
    triton_bandwidth = io_bytes / (triton_ms * 1e-3) / 1e9

    # =========================================================================
    # Benchmark PyTorch SDPA (native FlashAttention via cuDNN)
    # =========================================================================
    pytorch_ms = None
    pytorch_tflops = None
    pytorch_bandwidth = None
    speedup_pytorch = None

    try:
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()

        # Benchmark
        pytorch_times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            end.record()
            torch.cuda.synchronize()
            pytorch_times.append(start.elapsed_time(end))

        pytorch_ms = sum(pytorch_times) / len(pytorch_times)
        pytorch_tflops = flops / (pytorch_ms * 1e-3) / 1e12
        pytorch_bandwidth = io_bytes / (pytorch_ms * 1e-3) / 1e9
        speedup_pytorch = pytorch_ms / triton_ms

    except Exception as e:
        print(f"  PyTorch SDPA failed: {e}")

    # =========================================================================
    # Benchmark Standard Attention (if memory allows)
    # =========================================================================
    standard_ms = None
    standard_peak_mem = None
    speedup_standard = None

    if seq_len <= 4096:  # Avoid OOM
        try:
            torch.cuda.reset_peak_memory_stats()

            # Warmup
            for _ in range(warmup):
                _ = standard_attention(q, k, v, causal=causal)
            torch.cuda.synchronize()

            standard_peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Benchmark
            standard_times = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = standard_attention(q, k, v, causal=causal)
                end.record()
                torch.cuda.synchronize()
                standard_times.append(start.elapsed_time(end))

            standard_ms = sum(standard_times) / len(standard_times)
            speedup_standard = standard_ms / triton_ms

        except Exception as e:
            print(f"  Standard attention failed: {e}")

    return BenchmarkResult(
        config=config,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        triton_ms=triton_ms,
        pytorch_sdpa_ms=pytorch_ms,
        standard_ms=standard_ms,
        triton_tflops=triton_tflops,
        pytorch_tflops=pytorch_tflops,
        triton_bandwidth_gbs=triton_bandwidth,
        pytorch_bandwidth_gbs=pytorch_bandwidth,
        triton_peak_mem_mb=triton_peak_mem,
        standard_peak_mem_mb=standard_peak_mem,
        speedup_vs_pytorch=speedup_pytorch,
        speedup_vs_standard=speedup_standard,
    )


def verify_correctness():
    """Verify our implementation matches PyTorch SDPA."""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)

    test_configs = [
        (2, 4, 256, 64, False),
        (2, 4, 256, 64, True),
        (4, 8, 512, 64, True),
        (2, 8, 1024, 128, True),
    ]

    all_passed = True
    for batch, heads, seq, dim, causal in test_configs:
        q = torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        triton_out = flash_attention_triton(q, k, v, causal=causal)
        pytorch_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        max_diff = (triton_out - pytorch_out).abs().max().item()
        mean_diff = (triton_out - pytorch_out).abs().mean().item()

        passed = max_diff < 0.05  # Allow small numerical differences
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"[{status}] B={batch}, H={heads}, S={seq}, D={dim}, causal={causal}")
        print(f"       Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    return all_passed


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 120)

    # Header
    print(f"\n{'Configuration':<25} | {'Triton':>10} | {'PyTorch':>10} | {'Standard':>10} | "
          f"{'vs PT':>8} | {'vs Std':>8} | {'TFLOPS':>8} | {'BW (GB/s)':>10}")
    print("-" * 120)

    for r in results:
        pytorch = f"{r.pytorch_sdpa_ms:.3f}" if r.pytorch_sdpa_ms else "N/A"
        standard = f"{r.standard_ms:.3f}" if r.standard_ms else "OOM"
        vs_pt = f"{r.speedup_vs_pytorch:.2f}x" if r.speedup_vs_pytorch else "N/A"
        vs_std = f"{r.speedup_vs_standard:.2f}x" if r.speedup_vs_standard else "N/A"

        print(f"{r.config:<25} | {r.triton_ms:>10.3f} | {pytorch:>10} | {standard:>10} | "
              f"{vs_pt:>8} | {vs_std:>8} | {r.triton_tflops:>7.1f} | {r.triton_bandwidth_gbs:>10.1f}")

    # Memory comparison
    print("\n" + "=" * 80)
    print("MEMORY USAGE COMPARISON")
    print("=" * 80)
    print(f"\n{'Configuration':<25} | {'Triton Peak (MB)':>18} | {'Standard Peak (MB)':>18} | {'Savings':>10}")
    print("-" * 80)

    for r in results:
        if r.standard_peak_mem_mb:
            savings = (1 - r.triton_peak_mem_mb / r.standard_peak_mem_mb) * 100
            std_mem = f"{r.standard_peak_mem_mb:.1f}"
            save_str = f"{savings:.1f}%"
        else:
            std_mem = "OOM"
            save_str = "N/A"

        print(f"{r.config:<25} | {r.triton_peak_mem_mb:>18.1f} | {std_mem:>18} | {save_str:>10}")


def print_analysis(results: List[BenchmarkResult]):
    """Print detailed analysis."""
    print("\n" + "=" * 80)
    print("ANALYSIS & INSIGHTS")
    print("=" * 80)

    # Get GPU info
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    memory_bw = {
        'T4': 320,
        'V100': 900,
        'A100': 2039,
        'A10': 600,
        'RTX 3090': 936,
        'RTX 4090': 1008,
        'H100': 3350,
    }

    # Estimate peak bandwidth for this GPU
    peak_bw = None
    for name, bw in memory_bw.items():
        if name in gpu_name:
            peak_bw = bw
            break

    if peak_bw:
        print(f"\nGPU: {gpu_name}")
        print(f"Theoretical Peak Memory Bandwidth: {peak_bw} GB/s")
        print("\nBandwidth Utilization:")
        for r in results:
            util = (r.triton_bandwidth_gbs / peak_bw) * 100
            print(f"  {r.config}: {r.triton_bandwidth_gbs:.1f} GB/s ({util:.1f}% of peak)")

    # Scaling analysis
    print("\n" + "-" * 40)
    print("Sequence Length Scaling:")
    print("-" * 40)

    # Find results with same batch/heads/dim but different seq_len
    base_results = [r for r in results if r.batch == 4 and r.heads == 8 and r.head_dim == 64]
    if len(base_results) >= 2:
        sorted_results = sorted(base_results, key=lambda x: x.seq_len)
        base = sorted_results[0]
        for r in sorted_results[1:]:
            seq_ratio = r.seq_len / base.seq_len
            time_ratio = r.triton_ms / base.triton_ms
            # FlashAttention should scale ~O(N^2) for FLOPs but O(N) for memory
            expected_flop_ratio = seq_ratio ** 2
            print(f"  S={base.seq_len} -> S={r.seq_len}: "
                  f"{time_ratio:.2f}x time (FLOPs scale: {expected_flop_ratio:.1f}x)")

    print("\n" + "-" * 40)
    print("Key Takeaways:")
    print("-" * 40)

    # Calculate average vs PyTorch
    pytorch_comparisons = [r.speedup_vs_pytorch for r in results if r.speedup_vs_pytorch]
    if pytorch_comparisons:
        avg_vs_pytorch = sum(pytorch_comparisons) / len(pytorch_comparisons)
        print(f"  - Average performance vs PyTorch SDPA: {avg_vs_pytorch:.2f}x")
        if avg_vs_pytorch >= 0.9:
            print("    -> Within 10% of highly optimized cuDNN implementation!")
        elif avg_vs_pytorch >= 0.7:
            print("    -> Reasonable performance, room for optimization")
        else:
            print("    -> Significant optimization opportunity")

    # Memory savings
    mem_savings = [r for r in results if r.standard_peak_mem_mb]
    if mem_savings:
        print(f"  - FlashAttention enables processing sequences that would OOM with standard attention")
        print(f"  - Memory scales O(N) vs O(N^2)")


def save_results_json(results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
    """Save results to JSON for later analysis."""
    data = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "results": [
            {
                "config": r.config,
                "batch": r.batch,
                "heads": r.heads,
                "seq_len": r.seq_len,
                "head_dim": r.head_dim,
                "triton_ms": r.triton_ms,
                "pytorch_sdpa_ms": r.pytorch_sdpa_ms,
                "standard_ms": r.standard_ms,
                "triton_tflops": r.triton_tflops,
                "pytorch_tflops": r.pytorch_tflops,
                "triton_bandwidth_gbs": r.triton_bandwidth_gbs,
                "speedup_vs_pytorch": r.speedup_vs_pytorch,
                "speedup_vs_standard": r.speedup_vs_standard,
                "triton_peak_mem_mb": r.triton_peak_mem_mb,
                "standard_peak_mem_mb": r.standard_peak_mem_mb,
            }
            for r in results
        ]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print("=" * 80)
    print("FlashAttention Triton Benchmark Suite")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Verify correctness first
    if not verify_correctness():
        print("\nWARNING: Correctness verification failed!")
        print("Proceeding with benchmarks anyway...")

    # Benchmark configurations
    configs = [
        # (batch, heads, seq_len, head_dim)
        # Standard LLM-like configs
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
        (4, 8, 4096, 64),

        # Long context (where FlashAttention shines)
        (2, 8, 8192, 64),
        (1, 8, 16384, 64),

        # Different head dims
        (4, 8, 2048, 32),
        (4, 8, 2048, 128),

        # Multi-head attention variants
        (8, 32, 1024, 64),  # More heads
        (2, 64, 512, 64),   # Many heads (GQA-like)
    ]

    print("\n" + "=" * 80)
    print("Running benchmarks...")
    print("=" * 80)

    results = []
    for batch, heads, seq, dim in configs:
        print(f"\nBenchmarking: B={batch}, H={heads}, S={seq}, D={dim}")
        try:
            result = benchmark_single_config(batch, heads, seq, dim)
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")

    # Print results
    print_results_table(results)
    print_analysis(results)

    # Save to JSON
    save_results_json(results)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
