"""
Comprehensive Benchmark Suite for Triton Kernels
================================================

Run all benchmarks and generate a summary report.
"""

import torch
import sys
import os
from datetime import datetime
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_vector_add_benchmarks():
    """Run vector addition benchmarks."""
    print_header("Vector Addition Benchmarks")

    from vector_add import benchmark_vector_add, verify_correctness

    # Verify correctness
    print("\nVerifying correctness...")
    assert verify_correctness(size=100000), "Correctness check failed!"

    # Run benchmarks
    sizes = [1024, 100_000, 1_000_000, 10_000_000, 100_000_000]
    results = []

    for size in sizes:
        try:
            r = benchmark_vector_add(size=size)
            results.append([
                f"{size:,}",
                f"{r['triton_ms']:.4f}",
                f"{r['torch_ms']:.4f}",
                f"{r['speedup']:.2f}x",
                f"{r['triton_bandwidth_gbps']:.1f}",
            ])
        except Exception as e:
            results.append([f"{size:,}", "Error", str(e), "-", "-"])

    print("\n" + tabulate(results,
        headers=["Size", "Triton (ms)", "PyTorch (ms)", "Speedup", "Bandwidth (GB/s)"],
        tablefmt="grid"))


def run_matmul_benchmarks():
    """Run matrix multiplication benchmarks."""
    print_header("Matrix Multiplication Benchmarks")

    from matmul import benchmark_matmul, verify_correctness

    # Verify correctness
    print("\nVerifying correctness...")
    assert verify_correctness(M=512, N=512, K=512), "Correctness check failed!"

    # Run benchmarks
    sizes = [512, 1024, 2048, 4096]
    results = []

    for size in sizes:
        try:
            r = benchmark_matmul(M=size, N=size, K=size)
            results.append([
                size,
                f"{r['triton_ms']:.3f}",
                f"{r['torch_ms']:.3f}",
                f"{r['triton_tflops']:.2f}",
                f"{r['torch_tflops']:.2f}",
                f"{r['efficiency']:.1f}%",
            ])
        except Exception as e:
            results.append([size, "Error", str(e), "-", "-", "-"])

    print("\n" + tabulate(results,
        headers=["Size", "Triton (ms)", "cuBLAS (ms)", "Triton TFLOPS", "cuBLAS TFLOPS", "Efficiency"],
        tablefmt="grid"))


def run_softmax_benchmarks():
    """Run softmax benchmarks."""
    print_header("Fused Softmax Benchmarks")

    from softmax import benchmark_softmax, verify_correctness

    # Verify correctness
    print("\nVerifying correctness...")
    assert verify_correctness(batch_size=32, seq_len=1024), "Correctness check failed!"

    # Run benchmarks
    configs = [(32, 512), (32, 1024), (32, 2048), (32, 4096), (8, 8192)]
    results = []

    for batch, seq in configs:
        try:
            r = benchmark_softmax(batch_size=batch, seq_len=seq)
            results.append([
                f"({batch}, {seq})",
                f"{r['triton_ms']:.4f}",
                f"{r['torch_ms']:.4f}",
                f"{r['speedup']:.2f}x",
                f"{r['triton_bandwidth_gbps']:.1f}",
            ])
        except Exception as e:
            results.append([f"({batch}, {seq})", "Error", str(e), "-", "-"])

    print("\n" + tabulate(results,
        headers=["(Batch, SeqLen)", "Triton (ms)", "PyTorch (ms)", "Speedup", "Bandwidth (GB/s)"],
        tablefmt="grid"))


def run_layernorm_benchmarks():
    """Run LayerNorm benchmarks."""
    print_header("Fused LayerNorm Benchmarks")

    from layernorm import benchmark_layernorm, verify_correctness

    # Verify correctness
    print("\nVerifying correctness...")
    assert verify_correctness(batch_size=4, seq_len=128, hidden_size=768), "Correctness check failed!"

    # Run benchmarks
    configs = [(32, 512, 768), (16, 1024, 768), (8, 2048, 1024), (4, 2048, 2048)]
    results = []

    for batch, seq, hidden in configs:
        try:
            r = benchmark_layernorm(batch_size=batch, seq_len=seq, hidden_size=hidden)
            results.append([
                f"({batch}, {seq}, {hidden})",
                f"{r['triton_ms']:.4f}",
                f"{r['torch_ms']:.4f}",
                f"{r['speedup']:.2f}x",
                f"{r['triton_bandwidth_gbps']:.1f}",
            ])
        except Exception as e:
            results.append([f"({batch}, {seq}, {hidden})", "Error", str(e), "-", "-"])

    print("\n" + tabulate(results,
        headers=["(Batch, Seq, Hidden)", "Triton (ms)", "PyTorch (ms)", "Speedup", "Bandwidth (GB/s)"],
        tablefmt="grid"))


def run_flash_attention_benchmarks():
    """Run FlashAttention benchmarks."""
    print_header("FlashAttention Benchmarks")

    from flash_attention import benchmark_attention, verify_correctness

    # Verify correctness
    print("\nVerifying correctness...")
    assert verify_correctness(batch_size=2, n_heads=4, seq_len=256, head_dim=64), "Correctness check failed!"

    # Run benchmarks
    configs = [
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
        (2, 8, 4096, 64),
        (1, 8, 8192, 64),
    ]
    results = []

    for batch, heads, seq, dim in configs:
        try:
            r = benchmark_attention(batch_size=batch, n_heads=heads, seq_len=seq, head_dim=dim, causal=True)
            std_ms = f"{r['standard_ms']:.3f}" if r['standard_ms'] else "OOM"
            speedup = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"

            results.append([
                f"({batch}, {heads}, {seq}, {dim})",
                f"{r['flash_ms']:.3f}",
                std_ms,
                speedup,
                f"{r['flash_tflops']:.1f}",
                f"{r['attn_memory_mb']:.1f}",
            ])
        except Exception as e:
            results.append([f"({batch}, {heads}, {seq}, {dim})", "Error", str(e), "-", "-", "-"])

    print("\n" + tabulate(results,
        headers=["(B, H, S, D)", "Flash (ms)", "Standard (ms)", "Speedup", "TFLOPS", "Attn Mem (MB)"],
        tablefmt="grid"))


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("  TRITON KERNELS - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    # Print system info
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all benchmarks
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '01_vector_add'))
        run_vector_add_benchmarks()
    except Exception as e:
        print(f"Vector add benchmark failed: {e}")

    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_matmul'))
        run_matmul_benchmarks()
    except Exception as e:
        print(f"Matmul benchmark failed: {e}")

    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_softmax'))
        run_softmax_benchmarks()
    except Exception as e:
        print(f"Softmax benchmark failed: {e}")

    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_layernorm'))
        run_layernorm_benchmarks()
    except Exception as e:
        print(f"LayerNorm benchmark failed: {e}")

    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '05_flash_attention'))
        run_flash_attention_benchmarks()
    except Exception as e:
        print(f"FlashAttention benchmark failed: {e}")

    print("\n" + "=" * 80)
    print("  BENCHMARK SUITE COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
