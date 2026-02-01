"""
Tests for vector addition kernel.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVectorAdd:
    """Tests for vector addition kernel."""

    @pytest.fixture(autouse=True)
    def setup_imports(self):
        """Import kernel functions."""
        # Import here to allow skipping on non-CUDA systems
        from importlib import import_module
        mod = import_module("01_vector_add.vector_add")
        self.vector_add_triton = mod.vector_add_triton
        self.vector_add_torch = mod.vector_add_torch

    def test_correctness_small(self, device, dtype):
        """Test correctness on small vectors."""
        a = torch.randn(1024, device=device, dtype=dtype)
        b = torch.randn(1024, device=device, dtype=dtype)

        triton_result = self.vector_add_triton(a, b)
        torch_result = self.vector_add_torch(a, b)

        assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)

    def test_correctness_large(self, device, dtype):
        """Test correctness on large vectors."""
        a = torch.randn(10_000_000, device=device, dtype=dtype)
        b = torch.randn(10_000_000, device=device, dtype=dtype)

        triton_result = self.vector_add_triton(a, b)
        torch_result = self.vector_add_torch(a, b)

        assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)

    def test_correctness_fp16(self, device):
        """Test correctness with FP16."""
        a = torch.randn(100_000, device=device, dtype=torch.float16)
        b = torch.randn(100_000, device=device, dtype=torch.float16)

        triton_result = self.vector_add_triton(a, b)
        torch_result = self.vector_add_torch(a, b)

        assert torch.allclose(triton_result, torch_result, rtol=1e-3, atol=1e-3)

    def test_non_divisible_size(self, device, dtype):
        """Test with size not divisible by block size."""
        # 1023 is not divisible by typical block sizes
        a = torch.randn(1023, device=device, dtype=dtype)
        b = torch.randn(1023, device=device, dtype=dtype)

        triton_result = self.vector_add_triton(a, b)
        torch_result = self.vector_add_torch(a, b)

        assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)

    def test_different_block_sizes(self, device, dtype):
        """Test with different block sizes."""
        a = torch.randn(10000, device=device, dtype=dtype)
        b = torch.randn(10000, device=device, dtype=dtype)
        expected = self.vector_add_torch(a, b)

        for block_size in [256, 512, 1024, 2048]:
            result = self.vector_add_triton(a, b, block_size=block_size)
            assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_requires_cuda(self):
        """Test that CPU tensors raise an error."""
        a = torch.randn(1024)  # CPU tensor
        b = torch.randn(1024)

        with pytest.raises(AssertionError):
            self.vector_add_triton(a, b)

    def test_shape_mismatch(self, device, dtype):
        """Test that mismatched shapes raise an error."""
        a = torch.randn(1024, device=device, dtype=dtype)
        b = torch.randn(512, device=device, dtype=dtype)

        with pytest.raises(AssertionError):
            self.vector_add_triton(a, b)
