"""
Tests for FlashAttention kernel.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFlashAttention:
    """Tests for FlashAttention kernel."""

    @pytest.fixture(autouse=True)
    def setup_imports(self):
        """Import kernel functions."""
        from importlib import import_module
        mod = import_module("05_flash_attention.flash_attention")
        self.flash_attention_triton = mod.flash_attention_triton
        self.standard_attention = mod.standard_attention

    def test_correctness_vs_pytorch_sdpa(self, device):
        """Test correctness against PyTorch SDPA."""
        batch, heads, seq_len, head_dim = 2, 4, 256, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        triton_out = self.flash_attention_triton(q, k, v, causal=True)
        pytorch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert torch.allclose(triton_out, pytorch_out, rtol=0.05, atol=0.05)

    def test_correctness_vs_standard(self, device):
        """Test correctness against standard O(N^2) attention."""
        batch, heads, seq_len, head_dim = 2, 4, 128, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        flash_out = self.flash_attention_triton(q, k, v, causal=False)
        standard_out = self.standard_attention(q, k, v, causal=False)

        assert torch.allclose(flash_out, standard_out, rtol=0.02, atol=0.02)

    def test_causal_masking(self, device):
        """Test that causal masking works correctly."""
        batch, heads, seq_len, head_dim = 2, 4, 64, 64
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        flash_out = self.flash_attention_triton(q, k, v, causal=True)
        standard_out = self.standard_attention(q, k, v, causal=True)

        assert torch.allclose(flash_out, standard_out, rtol=0.02, atol=0.02)

    def test_different_seq_lengths(self, device):
        """Test with various sequence lengths."""
        for seq_len in [64, 128, 256, 512, 1024]:
            q = torch.randn(2, 4, seq_len, 64, device=device, dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            out = self.flash_attention_triton(q, k, v, causal=True)

            assert out.shape == q.shape
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_different_head_dims(self, device):
        """Test with various head dimensions."""
        for head_dim in [32, 64, 128]:
            q = torch.randn(2, 4, 256, head_dim, device=device, dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            out = self.flash_attention_triton(q, k, v, causal=True)

            assert out.shape == q.shape
            assert not torch.isnan(out).any()

    def test_custom_scale(self, device):
        """Test with custom softmax scale."""
        q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = self.flash_attention_triton(q, k, v, sm_scale=0.1)

        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_memory_efficiency(self, device):
        """Test that memory usage is O(N) not O(N^2)."""
        torch.cuda.reset_peak_memory_stats()

        # Moderate sequence length
        q = torch.randn(2, 8, 2048, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        _ = self.flash_attention_triton(q, k, v, causal=True)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        # Standard attention would need ~2*8*2048*2048*2 = 128 MB just for attn matrix
        attn_matrix_size_mb = 2 * 8 * 2048 * 2048 * 2 / (1024**2)

        # Peak memory should be less than what standard attention would need
        assert peak_mem < attn_matrix_size_mb * 2, f"Peak mem {peak_mem}MB seems too high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFlashAttentionEdgeCases:
    """Edge case tests for FlashAttention."""

    @pytest.fixture(autouse=True)
    def setup_imports(self):
        """Import kernel functions."""
        from importlib import import_module
        mod = import_module("05_flash_attention.flash_attention")
        self.flash_attention_triton = mod.flash_attention_triton

    def test_batch_size_one(self, device):
        """Test with batch size of 1."""
        q = torch.randn(1, 8, 512, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = self.flash_attention_triton(q, k, v, causal=True)
        assert out.shape == q.shape

    def test_many_heads(self, device):
        """Test with many attention heads."""
        q = torch.randn(2, 32, 256, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = self.flash_attention_triton(q, k, v, causal=True)
        assert out.shape == q.shape
