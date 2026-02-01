"""
Pytest configuration and fixtures for triton-kernels tests.
"""

import pytest
import torch


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "cuda: marks tests as requiring CUDA (deselect with '-m \"not cuda\"')"
    )


@pytest.fixture
def device():
    """Return CUDA device if available."""
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("CUDA not available")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


@pytest.fixture
def dtype_fp16():
    """FP16 dtype."""
    return torch.float16


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
