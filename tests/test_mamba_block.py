"""Tests for mamba_video.mamba_block module."""

import pytest
import torch

from mamba_video.mamba_block import MambaBlock


class TestMambaBlockShape:
    """Verify MambaBlock produces correct output shapes."""

    def test_basic_forward(self) -> None:
        """Output shape should match input shape."""
        block = MambaBlock(d_model=64, d_state=8, d_conv=4, expand=2)
        x = torch.randn(2, 16, 64)  # (batch=2, seq=16, d_model=64)
        with torch.no_grad():
            y = block(x)
        assert y.shape == x.shape

    def test_single_timestep(self) -> None:
        """Should handle sequence length of 1."""
        block = MambaBlock(d_model=32, d_state=4)
        x = torch.randn(1, 1, 32)
        with torch.no_grad():
            y = block(x)
        assert y.shape == (1, 1, 32)

    def test_long_sequence(self) -> None:
        """Should handle longer sequences."""
        block = MambaBlock(d_model=32, d_state=8)
        x = torch.randn(1, 128, 32)
        with torch.no_grad():
            y = block(x)
        assert y.shape == (1, 128, 32)

    def test_different_d_model(self) -> None:
        """Should work with various d_model sizes."""
        for d_model in [32, 64, 128, 256]:
            block = MambaBlock(d_model=d_model, d_state=8)
            x = torch.randn(1, 8, d_model)
            with torch.no_grad():
                y = block(x)
            assert y.shape == (1, 8, d_model), f"Failed for d_model={d_model}"


class TestMambaBlockGradient:
    """Verify gradients flow through the MambaBlock."""

    def test_backward_pass(self) -> None:
        """Gradients should flow through the block."""
        block = MambaBlock(d_model=32, d_state=4, d_conv=4, expand=2)
        block.train()
        x = torch.randn(1, 8, 32, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameters_receive_gradients(self) -> None:
        """All parameters should receive gradients."""
        block = MambaBlock(d_model=32, d_state=4)
        block.train()
        x = torch.randn(1, 8, 32)
        y = block(x)
        loss = y.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMambaBlockNumerical:
    """Verify numerical stability."""

    def test_no_nan_output(self) -> None:
        """Output should not contain NaN values."""
        block = MambaBlock(d_model=64, d_state=8)
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            y = block(x)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_no_inf_output(self) -> None:
        """Output should not contain Inf values."""
        block = MambaBlock(d_model=64, d_state=8)
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            y = block(x)
        assert not torch.isinf(y).any(), "Output contains Inf"

    def test_float16_stability(self) -> None:
        """Should produce valid output in float16."""
        block = MambaBlock(d_model=64, d_state=8).half()
        x = torch.randn(1, 8, 64, dtype=torch.float16)
        with torch.no_grad():
            y = block(x)
        assert not torch.isnan(y).any(), "float16 output contains NaN"
        assert not torch.isinf(y).any(), "float16 output contains Inf"

    def test_output_bounded(self) -> None:
        """Output magnitude should be reasonable (not exploding)."""
        block = MambaBlock(d_model=64, d_state=8)
        x = torch.randn(1, 16, 64)
        with torch.no_grad():
            y = block(x)
        # Output should not be orders of magnitude larger than input
        assert y.abs().max() < 1000, f"Output too large: {y.abs().max()}"


class TestMambaBlockResidual:
    """Test residual connection behavior."""

    def test_with_explicit_residual(self) -> None:
        """Should add explicit residual when provided."""
        block = MambaBlock(d_model=32, d_state=4)
        x = torch.randn(1, 8, 32)
        residual = torch.randn(1, 8, 32)
        with torch.no_grad():
            y = block(x, residual=residual)
        assert y.shape == x.shape

    def test_residual_connection_active(self) -> None:
        """Default residual should make output close to input for identity init."""
        block = MambaBlock(d_model=32, d_state=4)
        # Zero out output projection to test pure residual
        with torch.no_grad():
            block.out_proj.weight.zero_()
            if block.out_proj.bias is not None:
                block.out_proj.bias.zero_()

        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            y = block(x)

        # With zeroed output, y should equal x (pure residual)
        assert torch.allclose(y, x, atol=1e-5), "Residual not working correctly"
