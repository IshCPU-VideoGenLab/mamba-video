"""Tests for mamba_video.surgery module."""

import pytest
import torch
import torch.nn as nn

from mamba_video.config import MambaConfig, SurgeryConfig
from mamba_video.mamba_block import MambaBlock
from mamba_video.surgery import (
    find_attention_modules,
    replace_module,
    create_mamba_replacement,
    select_blocks_to_replace,
)


class DummyAttention(nn.Module):
    """Minimal attention module for testing."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.v_proj(x))


class DummyTransformerBlock(nn.Module):
    """Minimal transformer block with attention + FFN."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.attention = DummyAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DummyModel(nn.Module):
    """Minimal model with multiple transformer blocks."""

    def __init__(self, num_blocks: int = 4, d_model: int = 64) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DummyTransformerBlock(d_model) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TestFindAttentionModules:
    """Tests for finding attention modules."""

    def test_finds_all_attention(self) -> None:
        """Should find all attention modules in a model."""
        model = DummyModel(num_blocks=4)
        modules = find_attention_modules(model)
        assert len(modules) == 4

    def test_correct_names(self) -> None:
        """Found modules should have correct names."""
        model = DummyModel(num_blocks=2)
        modules = find_attention_modules(model)
        names = [m.name for m in modules]
        assert "blocks.0.attention" in names
        assert "blocks.1.attention" in names

    def test_infers_d_model(self) -> None:
        """Should correctly infer d_model from attention modules."""
        model = DummyModel(num_blocks=1, d_model=128)
        modules = find_attention_modules(model)
        assert modules[0].d_model == 128

    def test_no_attention(self) -> None:
        """Should return empty list for model without attention."""
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16))
        modules = find_attention_modules(model)
        assert len(modules) == 0


class TestReplaceModule:
    """Tests for module replacement."""

    def test_replace_named_module(self) -> None:
        """Should replace a module at the given path."""
        model = DummyModel(num_blocks=2)
        new_module = MambaBlock(d_model=64, d_state=4)
        replace_module(model, "blocks.0.attention", new_module)

        # Check replacement happened
        assert isinstance(model.blocks[0].attention, MambaBlock)
        assert isinstance(model.blocks[1].attention, DummyAttention)

    def test_model_still_forwards(self) -> None:
        """Model should still forward pass after replacement."""
        model = DummyModel(num_blocks=2, d_model=64)
        new_module = MambaBlock(d_model=64, d_state=4)
        replace_module(model, "blocks.0.attention", new_module)

        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_invalid_path_raises(self) -> None:
        """Should raise ValueError for nonexistent path."""
        model = DummyModel(num_blocks=1)
        new_module = MambaBlock(d_model=64)
        with pytest.raises(ValueError):
            replace_module(model, "blocks.99.attention", new_module)


class TestCreateMambaReplacement:
    """Tests for creating matching Mamba blocks."""

    def test_matching_dimensions(self) -> None:
        """Created block should match d_model."""
        cfg = MambaConfig(d_state=8, d_conv=4, expand=2)
        block = create_mamba_replacement(d_model=128, mamba_config=cfg)
        assert block.d_model == 128
        assert block.d_state == 8

    def test_dtype_applied(self) -> None:
        """Block should be in the specified dtype."""
        cfg = MambaConfig()
        block = create_mamba_replacement(d_model=64, mamba_config=cfg, dtype=torch.float16)
        param = next(block.parameters())
        assert param.dtype == torch.float16


class TestSelectBlocks:
    """Tests for surgery strategy selection."""

    def _make_config(self, strategy: str, **kwargs: Any) -> SurgeryConfig:
        """Helper to create a SurgeryConfig for testing."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        return SurgeryConfig(
            output_dir=tmpdir,
            strategy=strategy,
            **kwargs,
        )

    def test_strategy_all(self) -> None:
        """'all' strategy should select every module."""
        model = DummyModel(num_blocks=4)
        modules = find_attention_modules(model)
        config = self._make_config("all")
        selected = select_blocks_to_replace(modules, config)
        assert len(selected) == 4

    def test_strategy_progressive(self) -> None:
        """'progressive' should select first N modules."""
        model = DummyModel(num_blocks=4)
        modules = find_attention_modules(model)
        config = self._make_config("progressive", num_replace=2)
        selected = select_blocks_to_replace(modules, config)
        assert len(selected) == 2
        assert selected[0].name == "blocks.0.attention"
        assert selected[1].name == "blocks.1.attention"

    def test_strategy_alternating(self) -> None:
        """'alternating' should select every other module."""
        model = DummyModel(num_blocks=4)
        modules = find_attention_modules(model)
        config = self._make_config("alternating")
        selected = select_blocks_to_replace(modules, config)
        assert len(selected) == 2
        assert selected[0].name == "blocks.0.attention"
        assert selected[1].name == "blocks.2.attention"


# Need this for the Any type hint in _make_config
from typing import Any
