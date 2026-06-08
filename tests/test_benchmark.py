"""Tests for mamba_video.benchmark module."""

import pytest
import torch
import torch.nn as nn

from mamba_video.benchmark import (
    BenchmarkResult,
    ComparisonResult,
    benchmark_model,
)


class SimpleModel(nn.Module):
    """Tiny model for benchmark testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_avg_time(self) -> None:
        result = BenchmarkResult(
            model_name="test",
            times_ms=[10.0, 20.0, 30.0],
        )
        assert result.avg_time_ms == pytest.approx(20.0)

    def test_std_time(self) -> None:
        result = BenchmarkResult(
            model_name="test",
            times_ms=[10.0, 10.0, 10.0],
        )
        assert result.std_time_ms == pytest.approx(0.0)

    def test_min_time(self) -> None:
        result = BenchmarkResult(
            model_name="test",
            times_ms=[30.0, 10.0, 20.0],
        )
        assert result.min_time_ms == pytest.approx(10.0)

    def test_empty_times(self) -> None:
        result = BenchmarkResult(model_name="test")
        assert result.avg_time_ms == 0.0
        assert result.min_time_ms == 0.0


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_speedup(self) -> None:
        orig = BenchmarkResult(model_name="orig", times_ms=[100.0])
        mod = BenchmarkResult(model_name="mod", times_ms=[50.0])
        comp = ComparisonResult(original=orig, modified=mod)
        assert comp.speedup == pytest.approx(2.0)

    def test_memory_savings(self) -> None:
        orig = BenchmarkResult(model_name="orig", peak_memory_mb=1000)
        mod = BenchmarkResult(model_name="mod", peak_memory_mb=800)
        comp = ComparisonResult(original=orig, modified=mod)
        assert comp.memory_savings_mb == pytest.approx(200.0)

    def test_param_reduction(self) -> None:
        orig = BenchmarkResult(model_name="orig", param_count=1000000)
        mod = BenchmarkResult(model_name="mod", param_count=750000)
        comp = ComparisonResult(original=orig, modified=mod)
        assert comp.param_reduction_pct == pytest.approx(25.0)


class TestBenchmarkModel:
    """Tests for the benchmark_model function."""

    def test_produces_results(self) -> None:
        """Should produce timing results for a simple model."""
        model = SimpleModel()
        model.eval()

        # We need to monkey-patch to accept dict input
        class WrappedModel(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
                b = latents.shape[0]
                flat = latents.reshape(b, -1)[:, :64]
                if flat.shape[-1] < 64:
                    flat = torch.nn.functional.pad(flat, (0, 64 - flat.shape[-1]))
                return self.inner(flat)

        wrapped = WrappedModel(model)
        wrapped.eval()

        result = benchmark_model(
            wrapped, "test",
            num_warmup=1, num_steps=3,
            input_frames=2, input_resolution=(16, 16),
            dtype=torch.float32,
        )

        assert len(result.times_ms) == 3
        assert all(t > 0 for t in result.times_ms)
        assert result.param_count > 0
