"""Speed and memory benchmarking for original vs modified models.

Runs timed forward passes on both models under identical conditions
and produces a comparison report.
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model.

    Args:
        model_name: Identifier for this model variant.
        times_ms: Wall-clock times per forward pass.
        peak_memory_mb: Peak process memory during benchmark.
        param_count: Total parameter count.
    """

    model_name: str
    times_ms: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    param_count: int = 0

    @property
    def avg_time_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def std_time_ms(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        avg = self.avg_time_ms
        var = sum((t - avg) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)
        return var ** 0.5

    @property
    def min_time_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0


@dataclass
class ComparisonResult:
    """Side-by-side comparison of original vs modified model.

    Args:
        original: Benchmark results for the original model.
        modified: Benchmark results for the modified model.
        quality_metrics: Quality comparison metrics.
    """

    original: BenchmarkResult
    modified: BenchmarkResult
    quality_metrics: Optional[Dict[str, float]] = None

    @property
    def speedup(self) -> float:
        """Speedup ratio (>1 means modified is faster)."""
        if self.modified.avg_time_ms == 0:
            return 0.0
        return self.original.avg_time_ms / self.modified.avg_time_ms

    @property
    def memory_savings_mb(self) -> float:
        """Memory saved in MB (positive means modified uses less)."""
        return self.original.peak_memory_mb - self.modified.peak_memory_mb

    @property
    def param_reduction_pct(self) -> float:
        """Percentage reduction in parameters."""
        if self.original.param_count == 0:
            return 0.0
        diff = self.original.param_count - self.modified.param_count
        return (diff / self.original.param_count) * 100


def create_dummy_input(
    num_frames: int = 8,
    resolution: Tuple[int, int] = (256, 256),
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Create a dummy input for forward pass benchmarking.

    Args:
        num_frames: Number of video frames.
        resolution: (height, width).
        dtype: Tensor data type.

    Returns:
        Dictionary of input tensors.
    """
    h, w = resolution
    latent_h, latent_w = h // 8, w // 8

    return {
        "latents": torch.randn(1, 4, num_frames, latent_h, latent_w, dtype=dtype),
        "timestep": torch.tensor([500], dtype=torch.long),
    }


def _run_forward(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Any:
    """Run a forward pass with error handling for different model interfaces.

    Args:
        model: The model.
        inputs: Input dictionary.

    Returns:
        Model output.
    """
    try:
        return model(**inputs)
    except Exception:
        try:
            return model(inputs["latents"], inputs["timestep"])
        except Exception:
            return model(inputs["latents"])


def benchmark_model(
    model: nn.Module,
    model_name: str,
    num_warmup: int = 2,
    num_steps: int = 5,
    input_frames: int = 8,
    input_resolution: Tuple[int, int] = (256, 256),
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Run timed forward passes on a model.

    Args:
        model: PyTorch model in eval mode.
        model_name: Identifier for this model variant.
        num_warmup: Number of warmup iterations.
        num_steps: Number of timed iterations.
        input_frames: Number of video frames.
        input_resolution: Resolution (H, W).
        dtype: Data type for inputs.

    Returns:
        BenchmarkResult with timing data.
    """
    import psutil

    model.eval()
    inputs = create_dummy_input(input_frames, input_resolution, dtype)
    param_count = sum(p.numel() for p in model.parameters())

    # Warmup
    logger.info("Warmup: %d steps for '%s'...", num_warmup, model_name)
    with torch.no_grad():
        for _ in range(num_warmup):
            _run_forward(model, inputs)
            gc.collect()

    # Benchmark
    logger.info("Benchmarking: %d steps for '%s'...", num_steps, model_name)
    times = []
    peak_mem = 0.0

    with torch.no_grad():
        for i in range(num_steps):
            gc.collect()

            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            start = time.perf_counter()
            _run_forward(model, inputs)
            elapsed = (time.perf_counter() - start) * 1000
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

            times.append(elapsed)
            peak_mem = max(peak_mem, mem_after)

            logger.info(
                "  Step %d/%d: %.1f ms (mem: %.0f MB)",
                i + 1, num_steps, elapsed, mem_after,
            )

    result = BenchmarkResult(
        model_name=model_name,
        times_ms=times,
        peak_memory_mb=peak_mem,
        param_count=param_count,
    )

    logger.info(
        "'%s' benchmark: avg=%.1f ms, std=%.1f ms, peak_mem=%.0f MB, params=%s",
        model_name, result.avg_time_ms, result.std_time_ms,
        result.peak_memory_mb, f"{param_count:,}",
    )

    return result


def compare_models(
    original: nn.Module,
    modified: nn.Module,
    num_warmup: int = 2,
    num_steps: int = 5,
    input_frames: int = 8,
    input_resolution: Tuple[int, int] = (256, 256),
    dtype: torch.dtype = torch.float16,
) -> ComparisonResult:
    """Benchmark and compare original vs modified model.

    Also computes quality metrics on the outputs.

    Args:
        original: Original model.
        modified: Modified (post-surgery) model.
        num_warmup: Warmup iterations.
        num_steps: Measurement iterations.
        input_frames: Number of video frames.
        input_resolution: Resolution.
        dtype: Data type.

    Returns:
        ComparisonResult with benchmarks and quality metrics.
    """
    from mamba_video.quality import compare_outputs

    # Benchmark original
    orig_result = benchmark_model(
        original, "original",
        num_warmup, num_steps, input_frames, input_resolution, dtype,
    )

    # Benchmark modified
    mod_result = benchmark_model(
        modified, "modified",
        num_warmup, num_steps, input_frames, input_resolution, dtype,
    )

    # Quality comparison on same input
    inputs = create_dummy_input(input_frames, input_resolution, dtype)
    with torch.no_grad():
        orig_output = _run_forward(original, inputs)
        mod_output = _run_forward(modified, inputs)

    # Handle tuple outputs
    if isinstance(orig_output, tuple):
        orig_output = orig_output[0]
    if isinstance(mod_output, tuple):
        mod_output = mod_output[0]

    quality = compare_outputs(orig_output, mod_output)

    comparison = ComparisonResult(
        original=orig_result,
        modified=mod_result,
        quality_metrics=quality,
    )

    logger.info(
        "Comparison: speedup=%.2fx, memory_saved=%.0f MB, param_reduction=%.1f%%",
        comparison.speedup, comparison.memory_savings_mb,
        comparison.param_reduction_pct,
    )

    return comparison
