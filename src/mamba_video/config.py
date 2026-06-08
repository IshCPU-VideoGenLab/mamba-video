"""Configuration for mamba-video architecture surgery."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


VALID_STRATEGIES = {"all", "progressive", "by-cost", "alternating"}
VALID_INIT_METHODS = {"random", "scaled", "identity"}


@dataclass
class MambaConfig:
    """Configuration for the Mamba SSM block.

    Args:
        d_state: SSM state dimension. Keep ≤ 16 for memory-constrained machines.
        d_conv: Local convolution kernel width.
        expand: Expansion factor for inner dimension (d_inner = expand * d_model).
        dt_min: Minimum value for delta (Δ) clamping.
        dt_max: Maximum value for delta (Δ) clamping.
        dt_init: Initialization method for delta projection ("random" or "constant").
        dt_scale: Scale factor for delta initialization.
        bias: Whether to use bias in linear projections.
        conv_bias: Whether to use bias in the Conv1D layer.
    """

    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_min: float = 1e-4
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self) -> None:
        if self.d_state < 1:
            raise ValueError("d_state must be at least 1")
        if self.d_conv < 1:
            raise ValueError("d_conv must be at least 1")
        if self.expand < 1:
            raise ValueError("expand must be at least 1")


@dataclass
class SurgeryConfig:
    """Configuration for architecture surgery.

    Args:
        model_name: Model name or HuggingFace ID.
        model_path: Local path to model weights.
        output_dir: Directory to save modified model and results.
        strategy: Surgery strategy ("all", "progressive", "by-cost", "alternating").
        num_replace: Number of blocks to replace (for "progressive" strategy).
        block_indices: Specific block indices to replace (overrides strategy).
        profiling_data: Path to Phase 1 profiling results (for "by-cost" strategy).
        init_method: Weight initialization method ("random", "scaled", "identity").
        mamba: Mamba block configuration.
        dtype: Data type for model loading.
        low_memory: Enable memory-efficient loading.
        verbose: Print progress.
    """

    model_name: str = "wan-1.3b"
    model_path: Optional[str] = None
    output_dir: str = "results"
    strategy: str = "progressive"
    num_replace: Optional[int] = None
    block_indices: Optional[List[int]] = None
    profiling_data: Optional[str] = None
    init_method: str = "scaled"
    mamba: MambaConfig = field(default_factory=MambaConfig)
    dtype: str = "float16"
    low_memory: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Must be one of: {VALID_STRATEGIES}"
            )

        if self.init_method not in VALID_INIT_METHODS:
            raise ValueError(
                f"Invalid init_method '{self.init_method}'. "
                f"Must be one of: {VALID_INIT_METHODS}"
            )

        if self.strategy == "by-cost" and self.profiling_data is None:
            raise ValueError(
                "Strategy 'by-cost' requires --profiling-data path to "
                "Phase 1 results (profile_results.json)"
            )

        if self.strategy == "progressive" and self.num_replace is None:
            raise ValueError(
                "Strategy 'progressive' requires --num-replace"
            )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking original vs modified model.

    Args:
        original_name: Original model name.
        modified_path: Path to modified model state dict.
        output_dir: Directory for benchmark results.
        num_warmup: Warmup iterations.
        num_steps: Measurement iterations.
        input_frames: Number of video frames for dummy input.
        input_resolution: Resolution (H, W) for dummy input.
        dtype: Data type.
    """

    original_name: str = "wan-1.3b"
    modified_path: Optional[str] = None
    output_dir: str = "results/benchmarks"
    num_warmup: int = 2
    num_steps: int = 5
    input_frames: int = 8
    input_resolution: tuple = (256, 256)
    dtype: str = "float16"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
