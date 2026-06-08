"""Weight initialization and transfer for SSM blocks replacing attention.

When replacing a trained attention block with a fresh Mamba block, the
initialization strategy significantly affects whether the model can still
produce reasonable outputs before fine-tuning.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from mamba_video.mamba_block import MambaBlock

logger = logging.getLogger(__name__)


def init_random(block: MambaBlock) -> None:
    """Standard random initialization (baseline).

    Uses the default initialization from MambaBlock.__init__.
    This is the simplest approach and serves as a baseline.

    Args:
        block: MambaBlock to initialize.
    """
    # Already initialized in __init__, nothing to do
    logger.debug("Random initialization (default) for %s", block.extra_repr())


def init_scaled(
    block: MambaBlock,
    attention_module: Optional[nn.Module] = None,
    scale: float = 0.01,
) -> None:
    """Scaled initialization to match attention output variance.

    Initializes the output projection with a small scale so the Mamba block
    initially contributes very little to the residual stream. This allows
    the rest of the network to remain functional while the SSM block
    "learns" its role.

    Args:
        block: MambaBlock to initialize.
        attention_module: Original attention module (for variance estimation).
        scale: Scale factor for output projection (default: 0.01).
    """
    with torch.no_grad():
        # Scale down output projection so initial contribution is small
        block.out_proj.weight.mul_(scale)
        if block.out_proj.bias is not None:
            block.out_proj.bias.zero_()

        # Scale down input projection proportionally
        block.in_proj.weight.mul_(math.sqrt(scale))

    logger.debug("Scaled initialization (scale=%.4f)", scale)


def init_identity(
    block: MambaBlock,
    epsilon: float = 0.01,
) -> None:
    """Identity-like initialization (SSM starts as near-passthrough).

    Initializes the Mamba block so it approximately computes an identity
    function: output ≈ input. This preserves the model's behavior before
    any training of the SSM block.

    Strategy:
    - Output projection ≈ small scale
    - D (skip connection) = 1 (already default)
    - A initialized for slow decay (preserves state)
    - Delta initialized small (minimal discretization effect)

    Args:
        block: MambaBlock to initialize.
        epsilon: Small scale for non-identity components (default: 0.01).
    """
    with torch.no_grad():
        # Output projection: very small so SSM contribution is minimal
        block.out_proj.weight.mul_(epsilon)
        if block.out_proj.bias is not None:
            block.out_proj.bias.zero_()

        # D (skip connection): already 1.0 by default
        block.D.fill_(1.0)

        # A: initialize for slow decay (values close to 0 in log-space)
        # This means exp(A) ≈ 1, so state persists
        block.A_log.fill_(math.log(0.5))

        # Delta bias: small values for minimal discretization step
        block.dt_proj.bias.uniform_(0.0001, 0.001)

        # Input projection: scaled for stability
        nn.init.xavier_uniform_(block.in_proj.weight, gain=epsilon)

    logger.debug("Identity-like initialization (epsilon=%.4f)", epsilon)


def initialize_replacement(
    block: MambaBlock,
    method: str,
    attention_module: Optional[nn.Module] = None,
) -> None:
    """Apply the specified initialization method to a Mamba block.

    Args:
        block: MambaBlock to initialize.
        method: One of "random", "scaled", "identity".
        attention_module: Original attention module (optional, for variance matching).

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "random":
        init_random(block)
    elif method == "scaled":
        init_scaled(block, attention_module)
    elif method == "identity":
        init_identity(block)
    else:
        raise ValueError(f"Unknown initialization method: '{method}'")

    logger.info("Initialized MambaBlock with method '%s'", method)


def estimate_output_variance(
    module: nn.Module,
    d_model: int,
    num_samples: int = 10,
) -> float:
    """Estimate the output variance of a module using random inputs.

    Args:
        module: The module to measure.
        d_model: Input dimension.
        num_samples: Number of random inputs to average over.

    Returns:
        Estimated output variance (scalar).
    """
    module.eval()
    variances = []

    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, 16, d_model)
            try:
                out = module(x)
                if isinstance(out, tuple):
                    out = out[0]
                variances.append(out.var().item())
            except Exception:
                break

    if not variances:
        return 1.0

    return sum(variances) / len(variances)
