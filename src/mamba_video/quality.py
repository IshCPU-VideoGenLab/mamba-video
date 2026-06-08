"""Quality evaluation metrics for comparing original vs modified model outputs.

Provides FID, LPIPS, and simple pixel-level metrics. FVD is documented
but left as a stub — it requires significant compute and may not be
feasible on the Pentium Gold for full video evaluation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_mse(
    original: torch.Tensor,
    modified: torch.Tensor,
) -> float:
    """Compute Mean Squared Error between two tensors.

    Args:
        original: Reference tensor.
        modified: Test tensor (same shape as original).

    Returns:
        MSE value (lower is better).
    """
    return ((original.float() - modified.float()) ** 2).mean().item()


def compute_psnr(
    original: torch.Tensor,
    modified: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Reference tensor (values in [0, max_val]).
        modified: Test tensor.
        max_val: Maximum pixel value.

    Returns:
        PSNR in dB (higher is better).
    """
    mse = compute_mse(original, modified)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def compute_cosine_similarity(
    original: torch.Tensor,
    modified: torch.Tensor,
) -> float:
    """Compute cosine similarity in feature space.

    Useful for comparing latent representations rather than pixel values.

    Args:
        original: Reference tensor.
        modified: Test tensor.

    Returns:
        Cosine similarity (1.0 = identical, 0.0 = orthogonal).
    """
    orig_flat = original.float().flatten()
    mod_flat = modified.float().flatten()

    dot = torch.dot(orig_flat, mod_flat)
    norm_orig = torch.norm(orig_flat)
    norm_mod = torch.norm(mod_flat)

    if norm_orig == 0 or norm_mod == 0:
        return 0.0

    return (dot / (norm_orig * norm_mod)).item()


def compute_output_statistics(
    tensor: torch.Tensor,
) -> Dict[str, float]:
    """Compute basic statistics of a tensor output.

    Useful for sanity-checking that the modified model produces
    outputs in a reasonable range.

    Args:
        tensor: Output tensor to analyze.

    Returns:
        Dictionary with mean, std, min, max, and norm.
    """
    t = tensor.float()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "norm": t.norm().item(),
        "num_nan": torch.isnan(t).sum().item(),
        "num_inf": torch.isinf(t).sum().item(),
    }


def compare_outputs(
    original_output: torch.Tensor,
    modified_output: torch.Tensor,
) -> Dict[str, float]:
    """Compute all available quality metrics between two outputs.

    This is the main quality comparison function. Computes pixel-level
    and feature-level metrics.

    Args:
        original_output: Output from the original model.
        modified_output: Output from the modified (surgery) model.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {}

    # Basic metrics (always available, no extra dependencies)
    metrics["mse"] = compute_mse(original_output, modified_output)
    metrics["psnr_db"] = compute_psnr(original_output, modified_output)
    metrics["cosine_similarity"] = compute_cosine_similarity(
        original_output, modified_output
    )

    # Output statistics for sanity checking
    orig_stats = compute_output_statistics(original_output)
    mod_stats = compute_output_statistics(modified_output)

    metrics["original_mean"] = orig_stats["mean"]
    metrics["modified_mean"] = mod_stats["mean"]
    metrics["original_std"] = orig_stats["std"]
    metrics["modified_std"] = mod_stats["std"]
    metrics["modified_num_nan"] = mod_stats["num_nan"]
    metrics["modified_num_inf"] = mod_stats["num_inf"]

    # Mean/std ratio (how close is the modified distribution?)
    if orig_stats["std"] > 0:
        metrics["std_ratio"] = mod_stats["std"] / orig_stats["std"]
    else:
        metrics["std_ratio"] = 0.0

    return metrics


def try_compute_lpips(
    original: torch.Tensor,
    modified: torch.Tensor,
) -> Optional[float]:
    """Attempt to compute LPIPS if torchmetrics is available.

    LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual
    distance using a pretrained network.

    Args:
        original: Reference images, shape (batch, 3, H, W).
        modified: Test images, same shape.

    Returns:
        LPIPS score (lower is better), or None if unavailable.
    """
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
        score = lpips(modified, original)
        return score.item()
    except ImportError:
        logger.debug("torchmetrics not available for LPIPS computation")
        return None
    except Exception as e:
        logger.warning("LPIPS computation failed: %s", str(e))
        return None


def try_compute_fid(
    original_features: torch.Tensor,
    modified_features: torch.Tensor,
) -> Optional[float]:
    """Attempt to compute FID from pre-extracted features.

    FID (Fréchet Inception Distance) measures the distance between
    two distributions of features.

    Args:
        original_features: Features from original outputs, shape (N, D).
        modified_features: Features from modified outputs, shape (N, D).

    Returns:
        FID score (lower is better), or None if computation fails.
    """
    try:
        orig = original_features.float().numpy()
        mod = modified_features.float().numpy()

        # Compute statistics
        mu1, sigma1 = orig.mean(axis=0), np.cov(orig, rowvar=False)
        mu2, sigma2 = mod.mean(axis=0), np.cov(mod, rowvar=False)

        # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu1 - mu2
        diff_sq = np.dot(diff, diff)

        # Matrix square root via eigendecomposition
        product = sigma1 @ sigma2
        eigenvalues = np.linalg.eigvalsh(product)
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
        sqrt_trace = np.sum(np.sqrt(eigenvalues))

        fid = diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * sqrt_trace
        return float(fid)

    except Exception as e:
        logger.warning("FID computation failed: %s", str(e))
        return None
