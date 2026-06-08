#!/usr/bin/env python
"""Compare outputs from original and modified models visually.

Generates side-by-side comparison plots and computes quality metrics.

Usage:
    python scripts/compare_outputs.py \
        --original-output results/original_frames/ \
        --modified-output results/modified_frames/
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare model outputs visually")
    parser.add_argument("--original-output", type=str, required=True)
    parser.add_argument("--modified-output", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/quality")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    os.makedirs(args.output, exist_ok=True)

    # Check for saved tensors
    orig_files = sorted(f for f in os.listdir(args.original_output) if f.endswith(".pt"))
    mod_files = sorted(f for f in os.listdir(args.modified_output) if f.endswith(".pt"))

    if not orig_files or not mod_files:
        logger.error("No .pt tensor files found in output directories.")
        logger.info("Run surgery first to generate outputs.")
        return 1

    import torch
    from mamba_video.quality import compare_outputs

    all_metrics = []

    for orig_f, mod_f in zip(orig_files, mod_files):
        orig_tensor = torch.load(os.path.join(args.original_output, orig_f), map_location="cpu")
        mod_tensor = torch.load(os.path.join(args.modified_output, mod_f), map_location="cpu")

        metrics = compare_outputs(orig_tensor, mod_tensor)
        metrics["file"] = orig_f
        all_metrics.append(metrics)

        logger.info(
            "%s: MSE=%.6f, PSNR=%.2f dB, cosine=%.4f",
            orig_f, metrics["mse"], metrics["psnr_db"], metrics["cosine_similarity"],
        )

    # Save aggregate results
    results_path = os.path.join(args.output, "quality_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("\nQuality results saved to: %s", results_path)

    # Try to generate plots
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = [m["file"] for m in all_metrics]
        mses = [m["mse"] for m in all_metrics]
        psnrs = [m["psnr_db"] for m in all_metrics]
        cosines = [m["cosine_similarity"] for m in all_metrics]

        axes[0].bar(names, mses, color="#F44336")
        axes[0].set_title("MSE (lower = better)")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(names, psnrs, color="#4CAF50")
        axes[1].set_title("PSNR dB (higher = better)")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(names, cosines, color="#2196F3")
        axes[2].set_title("Cosine Similarity (1.0 = identical)")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        chart_path = os.path.join(args.output, "quality_comparison.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Chart saved to: %s", chart_path)

    except ImportError:
        logger.info("matplotlib not available — skipping chart generation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
