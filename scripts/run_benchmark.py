#!/usr/bin/env python
"""Benchmark original vs modified model.

Usage:
    python scripts/run_benchmark.py --original wan-1.3b --modified results/model_modified/
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark original vs modified model")
    parser.add_argument("--original", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="Original model name")
    parser.add_argument("--original-path", type=str, default=None, help="Path to original weights")
    parser.add_argument("--modified", type=str, required=True, help="Path to modified model dir")
    parser.add_argument("--output", type=str, default="results/benchmarks")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    import torch
    from diffusers import WanTransformer3DModel

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    # Load original (Wan diffusion transformer)
    logger.info("Loading original model...")
    orig_path = args.original_path or args.original
    original = WanTransformer3DModel.from_pretrained(
        orig_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )
    original.eval()

    # Load modified
    logger.info("Loading modified model...")
    state_path = os.path.join(args.modified, "state_dict.pt")
    modified = WanTransformer3DModel.from_pretrained(
        orig_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )
    modified.load_state_dict(torch.load(state_path, map_location="cpu"), strict=False)
    modified.eval()

    # Run comparison
    from mamba_video.benchmark import compare_models
    from mamba_video.report import save_comparison_report, format_comparison_summary

    comparison = compare_models(
        original=original,
        modified=modified,
        num_warmup=args.warmup,
        num_steps=args.steps,
        input_frames=args.frames,
        input_resolution=tuple(args.resolution),
        dtype=torch_dtype,
    )

    save_comparison_report(comparison, args.output)
    print(format_comparison_summary(comparison))

    return 0


if __name__ == "__main__":
    sys.exit(main())
