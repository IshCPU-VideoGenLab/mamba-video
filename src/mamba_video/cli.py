"""Command-line interface for mamba-video architecture surgery."""

import argparse
import logging
import sys
from typing import List, Optional

from mamba_video.config import MambaConfig, SurgeryConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mamba-video",
        description="Replace transformer attention with Mamba SSM blocks in Wan 1.3B.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- surgery subcommand ---
    surgery_parser = subparsers.add_parser("surgery", help="Run architecture surgery")
    surgery_parser.add_argument("--model", type=str, default="wan-1.3b")
    surgery_parser.add_argument("--model-path", type=str, default=None)
    surgery_parser.add_argument("--output", type=str, default="results")
    surgery_parser.add_argument(
        "--strategy", type=str, default="progressive",
        choices=["all", "progressive", "by-cost", "alternating"],
    )
    surgery_parser.add_argument("--num-replace", type=int, default=None)
    surgery_parser.add_argument("--profiling-data", type=str, default=None)
    surgery_parser.add_argument(
        "--init-method", type=str, default="scaled",
        choices=["random", "scaled", "identity"],
    )
    surgery_parser.add_argument("--d-state", type=int, default=16)
    surgery_parser.add_argument("--d-conv", type=int, default=4)
    surgery_parser.add_argument("--expand", type=int, default=2)
    surgery_parser.add_argument("--dtype", type=str, default="float16")
    surgery_parser.add_argument("--quiet", action="store_true")
    surgery_parser.add_argument("--debug", action="store_true")

    # --- benchmark subcommand ---
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark original vs modified")
    bench_parser.add_argument("--model", type=str, default="wan-1.3b")
    bench_parser.add_argument("--model-path", type=str, default=None)
    bench_parser.add_argument("--modified-path", type=str, required=True)
    bench_parser.add_argument("--output", type=str, default="results/benchmarks")
    bench_parser.add_argument("--warmup", type=int, default=2)
    bench_parser.add_argument("--steps", type=int, default=5)
    bench_parser.add_argument("--dtype", type=str, default="float16")
    bench_parser.add_argument("--quiet", action="store_true")
    bench_parser.add_argument("--debug", action="store_true")

    # --- inspect subcommand ---
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect model attention modules without modifying"
    )
    inspect_parser.add_argument("--model", type=str, default="wan-1.3b")
    inspect_parser.add_argument("--model-path", type=str, default=None)
    inspect_parser.add_argument("--dtype", type=str, default="float16")
    inspect_parser.add_argument("--debug", action="store_true")

    return parser.parse_args(argv)


def setup_logging(debug: bool = False, quiet: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_surgery(args: argparse.Namespace) -> int:
    """Execute the surgery command."""
    from mamba_video.surgery import perform_surgery
    from mamba_video.report import (
        save_surgery_report,
        format_surgery_summary,
    )

    mamba_cfg = MambaConfig(
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    )

    config = SurgeryConfig(
        model_name=args.model,
        model_path=args.model_path,
        output_dir=args.output,
        strategy=args.strategy,
        num_replace=args.num_replace,
        profiling_data=args.profiling_data,
        init_method=args.init_method,
        mamba=mamba_cfg,
        dtype=args.dtype,
        verbose=not args.quiet,
    )

    model, report = perform_surgery(config)

    # Save report
    save_surgery_report(report, config.output_dir)

    # Print summary
    if not args.quiet:
        print(format_surgery_summary(report))

    # Save modified model state dict
    import torch
    import os

    model_dir = os.path.join(config.output_dir, "model_modified")
    os.makedirs(model_dir, exist_ok=True)
    state_path = os.path.join(model_dir, "state_dict.pt")
    torch.save(model.state_dict(), state_path)

    logger = logging.getLogger(__name__)
    logger.info("Modified model saved to: %s", state_path)

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Execute the inspect command."""
    import torch
    from transformers import AutoModel
    from mamba_video.surgery import find_attention_modules

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    logger = logging.getLogger(__name__)
    logger.info("Loading model for inspection...")

    load_path = args.model_path or args.model
    model = AutoModel.from_pretrained(
        load_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    modules = find_attention_modules(model)

    print(f"\nFound {len(modules)} attention modules:\n")
    print(f"  {'#':<4} {'Name':<45} {'Type':<25} {'d_model':>8} {'Params':>12}")
    print("  " + "-" * 96)

    for i, m in enumerate(modules):
        print(
            f"  {i:<4} {m.name:<45} {m.module_type:<25} "
            f"{m.d_model:>8} {m.param_count:>12,}"
        )

    total_params = sum(m.param_count for m in modules)
    print("  " + "-" * 96)
    print(f"  {'Total attention params':>50} {total_params:>37,}")
    print()

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if args.command is None:
        print("Usage: mamba-video {surgery|benchmark|inspect} [options]")
        print("Run 'mamba-video <command> --help' for details.")
        return 1

    setup_logging(
        debug=getattr(args, "debug", False),
        quiet=getattr(args, "quiet", False),
    )

    if args.command == "surgery":
        return cmd_surgery(args)
    elif args.command == "benchmark":
        logger = logging.getLogger(__name__)
        logger.info("Benchmark command — use scripts/run_benchmark.py for now")
        return 0
    elif args.command == "inspect":
        return cmd_inspect(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
