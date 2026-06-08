"""Report generation for mamba-video surgery and benchmark results."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def surgery_report_to_dict(report: Any) -> Dict[str, Any]:
    """Convert a SurgeryReport to a serializable dictionary.

    Args:
        report: A SurgeryReport instance.

    Returns:
        Dictionary for JSON serialization.
    """
    return {
        "meta": {
            "tool": "mamba-video",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "phase": 2,
        },
        "surgery": {
            "model_name": report.model_name,
            "strategy": report.strategy,
            "total_attention_blocks": report.total_attention_blocks,
            "replaced_blocks": report.replaced_blocks,
            "replaced_names": report.replaced_names,
            "kept_names": report.kept_names,
            "mamba_config": report.mamba_config,
            "original_params": report.original_params,
            "modified_params": report.modified_params,
            "param_change_pct": round(
                (report.modified_params - report.original_params)
                / max(report.original_params, 1) * 100, 2
            ),
        },
    }


def comparison_to_dict(comparison: Any) -> Dict[str, Any]:
    """Convert a ComparisonResult to a serializable dictionary.

    Args:
        comparison: A ComparisonResult instance.

    Returns:
        Dictionary for JSON serialization.
    """
    return {
        "meta": {
            "tool": "mamba-video",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "phase": 2,
        },
        "benchmark": {
            "original": {
                "name": comparison.original.model_name,
                "avg_time_ms": round(comparison.original.avg_time_ms, 3),
                "std_time_ms": round(comparison.original.std_time_ms, 3),
                "min_time_ms": round(comparison.original.min_time_ms, 3),
                "peak_memory_mb": round(comparison.original.peak_memory_mb, 1),
                "param_count": comparison.original.param_count,
                "all_times_ms": [round(t, 3) for t in comparison.original.times_ms],
            },
            "modified": {
                "name": comparison.modified.model_name,
                "avg_time_ms": round(comparison.modified.avg_time_ms, 3),
                "std_time_ms": round(comparison.modified.std_time_ms, 3),
                "min_time_ms": round(comparison.modified.min_time_ms, 3),
                "peak_memory_mb": round(comparison.modified.peak_memory_mb, 1),
                "param_count": comparison.modified.param_count,
                "all_times_ms": [round(t, 3) for t in comparison.modified.times_ms],
            },
            "comparison": {
                "speedup": round(comparison.speedup, 3),
                "memory_savings_mb": round(comparison.memory_savings_mb, 1),
                "param_reduction_pct": round(comparison.param_reduction_pct, 2),
            },
        },
        "quality": comparison.quality_metrics or {},
    }


def save_surgery_report(
    report: Any,
    output_dir: str,
    filename: str = "surgery_report.json",
) -> str:
    """Save surgery report as JSON.

    Args:
        report: SurgeryReport instance.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(surgery_report_to_dict(report), f, indent=2)

    logger.info("Surgery report saved to: %s", path)
    return path


def save_comparison_report(
    comparison: Any,
    output_dir: str,
    filename: str = "benchmark_comparison.json",
) -> str:
    """Save benchmark comparison as JSON.

    Args:
        comparison: ComparisonResult instance.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison_to_dict(comparison), f, indent=2)

    logger.info("Comparison report saved to: %s", path)
    return path


def format_surgery_summary(report: Any) -> str:
    """Format a human-readable surgery summary.

    Args:
        report: SurgeryReport instance.

    Returns:
        Formatted string for printing.
    """
    lines = [
        "",
        "=" * 70,
        "  mamba-video Surgery Report",
        "=" * 70,
        "",
        f"  Model:     {report.model_name}",
        f"  Strategy:  {report.strategy}",
        "",
        f"  Attention blocks found:    {report.total_attention_blocks}",
        f"  Blocks replaced:           {report.replaced_blocks}",
        f"  Blocks kept:               {len(report.kept_names)}",
        "",
        f"  Original parameters:       {report.original_params:,}",
        f"  Modified parameters:       {report.modified_params:,}",
        f"  Change:                    {report.modified_params - report.original_params:+,}",
        "",
        f"  Mamba config:              {report.mamba_config}",
        "",
    ]

    if report.replaced_names:
        lines.append("  Replaced modules:")
        for name in report.replaced_names:
            lines.append(f"    ✓ {name}")
        lines.append("")

    if report.kept_names:
        lines.append("  Kept modules:")
        for name in report.kept_names:
            lines.append(f"    ○ {name}")
        lines.append("")

    lines.extend(["=" * 70, ""])
    return "\n".join(lines)


def format_comparison_summary(comparison: Any) -> str:
    """Format a human-readable benchmark comparison.

    Args:
        comparison: ComparisonResult instance.

    Returns:
        Formatted string for printing.
    """
    orig = comparison.original
    mod = comparison.modified

    lines = [
        "",
        "=" * 70,
        "  mamba-video Benchmark Comparison",
        "=" * 70,
        "",
        f"  {'Metric':<30} {'Original':>15} {'Modified':>15}",
        "  " + "-" * 62,
        f"  {'Avg time (ms)':<30} {orig.avg_time_ms:>15.1f} {mod.avg_time_ms:>15.1f}",
        f"  {'Std time (ms)':<30} {orig.std_time_ms:>15.1f} {mod.std_time_ms:>15.1f}",
        f"  {'Min time (ms)':<30} {orig.min_time_ms:>15.1f} {mod.min_time_ms:>15.1f}",
        f"  {'Peak memory (MB)':<30} {orig.peak_memory_mb:>15.0f} {mod.peak_memory_mb:>15.0f}",
        f"  {'Parameters':<30} {orig.param_count:>15,} {mod.param_count:>15,}",
        "  " + "-" * 62,
        "",
        f"  Speedup:            {comparison.speedup:.2f}x",
        f"  Memory saved:       {comparison.memory_savings_mb:.0f} MB",
        f"  Param reduction:    {comparison.param_reduction_pct:.1f}%",
        "",
    ]

    if comparison.quality_metrics:
        lines.append("  Quality Metrics:")
        lines.append("  " + "-" * 40)
        for key, val in comparison.quality_metrics.items():
            if isinstance(val, float):
                lines.append(f"  {key:<30} {val:>10.4f}")
            else:
                lines.append(f"  {key:<30} {val!s:>10}")
        lines.append("")

    lines.extend(["=" * 70, ""])
    return "\n".join(lines)
