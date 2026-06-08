"""Architecture surgery engine for replacing attention with Mamba blocks.

Provides tools to inspect a model's attention modules, replace them with
Mamba SSM blocks, and save the modified model.
"""

import copy
import gc
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mamba_video.config import MambaConfig, SurgeryConfig
from mamba_video.mamba_block import MambaBlock

logger = logging.getLogger(__name__)


# Keywords that identify attention modules in named_modules()
ATTENTION_KEYWORDS = [
    "attn", "attention", "self_attn", "cross_attn",
    "multihead", "mha", "mhsa",
]


@dataclass
class AttentionInfo:
    """Information about a discovered attention module.

    Args:
        name: Fully qualified module name.
        module_type: Class name.
        d_model: Hidden dimension (inferred from parameters).
        param_count: Number of parameters.
        parent_name: Name of the parent module.
        index_in_parent: Attribute name or index within parent.
    """

    name: str
    module_type: str
    d_model: int
    param_count: int
    parent_name: str
    index_in_parent: str


@dataclass
class SurgeryReport:
    """Report of what was changed during surgery.

    Args:
        model_name: Original model name.
        strategy: Surgery strategy used.
        total_attention_blocks: Total attention blocks found.
        replaced_blocks: Number of blocks replaced.
        replaced_names: Names of replaced modules.
        kept_names: Names of kept (unreplaced) modules.
        mamba_config: Mamba configuration used.
        original_params: Original model parameter count.
        modified_params: Modified model parameter count.
    """

    model_name: str
    strategy: str
    total_attention_blocks: int
    replaced_blocks: int
    replaced_names: List[str]
    kept_names: List[str]
    mamba_config: Dict[str, Any]
    original_params: int = 0
    modified_params: int = 0


def find_attention_modules(model: nn.Module) -> List[AttentionInfo]:
    """Find all attention modules in a model.

    Searches through named_modules() for modules whose name or type
    contains attention-related keywords.

    Args:
        model: PyTorch model to inspect.

    Returns:
        List of AttentionInfo for each discovered attention module.
    """
    attention_modules = []

    for name, module in model.named_modules():
        name_lower = name.lower()
        type_lower = type(module).__name__.lower()

        is_attention = any(
            kw in name_lower or kw in type_lower
            for kw in ATTENTION_KEYWORDS
        )

        if not is_attention:
            continue

        # Skip if this is a container (has children that are also attention)
        children_names = [n for n, _ in module.named_children()]
        has_attn_children = any(
            any(kw in cn.lower() for kw in ATTENTION_KEYWORDS)
            for cn in children_names
        )
        if has_attn_children:
            continue

        # Infer d_model from parameters
        d_model = _infer_d_model(module)
        param_count = sum(p.numel() for p in module.parameters())

        # Parse parent name and index
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, index_in_parent = parts
        else:
            parent_name = ""
            index_in_parent = name

        attention_modules.append(AttentionInfo(
            name=name,
            module_type=type(module).__name__,
            d_model=d_model,
            param_count=param_count,
            parent_name=parent_name,
            index_in_parent=index_in_parent,
        ))

    logger.info("Found %d attention modules", len(attention_modules))
    for info in attention_modules:
        logger.debug(
            "  %s (%s, d_model=%d, params=%s)",
            info.name, info.module_type, info.d_model, f"{info.param_count:,}",
        )

    return attention_modules


def _infer_d_model(module: nn.Module) -> int:
    """Infer the hidden dimension from an attention module's parameters.

    Tries multiple strategies: checking for common attributes, then
    inspecting weight shapes.

    Args:
        module: An attention module.

    Returns:
        Inferred d_model dimension. Returns 0 if cannot determine.
    """
    # Strategy 1: Check common attributes
    for attr in ["embed_dim", "d_model", "hidden_size", "num_features"]:
        if hasattr(module, attr):
            return getattr(module, attr)

    # Strategy 2: Check query projection weight shape
    for name, param in module.named_parameters():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["q_proj", "query", "qkv"]):
            # For q_proj: weight shape is (out_features, in_features)
            # in_features = d_model
            return param.shape[-1]

    # Strategy 3: Use first linear layer's input dim
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child.in_features

    logger.warning("Could not infer d_model for %s", type(module).__name__)
    return 0


def create_mamba_replacement(
    d_model: int,
    mamba_config: MambaConfig,
    dtype: torch.dtype = torch.float16,
) -> MambaBlock:
    """Create a MambaBlock that matches the dimensions of an attention block.

    Args:
        d_model: Hidden dimension to match.
        mamba_config: Mamba configuration.
        dtype: Data type for parameters.

    Returns:
        A new MambaBlock instance.
    """
    block = MambaBlock(
        d_model=d_model,
        d_state=mamba_config.d_state,
        d_conv=mamba_config.d_conv,
        expand=mamba_config.expand,
        dt_min=mamba_config.dt_min,
        dt_max=mamba_config.dt_max,
        bias=mamba_config.bias,
        conv_bias=mamba_config.conv_bias,
    )

    # Cast to target dtype
    block = block.to(dtype)

    return block


def replace_module(
    model: nn.Module,
    target_name: str,
    new_module: nn.Module,
) -> None:
    """Replace a named module in the model in-place.

    Navigates the module tree using the dotted name path and replaces
    the target module with new_module.

    Args:
        model: The model to modify.
        target_name: Fully qualified dotted name (e.g., "blocks.0.attention").
        new_module: The replacement module.

    Raises:
        ValueError: If the target module is not found.
    """
    parts = target_name.split(".")
    parent = model

    # Navigate to the parent
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        elif hasattr(parent, part):
            parent = getattr(parent, part)
        else:
            raise ValueError(f"Module path '{target_name}' not found at '{part}'")

    # Replace the target
    final_name = parts[-1]
    if final_name.isdigit():
        parent[int(final_name)] = new_module
    elif hasattr(parent, final_name):
        setattr(parent, final_name, new_module)
    else:
        raise ValueError(
            f"Module '{final_name}' not found in {type(parent).__name__}"
        )

    logger.info("Replaced module: %s", target_name)


def select_blocks_to_replace(
    attention_modules: List[AttentionInfo],
    config: SurgeryConfig,
) -> List[AttentionInfo]:
    """Select which attention blocks to replace based on strategy.

    Args:
        attention_modules: All discovered attention modules.
        config: Surgery configuration.

    Returns:
        List of AttentionInfo for modules to be replaced.
    """
    if config.block_indices is not None:
        # Explicit indices override strategy
        return [attention_modules[i] for i in config.block_indices
                if i < len(attention_modules)]

    if config.strategy == "all":
        return list(attention_modules)

    elif config.strategy == "progressive":
        n = config.num_replace or 1
        return attention_modules[:n]

    elif config.strategy == "alternating":
        return [m for i, m in enumerate(attention_modules) if i % 2 == 0]

    elif config.strategy == "by-cost":
        return _select_by_cost(attention_modules, config)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


def _select_by_cost(
    attention_modules: List[AttentionInfo],
    config: SurgeryConfig,
) -> List[AttentionInfo]:
    """Select blocks by compute cost using Phase 1 profiling data.

    Args:
        attention_modules: All attention modules.
        config: Surgery config with profiling_data path.

    Returns:
        Modules sorted by cost (most expensive first), up to num_replace.
    """
    if config.profiling_data is None:
        logger.warning("No profiling data. Falling back to parameter count ordering.")
        sorted_modules = sorted(attention_modules, key=lambda m: -m.param_count)
        n = config.num_replace or len(sorted_modules)
        return sorted_modules[:n]

    with open(config.profiling_data, "r") as f:
        profiling = json.load(f)

    # Build time lookup from profiling data
    time_lookup = {}
    for entry in profiling.get("per_module", []):
        time_lookup[entry["name"]] = entry.get("avg_time_ms", 0)

    # Sort attention modules by profiled time (descending)
    def get_time(m: AttentionInfo) -> float:
        return time_lookup.get(m.name, 0.0)

    sorted_modules = sorted(attention_modules, key=get_time, reverse=True)
    n = config.num_replace or len(sorted_modules)
    return sorted_modules[:n]


def perform_surgery(
    config: SurgeryConfig,
) -> Tuple[nn.Module, SurgeryReport]:
    """Execute architecture surgery on the model.

    Loads the model, identifies attention blocks, replaces selected blocks
    with Mamba SSM blocks, and returns the modified model.

    Args:
        config: Surgery configuration.

    Returns:
        Tuple of (modified_model, surgery_report).
    """
    import sys
    sys.path.insert(0, ".")

    # Load model
    logger.info("Loading model '%s'...", config.model_name)

    try:
        from diffusers import WanTransformer3DModel

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.float16)

        # Wan is a diffusers model; operate on the diffusion transformer (DiT).
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "subfolder": "transformer",
        }
        if config.low_memory:
            load_kwargs["low_cpu_mem_usage"] = True

        if config.model_path:
            model = WanTransformer3DModel.from_pretrained(config.model_path, **load_kwargs)
        else:
            from wan_profiler.loader import get_model_id
            model_id = get_model_id(config.model_name)
            model = WanTransformer3DModel.from_pretrained(model_id, **load_kwargs)

    except ImportError:
        logger.error(
            "Could not import model loading utilities. "
            "Install wan-profiler or provide --model-path."
        )
        raise

    model.eval()
    original_params = sum(p.numel() for p in model.parameters())

    # Find attention modules
    attention_modules = find_attention_modules(model)
    if not attention_modules:
        logger.warning("No attention modules found. Nothing to replace.")
        return model, SurgeryReport(
            model_name=config.model_name,
            strategy=config.strategy,
            total_attention_blocks=0,
            replaced_blocks=0,
            replaced_names=[],
            kept_names=[],
            mamba_config={},
            original_params=original_params,
            modified_params=original_params,
        )

    # Select blocks to replace
    to_replace = select_blocks_to_replace(attention_modules, config)
    to_keep = [m for m in attention_modules if m not in to_replace]

    logger.info(
        "Surgery plan: replace %d of %d attention blocks (strategy: %s)",
        len(to_replace), len(attention_modules), config.strategy,
    )

    # Perform replacement
    for info in to_replace:
        if info.d_model == 0:
            logger.warning("Skipping %s: could not determine d_model", info.name)
            continue

        mamba_block = create_mamba_replacement(
            d_model=info.d_model,
            mamba_config=config.mamba,
            dtype=torch.float16 if config.dtype == "float16" else torch.float32,
        )

        replace_module(model, info.name, mamba_block)
        gc.collect()

    modified_params = sum(p.numel() for p in model.parameters())

    # Build report
    report = SurgeryReport(
        model_name=config.model_name,
        strategy=config.strategy,
        total_attention_blocks=len(attention_modules),
        replaced_blocks=len(to_replace),
        replaced_names=[m.name for m in to_replace],
        kept_names=[m.name for m in to_keep],
        mamba_config={
            "d_state": config.mamba.d_state,
            "d_conv": config.mamba.d_conv,
            "expand": config.mamba.expand,
        },
        original_params=original_params,
        modified_params=modified_params,
    )

    logger.info(
        "Surgery complete. Params: %s → %s (%.1f%% change)",
        f"{original_params:,}",
        f"{modified_params:,}",
        (modified_params - original_params) / original_params * 100,
    )

    return model, report
