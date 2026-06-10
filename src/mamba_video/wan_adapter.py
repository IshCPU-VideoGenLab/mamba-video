"""Drop-in Mamba replacement for Wan self-attention.

``WanTransformerBlock`` calls ``self.attn1(hidden_states, None, None, rotary_emb)``
and expects a same-shaped output that it then gates and adds to the residual
stream. This adapter wraps a :class:`MambaBlock` to match that exact call
signature, returning only the transform *delta* — it cancels MambaBlock's own
internal residual so the Wan block's external residual + gate is applied exactly
once. Extra attention args (encoder_hidden_states, attention_mask, rotary_emb)
are accepted and ignored (Mamba carries position via its conv + recurrence).
"""
from typing import Optional

import torch
import torch.nn as nn

from mamba_video.mamba_block import MambaBlock


class WanMambaSelfAttention(nn.Module):
    """Mamba block exposing the WanAttention self-attention call signature."""

    def __init__(self, dim: int, d_state: int = 8, expand: int = 1,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.mamba = MambaBlock(d_model=dim, d_state=d_state, expand=expand).to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Wan applies norm/residual/gate outside attn1, so return only the delta.
        return self.mamba(hidden_states) - hidden_states


def replace_self_attention(model: nn.Module, num_blocks: Optional[int] = None,
                           d_state: int = 8, expand: int = 1) -> int:
    """Replace the first ``num_blocks`` transformer blocks' self-attention
    (``attn1``) with an untrained Mamba adapter. Returns how many were replaced.

    NOTE: the Mamba blocks are randomly initialized — this is the surgery step;
    quality recovery requires fine-tuning (Phase 6). Cross-attention (``attn2``)
    is left intact because it needs the text-conditioning path a plain SSM lacks.
    """
    blocks = model.blocks
    n = len(blocks) if num_blocks is None else min(num_blocks, len(blocks))
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    for i in range(n):
        blocks[i].attn1 = WanMambaSelfAttention(dim, d_state=d_state, expand=expand)
    return n
