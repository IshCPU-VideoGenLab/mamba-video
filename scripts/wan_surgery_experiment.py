#!/usr/bin/env python
"""Phase 2 experiment: progressive Mamba surgery on Wan self-attention.

Replaces the first N transformer blocks' self-attention with (untrained) Mamba
adapters and measures output drift vs. the original model + forward time. This
is the "quality vs. blocks replaced" curve (the paper's Figure 2), measured at
the latent-output level on the real Wan 1.3B DiT. No new download (uses the
cached DiT); the ~11 GB T5 is replaced by a dummy embedding.

Usage:
    HF_TOKEN=... python scripts/wan_surgery_experiment.py
"""
import sys
import time

import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from mamba_video.wan_adapter import WanMambaSelfAttention  # noqa: E402

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
from diffusers import WanTransformer3DModel  # noqa: E402


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    print(f"DiT hidden dim={dim}, blocks={len(model.blocks)}")

    hs = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    ts = torch.tensor([500])
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    @torch.no_grad()
    def fwd():
        t0 = time.perf_counter()
        out = model(hs, ts, txt, return_dict=False)[0]
        return out.float(), time.perf_counter() - t0

    y0, t0 = fwd()
    print(f"baseline (0 replaced): {t0:.1f}s")
    print(f"{'replaced':>9}{'cosine':>10}{'rel err':>10}{'fwd s':>8}")
    for n in [1, 2, 4, 8]:
        for i in range(n):
            if not isinstance(model.blocks[i].attn1, WanMambaSelfAttention):
                model.blocks[i].attn1 = WanMambaSelfAttention(dim)
        y, t = fwd()
        cos = F.cosine_similarity(y.flatten(), y0.flatten(), dim=0).item()
        rel = (y - y0).norm().item() / (y0.norm().item() + 1e-9)
        print(f"{n:>9}{cos:>10.4f}{rel:>10.3f}{t:>8.1f}")


if __name__ == "__main__":
    main()
