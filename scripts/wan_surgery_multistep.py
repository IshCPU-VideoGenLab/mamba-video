#!/usr/bin/env python
"""Phase 2 experiment: multi-step denoise drift under Mamba surgery.

Single-step output drift (see wan_surgery_experiment.py) does not capture how
error compounds across the denoising trajectory. This runs a short denoise loop
with the original model and with the first N self-attention blocks replaced by
(untrained) Mamba, and reports per-step cosine similarity of the latent vs. the
original trajectory.

The loop is a simple Euler integration of the predicted velocity. It is not the
exact Wan scheduler, but it is applied identically to both models, so the
relative drift it reports is meaningful. No new download (cached DiT; dummy text
embedding stands in for the ~11 GB T5).

Usage:
    HF_TOKEN=... python scripts/wan_surgery_multistep.py
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

STEPS = 6


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    nb = len(model.blocks)
    print(f"dim={dim}, blocks={nb}")

    x_init = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    @torch.no_grad()
    def denoise():
        x = x_init.clone()
        traj = []
        for i in range(STEPS):
            t = torch.tensor([int((1 - i / STEPS) * 999)])
            v = model(x, t, txt, return_dict=False)[0]
            x = x - (1.0 / STEPS) * v
            traj.append(x.float().clone())
        return traj

    print("running ORIGINAL denoise ...")
    orig = denoise()

    for n in [8, nb]:
        for i in range(n):
            if not isinstance(model.blocks[i].attn1, WanMambaSelfAttention):
                model.blocks[i].attn1 = WanMambaSelfAttention(dim)
        t0 = time.perf_counter()
        mod = denoise()
        dt = time.perf_counter() - t0
        cs = [F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
              for a, b in zip(mod, orig)]
        print(f"\nN={n} self-attn replaced ({dt:.0f}s): per-step cosine vs original")
        print("  " + "  ".join(f"s{i + 1}:{c:.3f}" for i, c in enumerate(cs)))
        print(f"  final-latent cosine: {cs[-1]:.4f}")


if __name__ == "__main__":
    main()
