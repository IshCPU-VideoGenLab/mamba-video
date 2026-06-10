#!/usr/bin/env python
"""Phase 2 experiment: cross-attention surgery (text-conditioned SSM) drift.

Self-attention replaces gracefully (see wan_surgery_multistep.py). Cross-attention
is harder: it conditions the image tokens on the text embeddings. This tests a
FiLM-conditioned Mamba (WanMambaCrossAttention: pool text -> scale/shift ->
modulate -> scan) as a drop-in for attn2, measuring multi-step denoise drift for
(a) cross-attention only and (b) the full attention stack (self + cross).

Requires simd-kernels installed (for the fast scan). No model download (cached
DiT; dummy text embedding). All adapters untrained -- measures viability/drift.

Usage:
    HF_TOKEN=... python scripts/wan_cross_surgery.py
"""
import sys
import types

import psutil
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from mamba_video.wan_adapter import WanMambaSelfAttention, WanMambaCrossAttention  # noqa: E402

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
from diffusers import WanTransformer3DModel  # noqa: E402

STEPS = 4


def main() -> None:
    from simd_kernels.ssm_scan import simd_ssm_scan
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    x_init = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    def simd_scan(self, x, A, B, C):
        return simd_ssm_scan(x[0], A[0], B[0], C[0]).unsqueeze(0).to(x.dtype)

    def patch():
        for b in model.blocks:
            for a in (b.attn1, b.attn2):
                if hasattr(a, "mamba"):
                    a.mamba._selective_scan = types.MethodType(simd_scan, a.mamba)

    @torch.no_grad()
    def denoise():
        x = x_init.clone()
        tr = []
        for i in range(STEPS):
            t = torch.tensor([int((1 - i / STEPS) * 999)])
            v = model(x, t, txt, return_dict=False)[0]
            x = x - (1 / STEPS) * v
            tr.append(x.float().clone())
        return tr

    def cs(tr, o):
        return [F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item() for a, b in zip(tr, o)]

    orig = denoise()
    for b in model.blocks:
        b.attn2 = WanMambaCrossAttention(dim)
    patch()
    print("cross-attn -> FiLM-Mamba: " + " ".join(f"{c:.3f}" for c in cs(denoise(), orig)))
    for b in model.blocks:
        b.attn1 = WanMambaSelfAttention(dim)
    patch()
    print("FULL (self+cross) -> Mamba: " + " ".join(f"{c:.3f}" for c in cs(denoise(), orig)))


if __name__ == "__main__":
    main()
