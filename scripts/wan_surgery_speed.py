#!/usr/bin/env python
"""Phase 2 experiment: attention vs Mamba speed, and the O(n) / O(n^2) crossover.

Times the self-attention path (original) against the Mamba-surgery path at
increasing latent token counts, with the Mamba scan running both as the
pure-Python loop and via the native NEON SSM kernel from simd-kernels (Phase 5).

Requires simd-kernels installed (`pip install -e ../simd-kernels`) for the SIMD
scan; falls back to Python otherwise. No model download (cached DiT; dummy text
embedding). All Mamba blocks are untrained — this measures *speed*, not quality.

Usage:
    HF_TOKEN=... python scripts/wan_surgery_speed.py
"""
import sys
import time
import types

import psutil
import torch

sys.path.insert(0, "src")
from mamba_video.wan_adapter import WanMambaSelfAttention  # noqa: E402

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
from diffusers import WanTransformer3DModel  # noqa: E402

FRAMES = [1, 2, 3]  # -> 256, 512, 768 latent tokens


def _simd_scan(self, x, A_bar, B_bar, C):
    """Drop-in for MambaBlock._selective_scan using the native NEON kernel."""
    from simd_kernels.ssm_scan import simd_ssm_scan
    y = simd_ssm_scan(x[0], A_bar[0], B_bar[0], C[0])
    return y.unsqueeze(0).to(x.dtype)


def _time_tagged(model, predicate, hs, ts, txt):
    acc = {"t": 0.0}
    st = {}
    handles = []
    for _, mod in model.named_modules():
        if predicate(mod):
            handles.append(mod.register_forward_pre_hook(
                lambda m, i: st.update(s=time.perf_counter())))
            handles.append(mod.register_forward_hook(
                lambda m, i, o: acc.update(t=acc["t"] + time.perf_counter() - st["s"])))
    with torch.no_grad():
        model(hs, ts, txt, return_dict=False)
        acc["t"] = 0.0
        model(hs, ts, txt, return_dict=False)
    for h in handles:
        h.remove()
    return acc["t"]


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    dim = model.config.num_attention_heads * model.config.attention_head_dim
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    ts = torch.tensor([500])

    attn = {}
    for f in FRAMES:
        hs = torch.randn(1, 16, f, 32, 32, dtype=torch.bfloat16)
        attn[f * 256] = _time_tagged(model, lambda m: type(m).__name__ == "WanAttention"
                                     and not isinstance(m, WanMambaSelfAttention), hs, ts, txt)

    for i in range(len(model.blocks)):
        model.blocks[i].attn1 = WanMambaSelfAttention(dim)
    for b in model.blocks:
        b.attn1.mamba._selective_scan = types.MethodType(_simd_scan, b.attn1.mamba)

    print(f"{'tokens':>7}{'attention(s)':>14}{'mamba+SIMD(s)':>15}{'winner':>9}")
    for f in FRAMES:
        hs = torch.randn(1, 16, f, 32, 32, dtype=torch.bfloat16)
        m = _time_tagged(model, lambda mod: isinstance(mod, WanMambaSelfAttention), hs, ts, txt)
        a = attn[f * 256]
        print(f"{f*256:>7}{a:>14.2f}{m:>15.2f}{('Mamba' if m < a else 'attn'):>9}")


if __name__ == "__main__":
    main()
