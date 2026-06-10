#!/usr/bin/env python
"""Self- vs cross-attention scaling — which one actually needs replacing?

Self-attention is O(n^2) in image tokens; cross-attention is O(n*m) with a fixed
text length m, i.e. only *linear* in image tokens. This times both as the token
count grows, to decide the architecture: replace the quadratic part (self-attn)
with Mamba, keep the linear part (cross-attn). No model download (cached DiT).

Usage:
    HF_TOKEN=... python scripts/attn_scaling.py
"""
import time
import gc
from collections import defaultdict

import psutil
import torch

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
from diffusers import WanTransformer3DModel  # noqa: E402


def main() -> None:
    model = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    ts = torch.tensor([500])

    acc = defaultdict(float)
    st = {}
    for name, mod in model.named_modules():
        if name.endswith(".attn1") or name.endswith(".attn2"):
            tag = "self" if name.endswith("attn1") else "cross"
            mod.register_forward_pre_hook(
                lambda m, i, k=id(mod): st.__setitem__(k, time.perf_counter()))
            mod.register_forward_hook(
                lambda m, i, o, k=id(mod), t=tag: acc.__setitem__(
                    t, acc[t] + time.perf_counter() - st[k]))

    print(f"{'tokens':>7}{'self-attn(s)':>14}{'cross-attn(s)':>15}")
    rows = []
    for f in [1, 2, 3]:
        hs = torch.randn(1, 16, f, 32, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            model(hs, ts, txt, return_dict=False)
            acc["self"] = acc["cross"] = 0.0
            model(hs, ts, txt, return_dict=False)
        print(f"{f*256:>7}{acc['self']:>14.2f}{acc['cross']:>15.2f}")
        rows.append((f * 256, acc["self"], acc["cross"]))
        del hs
        gc.collect()

    (t1, s1, c1), (t3, s3, c3) = rows[0], rows[-1]
    print(f"\ntokens x{t3//t1}: self-attn x{s3/s1:.1f} (->quadratic)  "
          f"cross-attn x{c3/c1:.1f} (->linear)")


if __name__ == "__main__":
    main()
