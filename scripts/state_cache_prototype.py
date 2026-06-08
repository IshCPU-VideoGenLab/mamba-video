#!/usr/bin/env python
"""Prototype: Diffusion-Trajectory SSM State Caching.

Novel idea: because the Mamba SSM is a *stateful* linear recurrence
(h[t] = A_bar[t]*h[t-1] + B_bar[t]*x[t], with |A_bar| < 1 so state decays),
the hidden state computed at one diffusion step is an almost-correct
initialization for the next step — consecutive denoising steps barely change
the latent. So across the denoising trajectory we can REUSE cached SSM state
for tokens whose input didn't change, and only re-scan changed tokens plus
their (bounded, self-terminating) decay tails. A transformer cannot do this
(it is stateless); this is unlocked precisely by choosing Mamba.

Runs on the real MambaBlock with a SYNTHETIC denoising trajectory — no model
download required. Measures scan-work saved vs. output error as a tunable
knob, with a periodic full refresh to bound drift.

Usage:
    python scripts/state_cache_prototype.py

See docs/state_caching.md for results and analysis.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mamba_video.mamba_block import MambaBlock  # noqa: E402

torch.manual_seed(0)
B, L, D_MODEL, K = 1, 256, 256, 24  # batch, tokens, dim, denoising steps
block = MambaBlock(d_model=D_MODEL, d_state=16).eval()
DI, DS = block.d_inner, block.d_state


@torch.no_grad()
def front_end(x):
    """Replicate MambaBlock.forward up to the selective-scan inputs."""
    xz = block.in_proj(block.norm(x))
    xb, _ = xz.chunk(2, dim=-1)
    xc = F.silu(block.conv1d(xb.transpose(1, 2))[:, :, :L].transpose(1, 2))
    xpo = block.x_proj(xc)
    Bm, Cm = xpo[:, :, :DS], xpo[:, :, DS:2 * DS]
    delta = F.softplus(block.dt_proj(xpo[:, :, -1:])).clamp(block.dt_min, block.dt_max)
    A_bar, B_bar = block._discretize(-torch.exp(block.A_log), Bm, delta)
    return xc, A_bar, B_bar, Cm


@torch.no_grad()
def full_scan(A_bar, B_bar, C, xc):
    h = torch.zeros(B, DI, DS)
    H = torch.empty(L, B, DI, DS)
    Y = torch.empty(B, L, DI)
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t] * xc[:, t].unsqueeze(-1)
        Y[:, t] = torch.einsum("bn,bdn->bd", C[:, t], h)
        H[t] = h
    return Y, H


@torch.no_grad()
def cached_scan(A_bar, B_bar, C, xc, dxc, H_cache, Y_cache, q, h_rel):
    """Reuse cached state where input unchanged AND carried state matches cache."""
    thr = torch.quantile(dxc, q).item()  # recompute the top (1-q) most-changed tokens
    h = torch.zeros(B, DI, DS)
    H = torch.empty(L, B, DI, DS)
    Y = torch.empty(B, L, DI)
    work = 0
    for t in range(L):
        cin = H_cache[t - 1] if t > 0 else torch.zeros(B, DI, DS)
        changed = dxc[t].item() > thr
        diverged = (h - cin).abs().max().item() > h_rel * (cin.abs().max().item() + 1e-6)
        if not changed and not diverged:
            h = H_cache[t]
            Y[:, t] = Y_cache[:, t]  # REUSE — no compute
        else:
            h = A_bar[:, t] * h + B_bar[:, t] * xc[:, t].unsqueeze(-1)
            Y[:, t] = torch.einsum("bn,bdn->bd", C[:, t], h)
            work += 1
        H[t] = h
    return Y, H, work


def build_trajectory():
    """Synthetic denoising trajectory: 30% 'dynamic' tokens (motion), 70% static
    background; per-step change magnitude decays as denoising proceeds."""
    dynamic = (torch.rand(L) < 0.3).float().view(1, L, 1)
    x = torch.randn(B, L, D_MODEL)
    traj = [x.clone()]
    for k in range(K):
        scale = 0.6 * (1 - k / K)
        per_tok = dynamic * 1.0 + (1 - dynamic) * 0.12
        x = x + scale * per_tok * torch.randn(B, L, D_MODEL)
        traj.append(x.clone())
    return traj


def run(traj, q, h_rel, refresh_every):
    xc_prev, A0, Bb0, C0 = front_end(traj[0])
    Ycache, Hc = full_scan(A0, Bb0, C0, xc_prev)
    tot_work = tot_full = 0
    errs = []
    for k in range(1, len(traj)):
        xc, A_bar, B_bar, C = front_end(traj[k])
        Yf, Hf = full_scan(A_bar, B_bar, C, xc)
        if k % refresh_every == 0:  # periodic full refresh bounds drift
            Yc, Hcn, wc = Yf, Hf, L
        else:
            dxc = (xc - xc_prev).norm(dim=-1).flatten()
            Yc, Hcn, wc = cached_scan(A_bar, B_bar, C, xc, dxc, Hc, Ycache, q, h_rel)
        errs.append((Yc - Yf).norm().item() / (Yf.norm().item() + 1e-9))
        tot_work += wc
        tot_full += L
        xc_prev, Hc, Ycache = xc, Hcn, Yc
    return 1 - tot_work / tot_full, sum(errs) / len(errs), max(errs)


def main():
    traj = build_trajectory()
    print(f"MambaBlock d_model={D_MODEL} d_inner={DI} d_state={DS} | "
          f"seq_len={L} | {K} denoising steps")
    print("trajectory: 30% dynamic tokens, 70% static; change magnitude decays\n")
    print(f"{'q':>5}{'h_rel':>7}{'refresh':>8}{'work saved':>12}{'mean err':>10}{'max err':>9}")
    for q, h_rel, rf in [(0.3, 1.0, 8), (0.5, 1.0, 4), (0.5, 0.5, 6), (0.7, 0.5, 6)]:
        s, me, mx = run(traj, q, h_rel, rf)
        print(f"{q:>5}{h_rel:>7}{rf:>8}{s*100:>10.1f}%{me*100:>9.2f}%{mx*100:>8.2f}%")


if __name__ == "__main__":
    main()
