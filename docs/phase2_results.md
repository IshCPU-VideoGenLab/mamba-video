# Phase 2 Results — Mamba Surgery on Wan Self-Attention (first pass)

Replacing Wan's self-attention (`attn1`) with Mamba SSM blocks, measured on the
real Wan 1.3B DiT (CPU, bfloat16). Status: **surgery works mechanically; output
drift from untrained Mamba is small and graceful. Quality recovery (fine-tuning)
and the speed win (Phase 5 kernels + larger token counts) are future work.**

## How the surgery attaches

`WanTransformerBlock` calls `self.attn1(hidden_states, None, None, rotary_emb)`
and adds the result to the residual stream with a learned gate:

```python
attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
hidden_states = hidden_states + attn_output * gate_msa
```

`WanMambaSelfAttention` (`src/mamba_video/wan_adapter.py`) matches that signature
and returns a same-shaped **delta** (it cancels MambaBlock's own residual so the
Wan block's external residual/gate applies once). Cross-attention (`attn2`) is
left intact — it needs the text-conditioning path a plain SSM lacks (see Phase 1
Table 1: cross-attention is 36.8% of compute).

## Progressive replacement — output drift vs. original

Real Wan 1.3B DiT: hidden dim 1536, **30 blocks**, 256 latent tokens.

| self-attn blocks replaced | cosine vs. original | rel. error | forward (s) |
|--------------------------:|--------------------:|-----------:|------------:|
| 1 | 0.9880 | 0.159 | 5.4 |
| 2 | 0.9863 | 0.165 | 5.5 |
| 4 | 0.9863 | 0.165 | 5.7 |
| 8 | 0.9856 | 0.170 | 5.9 |

**Finding:** even with 8/30 self-attention layers replaced by *untrained* Mamba,
the output stays at cosine ~0.986 and degrades gracefully. The residual stream
plus per-block gating absorbs the perturbation — the network is far more robust
to this surgery than a naive "untrained = garbage" expectation. This is
encouraging for a fine-tune-to-recover strategy.

## Honest caveats

- This is **single-step latent output drift**, not video quality. Errors compound
  over the 20–50 denoising steps of real generation; needs end-to-end evaluation.
- **No speedup yet.** The pure-Python Mamba scan is slow, and self-attention is
  cheap at 256 tokens. The win requires the Phase 5 SIMD scan kernels *and* the
  larger token counts where self-attention is O(n²) (Phase 1 scaling result).
- Mamba blocks are **untrained** (random init). The small drift is promising but
  quality recovery still needs fine-tuning (Phase 6).
- Tested to 8/30 blocks (memory). Full-depth + multi-step evaluation is next.

## Next

1. Extend to all 30 blocks; evaluate over a full multi-step denoise.
2. Measure speed at larger token counts (where O(n²) attention dominates) with
   the Phase 5 SIMD scan wired in.
3. Design a text-conditioned SSM for cross-attention (the 36.8% Table 1 found).
4. Fine-tune the Mamba blocks (Phase 6 ES) to recover quality.
