# Diffusion-Trajectory SSM State Caching (exploratory)

A novel efficiency mechanism that falls out of choosing Mamba over attention.
Status: **prototype validated on synthetic data; not yet validated on the real DiT.**

## The idea

Video diffusion cost = **(denoising steps T) × (frames F) × (cost of one forward)**.
The project's four pillars (Mamba, codec, BitNet, SIMD) all reduce the *forward cost*
or the *frame count*. **None of them touch the `T` axis** — yet every denoising step
re-runs the whole network from scratch on a latent that barely changed.

A transformer *must* recompute (it is stateless). But the Mamba SSM is a **stateful
linear recurrence**:

```
h[t] = Ā[t] · h[t-1] + B̄[t] · x[t]      with  Ā = exp(Δ·A),  A < 0  ⇒  |Ā| < 1
```

Two consequences:

1. **State carries over.** The hidden state computed at diffusion step *k* is an
   almost-correct initialization for step *k+1* (the latent changed ~few %).
2. **State decays geometrically** (`|Ā| < 1`). A change at token *t* perturbs the
   state for only a **bounded, self-terminating forward window** before it dies out.

So across the denoising trajectory we **reuse cached SSM state** for tokens whose
input didn't change, and only re-scan the changed tokens plus their decay tails.
This is unlocked *precisely because the model is a stateful SSM* — it cannot be done
with attention, and it is finer-grained than block-output caches (e.g. DeepCache).

## Prototype results

`scripts/state_cache_prototype.py` runs the **real `MambaBlock`** over a synthetic
denoising trajectory (256 tokens, 24 steps, 30% "dynamic"/motion tokens + 70% static
background, per-step change magnitude decaying as denoising proceeds). It reuses
cached state only when a token's input is unchanged **and** the carried state matches
the cache, with a periodic full refresh to bound drift.

| setting | scan work saved | mean rel-err | max rel-err |
|---|---:|---:|---:|
| conservative | **26.3%** | 2.28% | 4.98% |
| moderate | **37.5%** | 2.76% | 8.68% |
| moderate+ | **41.7%** | 3.54% | 8.68% |
| aggressive | **58.3%** | 5.16% | 14.04% |

**Takeaway:** the SSM hidden state *is* reusable across diffusion steps — a tunable
**1.3–2.4× reduction** of the scan (the most expensive loop), stacking
multiplicatively with the other pillars.

## Honest caveats

- The win is **moderate (1.3–2.4×), not dramatic** — decay-tail recomputes cost real work.
- The synthetic trajectory **perturbs every token each step**, so it is *conservative*;
  real late-stage denoising and real video static backgrounds likely have more
  redundancy. Treat these numbers as a rough lower bound.
- It is an **approximation**; quality must be validated end-to-end (use `quality.py`).
- **Not yet tested on the real Wan DiT** (deferred — requires the ~2.8 GB DiT download).

## Next steps

1. Validate on the real `WanTransformer3DModel` DiT with a real denoising schedule.
2. Make the reuse schedule adaptive (more reuse in late, low-noise steps).
3. Combine with codec P-frames (P-frames start from the I-frame's converged state).
4. Measure end-to-end quality (FID/SSIM) at fixed compute budgets.
