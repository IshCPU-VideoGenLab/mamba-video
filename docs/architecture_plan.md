# Architecture Surgery Plan

## Objective

Replace transformer self-attention blocks in Wan 1.3B with Mamba SSM blocks
to eliminate the O(n²) bottleneck that makes video generation GPU-dependent.

## Why Attention Is the Target

From Phase 1 profiling (wan-profiler), self-attention is expected to consume
the largest fraction of compute time in the transformer backbone. Attention
requires O(n²) memory and compute in the sequence length, while the
sequence length for video (frames × spatial patches) is enormous.

Replacing attention with an O(n) sequential model (Mamba/SSM) directly
addresses the core scalability bottleneck.

## The Mamba SSM Block

### Architecture

```
Input x (batch, seq, d_model)
    │
    ├─── Linear → 2 * d_inner ──── split ────┐
    │                                         │
    x_branch                                  z (gate)
    │                                         │
    Conv1D (depthwise, kernel=4)              │
    │                                         │
    SiLU activation                           │
    │                                         │
    SSM parameter projection                  │
    ├── B (input matrix)                      │
    ├── C (output matrix)                     │
    └── Δ (discretization step)               │
    │                                         │
    Discretize: A, B → Ā, B̄                  │
    │                                         │
    Selective Scan (recurrent):               │
    │  h[t] = Ā[t] · h[t-1] + B̄[t] · x[t]  │
    │  y[t] = C[t] · h[t]                    │
    │                                         │
    + D · x (skip connection)                 │
    │                                         │
    × SiLU(z) ────────────────────────────────┘
    │
    Linear → d_model
    │
    + residual
    │
    Output y (batch, seq, d_model)
```

### Key Design Choices

1. **Pure PyTorch scan**: The selective scan is a Python for-loop. This is
   slow but correct, portable, and runs on any CPU. Phase 5 replaces this
   with portable SIMD kernels (AVX2 on x86, NEON on ARM).

2. **d_state = 16**: The SSM state dimension controls how much "memory"
   each position has of the past. 16 is a good balance between expressiveness
   and memory cost on 16GB machines.

3. **expand = 2**: The inner dimension is 2× the model dimension. This
   matches the capacity of a typical attention block's QKV projections.

4. **Discretization in float32**: The exp() and division operations in
   discretization are numerically sensitive. We compute in float32 and
   cast back to float16 for the scan.

## Surgery Strategies

### Strategy: `all`
Replace every attention block. Maximum speed gain, maximum quality risk.

### Strategy: `progressive`
Replace blocks one at a time, starting from the first (shallowest).
Allows finding the quality-speed Pareto frontier.

### Strategy: `by-cost`
Replace the most compute-expensive blocks first (using Phase 1 data).
Targets maximum speedup per quality point lost.

### Strategy: `alternating`
Replace every other block. Keeps some attention capacity distributed
throughout the network depth.

## Weight Initialization

When replacing a trained attention block with fresh Mamba weights, the
initialization determines whether the model produces garbage or merely
degraded output.

### Random (baseline)
Standard Xavier initialization. The Mamba block produces semi-random
output. The residual connection keeps the model barely functional.

### Scaled
Output projection initialized with scale=0.01. The Mamba block initially
contributes almost nothing to the residual stream. The model behaves
almost identically to its original until the SSM parameters are trained.

### Identity
All projections minimize their contribution. D=1 (strong skip connection).
A initialized for slow state decay. The block approximates an identity
function: output ≈ input + ε.

**Expected best choice: scaled or identity.** Both preserve model behavior
initially, but scaled is simpler and works well in practice.

## Evaluation Plan

### Speed Metrics
- Wall-clock time per forward pass (ms)
- Time per replaced block vs original attention block (ms)
- Total speedup ratio

### Memory Metrics
- Peak RSS during inference (MB)
- Memory savings from O(n²) → O(n) attention removal

### Quality Metrics
- MSE between original and modified model outputs (latent space)
- PSNR (if evaluating in pixel space)
- Cosine similarity in latent space
- LPIPS (perceptual quality, if torchmetrics available)
- FID (on a set of generated frames, if feasible on target hardware)

### The Key Chart: Quality vs Blocks Replaced
For N = 1, 2, 4, 8, ..., all attention blocks:

```
Quality
  │ ●──●──●──●──●
  │                 ●
  │                     ●
  │                         ●
  │                              ●
  └──────────────────────────────── Blocks Replaced
```

This chart identifies the "sweet spot" — maximum blocks replaceable
before quality collapses. This is the central result of Phase 2.

## Handoff to Phase 3

Phase 2 produces:
1. A modified Wan 1.3B with attention → Mamba replacements
2. The quality-speed Pareto curve
3. Knowledge of which blocks are safe to replace

Phase 3 (codec-video-gen) takes this modified architecture and adds
temporal compression (I-frame + P-frame design), further reducing
compute by exploiting inter-frame redundancy.
