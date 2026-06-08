# Phase 2 — mamba-video Task Roadmap

> Claude Code: check this file at the start of every session.
> Mark tasks `[x]` when complete.

---

## Milestone 1: Mamba Block Implementation
- [ ] Implement `MambaBlock` in pure PyTorch (no CUDA kernels)
- [ ] Implement selective scan (sequential loop version)
- [ ] Implement discretization of continuous SSM parameters (A, B → Ā, B̄)
- [ ] Implement input-dependent Δ (selectivity mechanism)
- [ ] Implement local Conv1D preprocessing
- [ ] Implement gated output (SiLU activation + multiply)
- [ ] Unit test: forward pass shape correctness
- [ ] Unit test: gradient flow (backward pass works)
- [ ] Unit test: numerical stability with float16
- [ ] Benchmark: single block forward time on CPU

## Milestone 2: Model Inspection & Surgery Engine
- [ ] Load Wan 1.3B and enumerate all attention modules
- [ ] Map attention module names and their input/output dimensions
- [ ] Implement `replace_module_by_name()` — swap a named module in-place
- [ ] Implement `create_matching_mamba()` — create a MambaBlock matching attention dims
- [ ] Implement surgery strategies: `all`, `progressive`, `by-cost`, `alternating`
- [ ] Test: surgery on a small dummy transformer model
- [ ] Test: surgery on Wan 1.3B produces a model that can forward pass
- [ ] Save modified model state dict (only changed layers + metadata)

## Milestone 3: Weight Initialization
- [ ] Research initialization strategies for SSM blocks replacing trained attention
- [ ] Implement random initialization (baseline)
- [ ] Implement scaled initialization (match attention output variance)
- [ ] Implement identity-like initialization (SSM starts as near-passthrough)
- [ ] Compare initialization strategies on forward pass output distribution
- [ ] Document which initialization works best

## Milestone 4: Benchmarking
- [ ] Implement `benchmark.py` — timed forward passes for original and modified
- [ ] Measure wall-clock time per forward pass (original vs modified)
- [ ] Measure peak memory (original vs modified)
- [ ] Measure time per replaced block vs original attention block
- [ ] Compute speedup ratio and memory savings
- [ ] Generate comparison charts

## Milestone 5: Quality Evaluation
- [ ] Implement FID metric computation (generated frames vs reference)
- [ ] Implement LPIPS metric (perceptual similarity)
- [ ] Implement FVD metric if feasible (Fréchet Video Distance)
- [ ] Generate frames from original model (reference set)
- [ ] Generate frames from modified model (test set)
- [ ] Compute quality metrics for each surgery strategy
- [ ] Plot quality vs speedup Pareto curve
- [ ] Identify the sweet spot: maximum blocks replaced before quality collapses

## Milestone 6: Progressive Replacement Analysis
- [ ] Run surgery with N=1, 2, 4, 8, ... replaced blocks
- [ ] For each N: measure time, memory, and quality
- [ ] Produce the key chart: "Quality vs Number of Blocks Replaced"
- [ ] Determine: which specific blocks are safe to replace?
- [ ] Determine: which blocks must remain as attention?
- [ ] Document findings for Phase 3 handoff

## Milestone 7: Documentation & Polish
- [ ] Write `docs/architecture_plan.md` with full methodology
- [ ] Update README with actual results
- [ ] Clean up code, docstrings, type hints
- [ ] Ensure all tests pass
- [ ] Final commit and tag v0.1.0
- [ ] Write findings summary for Phase 3 (codec-video-gen) handoff

---

## Notes
- Milestones 1–2 are the critical path. Get a working Mamba block and surgery engine first.
- Quality evaluation (Milestone 5) may need simplification if generating full video frames is too slow within the commodity-hardware budget. Consider evaluating on single-frame generation or latent-space metrics.
- The progressive replacement analysis (Milestone 6) produces the paper's Figure 2 — the quality-speed tradeoff curve. This is the most important result.
