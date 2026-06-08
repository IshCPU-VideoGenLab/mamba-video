# CLAUDE.md — mamba-video

> This file is read by Claude Code at the start of every session.
> It provides full context so you never have to re-explain the project.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** mamba-video
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon
- **Phase:** 2 of 7

---

## What This Project Is

This is the **architecture surgery** phase. We take Wan 1.3B's transformer backbone
and systematically replace its attention blocks with Mamba/SSM (Selective State Space
Model) blocks. Then we measure what breaks, what survives, and how much faster it gets.

**mamba-video** does three things:

1. **Surgery** — Programmatically replace attention modules with Mamba SSM blocks
   inside a Wan 1.3B checkpoint, producing a hybrid or fully-SSM model
2. **Benchmarking** — Compare speed, memory, and FLOPs before and after surgery
3. **Quality evaluation** — Measure output quality degradation (FID, FVD, LPIPS)
   to determine how much attention we can remove before quality collapses

The output is a modified model architecture + evidence of what the SSM replacement
costs in quality and what it gains in compute efficiency.

---

## Why Mamba / SSM?

Transformers use self-attention: O(n²) in sequence length. For video, the sequence
length is enormous (frames × spatial patches). This is the single biggest reason
video generation requires GPUs.

Mamba (Selective State Space Models) processes sequences in O(n) time with a
recurrent/sequential scan. This is exactly how CPUs prefer to compute — one
element at a time, sequentially, using the cache hierarchy efficiently.

**Key insight:** Attention's parallelism is what makes it GPU-friendly and CPU-hostile.
Mamba's sequential scan is what makes it CPU-friendly.

---

## Phase 1 → Phase 2 Handoff

Phase 1 (`wan-profiler`) produced a per-module compute breakdown of Wan 1.3B.
That data tells us:

- Which attention modules consume the most time (surgery priority order)
- What percentage of total compute is attention vs FFN vs other
- Memory footprint per module (SSM replacement must not exceed this)

**Before writing code, check `../wan-profiler/results/` for Phase 1 data.**
If Phase 1 results aren't available yet, use conservative estimates from the
Wan 1.3B architecture documentation.

---

## The Bigger Picture

This is Phase 2 of a 7-phase research project:

| Phase | Repo | Status |
|-------|------|--------|
| 1 | wan-profiler | ✅ Structure complete |
| **2** | **mamba-video** (this repo) | **Active** |
| 3 | codec-video-gen | Planned |
| 4 | bitnet-video | Planned |
| 5 | simd-kernels | Planned |
| 6 | (distributed) | Planned |
| 7 | cpu-video-gen | Planned |

---

## Hardware — READ THIS CAREFULLY

- **Primary (development + benchmarking):** MacBook Air M4 — ARM64 / NEON, no GPU.
- **Supported, CI-verified:** commodity x86 with AVX2 (any modern Intel/AMD CPU).
- **Origin, proof-of-concept (retired):** Intel Pentium Gold 7505 — x86-64 / AVX2, 2C/4T, 16 GB.

CPU-native, no GPU, across **both** architectures. We develop on the M4, but **all code must stay
within the commodity-hardware design budget** — assume **2–4 cores, 16 GB RAM (~12 GB usable),
no GPU**, and it must run on x86 (AVX2) **and** ARM (NEON). Never write code that assumes a specific
ISA, many cores, large RAM, or a GPU. The Pentium Gold proved the weakest-hardware case. Python 3.9,
venv (no conda).

### What This Means For Code:

- **CPU-only execution.** No CUDA anywhere. All operations must work on CPU.
- **Memory ceiling is ~12 GB usable** (OS takes ~4 GB). The modified model
  must fit in this budget alongside inputs and intermediates.
- **Mamba kernels must run on CPU.** The official `mamba-ssm` package requires
  CUDA. We use `mamba-ssm` for reference but implement a pure-PyTorch SSM
  scan that runs on CPU. This is slower but correct and portable.
- **float16 everywhere.** Never load in float32 unless explicitly needed for
  numerical stability in a specific operation.
- **Disk budget: ~100 GB.** Don't save multiple full model checkpoints.
  Save only the modified layers + a recipe to reconstruct.

---

## Code Conventions

- **Python 3.9** — no `match` statements, no `list[str]` (use `List[str]`)
- **Type hints** on all function signatures
- **Docstrings** on all public functions (Google style)
- **No classes unless necessary** — prefer functions and dataclasses
- **Logging** via `logging` module, never `print()` for production code
- **Tests** in `tests/` using `pytest`
- **Results** output as JSON

---

## File Structure

```
mamba-video/
├── CLAUDE.md               ← You are here
├── README.md               ← Public-facing project description
├── LICENSE                  ← MIT License
├── requirements.txt        ← Dependencies
├── setup.py                ← Package setup
├── .gitignore
├── lessons.md              ← Mistakes & learnings
├── tasks/
│   └── todo.md             ← Phase 2 task roadmap
├── .claude/
│   ├── settings.json       ← Claude Code guardrails
│   ├── commands/
│   │   ├── review.md       ← /project:review
│   │   └── progress.md     ← /project:progress
│   └── rules/
│       └── python.md       ← Python style rules
├── configs/
│   └── default.json        ← Default surgery configuration
├── src/
│   └── mamba_video/
│       ├── __init__.py
│       ├── config.py        ← Surgery configuration
│       ├── mamba_block.py   ← Pure-PyTorch Mamba SSM block
│       ├── surgery.py       ← Module replacement engine
│       ├── converter.py     ← Weight transfer / initialization
│       ├── quality.py       ← Quality metrics (FID, FVD, LPIPS)
│       ├── benchmark.py     ← Speed & memory comparison
│       ├── report.py        ← Results reporting
│       └── cli.py           ← Command-line entry point
├── scripts/
│   ├── run_surgery.py       ← Execute architecture replacement
│   ├── run_benchmark.py     ← Compare original vs modified
│   └── compare_outputs.py   ← Visual comparison of outputs
├── tests/
│   ├── __init__.py
│   ├── test_mamba_block.py
│   ├── test_surgery.py
│   └── test_benchmark.py
├── results/
│   └── .gitkeep
└── docs/
    └── architecture_plan.md ← Detailed surgery methodology
```

---

## Key Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .

# Run surgery (replace attention with Mamba)
python scripts/run_surgery.py --model wan-1.3b --strategy progressive --output results/

# Benchmark original vs modified
python scripts/run_benchmark.py --original wan-1.3b --modified results/model_modified/ --output results/

# Compare outputs visually
python scripts/compare_outputs.py --original-output results/original/ --modified-output results/modified/

# Run tests
pytest tests/ -v
```

---

## Mamba Block Design Decisions

### Why Pure PyTorch (not `mamba-ssm` package)?

The official `mamba-ssm` package uses custom CUDA kernels for the selective scan.
We can't use CUDA. Our implementation uses a sequential scan loop in pure PyTorch.
It's slower but:

1. Runs on any CPU
2. Is mathematically equivalent
3. Proves the architecture works before Phase 5 optimizes with portable SIMD (AVX2 / NEON)

### SSM Dimensions

The Mamba block maps: `(batch, seq_len, d_model) → (batch, seq_len, d_model)`

Key parameters:
- `d_model`: matches the attention block's hidden dimension
- `d_state`: SSM state dimension (typically 16)
- `d_conv`: local convolution width (typically 4)
- `expand`: expansion factor for inner dimension (typically 2)

### Selective Scan

The core operation: given input `x`, compute parameters `A, B, C, Δ` via
linear projections, then run a recurrent scan:

```
h[t] = A_bar * h[t-1] + B_bar * x[t]
y[t] = C * h[t]
```

where `A_bar, B_bar` are discretized from continuous parameters using `Δ`.

---

## Research Questions This Phase Answers

1. **How much quality is lost** when replacing attention with SSM?
   → Measured by FID/FVD/LPIPS on generated frames
2. **How much faster is the SSM model** on CPU?
   → Measured by wall-clock time per forward pass
3. **What's the memory savings** from removing attention's O(n²) buffers?
   → Measured by peak RSS during inference
4. **Can we do progressive replacement** (one block at a time) to find the
   quality-speed Pareto frontier?
   → Surgery supports replacing N of M attention blocks

---

## Task Management

Check `tasks/todo.md` before starting any work session.

## Lessons Learned

Check `lessons.md` before writing new code.
