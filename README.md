<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# mamba-video

**Replacing transformer attention with Mamba SSM blocks in Wan 1.3B for CPU-native video generation.**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab) — building the first video generation model that trains and runs entirely on commodity CPUs.

---

## Why Replace Attention?

Self-attention scales quadratically with sequence length — O(n²). In video generation, the sequence is frames × spatial patches, which means attention dominates both compute time and memory. This quadratic cost is the core reason video generation requires GPUs.

**Mamba** (Selective State Space Models) processes sequences in O(n) time using a recurrent scan. This sequential processing pattern aligns naturally with CPU execution: predictable memory access, cache-friendly, and no need for massive parallelism.

**mamba-video** performs architecture surgery on Wan 1.3B: systematically replacing attention blocks with Mamba SSM blocks, then measuring the cost in quality and the gain in speed.

---

## The Bigger Picture

This is **Phase 2** of a 7-phase research project:

| Phase | Repo | Goal |
|-------|------|------|
| 1 | [wan-profiler](https://github.com/IshCPU-VideoGenLab/wan-profiler) | Profile where Wan 1.3B spends compute |
| **2** | **mamba-video** (this repo) | **Replace attention with Mamba/SSM** |
| 3 | codec-video-gen | Codec-inspired temporal design |
| 4 | bitnet-video | 1-bit quantization (BitNet) |
| 5 | avx2-kernels | Native AVX2 CPU execution engine |
| 6 | (distributed) | Distributed CPU training |
| 7 | cpu-video-gen | Flagship paper repo |

---

## Features

- **Progressive surgery** — replace 1, some, or all attention blocks with Mamba
- **Pure PyTorch SSM** — no CUDA dependency; runs on any CPU
- **Quality metrics** — FID, FVD, LPIPS to measure degradation
- **Speed benchmarks** — wall-clock comparison of original vs modified model
- **Memory tracking** — measure the O(n²) → O(n) memory savings

---

## Installation

```bash
git clone https://github.com/IshCPU-VideoGenLab/mamba-video.git
cd mamba-video

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

pip install -r requirements.txt
pip install -e .
```

---

## Usage

### 1. Run Architecture Surgery

```bash
# Replace all attention blocks with Mamba
python scripts/run_surgery.py --model wan-1.3b --strategy all --output results/

# Progressive: replace only the first N attention blocks
python scripts/run_surgery.py --model wan-1.3b --strategy progressive --num-replace 4 --output results/

# Replace only the most expensive blocks (guided by Phase 1 profiling)
python scripts/run_surgery.py --model wan-1.3b --strategy by-cost --profiling-data ../wan-profiler/results/profile_results.json --output results/
```

### 2. Benchmark Original vs Modified

```bash
python scripts/run_benchmark.py \
    --original wan-1.3b \
    --modified results/model_modified/ \
    --output results/benchmarks/
```

### 3. Compare Output Quality

```bash
python scripts/compare_outputs.py \
    --original-output results/original_frames/ \
    --modified-output results/modified_frames/ \
    --output results/quality/
```

### Python API

```python
from mamba_video.config import SurgeryConfig
from mamba_video.surgery import perform_surgery
from mamba_video.benchmark import compare_models

config = SurgeryConfig(
    model_name="wan-1.3b",
    strategy="progressive",
    num_replace=4,
    d_state=16,
    d_conv=4,
    expand=2,
)

# Perform surgery
modified_model, surgery_report = perform_surgery(config)

# Benchmark
benchmark_results = compare_models(
    original_name="wan-1.3b",
    modified_model=modified_model,
    num_steps=5,
)
```

---

## How It Works

### The Mamba Block

Our pure-PyTorch Mamba block implements the selective scan from [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752):

```
Input x → Linear projection → Conv1D → SSM scan → Output projection → y

SSM scan (per timestep):
    h[t] = Ā · h[t-1] + B̄ · x[t]
    y[t] = C · h[t]

where Ā, B̄ are discretized from learned (A, B, Δ) parameters.
```

The block is a drop-in replacement for attention: same input/output dimensions, same position in the network.

### Surgery Strategies

| Strategy | Description |
|----------|-------------|
| `all` | Replace every attention block with Mamba |
| `progressive` | Replace blocks one at a time, starting from the first |
| `by-cost` | Replace the most compute-expensive blocks first (uses Phase 1 data) |
| `alternating` | Replace every other attention block |

---

## Project Structure

```
mamba-video/
├── CLAUDE.md              # Claude Code context
├── README.md              # You are here
├── src/
│   └── mamba_video/
│       ├── config.py      # Surgery configuration
│       ├── mamba_block.py  # Pure-PyTorch Mamba SSM
│       ├── surgery.py     # Module replacement engine
│       ├── converter.py   # Weight initialization
│       ├── quality.py     # FID/FVD/LPIPS metrics
│       ├── benchmark.py   # Speed & memory comparison
│       ├── report.py      # Report generation
│       └── cli.py         # CLI entry point
├── scripts/               # Convenience scripts
├── tests/                 # Unit tests
├── configs/               # Default configurations
├── docs/                  # Methodology docs
└── results/               # Output directory
```

---

## Citation

```bibtex
@software{kwakye2026mambavideo,
  author = {Kwakye, Ishmael Affum},
  title = {mamba-video: Attention-to-SSM Architecture Surgery for CPU-Native Video Generation},
  year = {2026},
  url = {https://github.com/IshCPU-VideoGenLab/mamba-video},
  institution = {University of Ghana, Legon}
}
```

---

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023
- [Wan Video](https://github.com/Wan-Video/Wan) — Base model
- [wan-profiler](https://github.com/IshCPU-VideoGenLab/wan-profiler) — Phase 1 profiling data

---

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Phase 2 of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab). Replacing quadratic attention with linear SSM — one block at a time.*
