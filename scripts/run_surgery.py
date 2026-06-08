#!/usr/bin/env python
"""Run architecture surgery on Wan 1.3B.

Usage:
    python scripts/run_surgery.py --model wan-1.3b --strategy progressive --num-replace 4
    python scripts/run_surgery.py --model wan-1.3b --strategy all
    python scripts/run_surgery.py --model wan-1.3b --strategy by-cost --profiling-data ../wan-profiler/results/profile_results.json
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_video.cli import main

if __name__ == "__main__":
    sys.exit(main(["surgery"] + sys.argv[1:]))
