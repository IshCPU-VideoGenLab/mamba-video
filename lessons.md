# Lessons Learned

> This file accumulates rules and corrections discovered during development.
> Claude Code checks this before writing new code to avoid repeating mistakes.

---

<!-- Example format:
## 2026-06-01 — SSM state dimension mismatch
**What happened:** Used d_state=64 which caused memory to exceed budget on 16GB machine.
**Rule:** Keep d_state ≤ 16 to fit the commodity-hardware budget (~16 GB, 2–4 cores). Only increase if benchmarks confirm headroom.
-->
