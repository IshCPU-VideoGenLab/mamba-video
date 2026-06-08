# /project:review

Perform a full code review of the mamba-video project.

## Steps

1. Read `CLAUDE.md` for project context and constraints.
2. Read `lessons.md` for known issues.
3. Review all Python files in `src/mamba_video/` for:
   - Type hint completeness (using `typing` imports for 3.9)
   - Google-style docstrings on all public functions
   - No `print()` — must use `logging`
   - Python 3.9 compatibility
   - Memory safety (no OOM risk on 16GB machine)
   - CPU-only compatibility (no CUDA calls, no `.cuda()`, no `device="cuda"`)
   - Numerical stability in SSM operations (discretization, scan)
   - Line length ≤ 100 characters
4. Verify `MambaBlock` dimensions match attention block dimensions.
5. Run `pytest tests/ -v` and report results.
6. Summarize findings as:
   - **Critical** — will break on target hardware or produce wrong results
   - **Important** — violates project conventions
   - **Suggestions** — improvements for clarity or performance
