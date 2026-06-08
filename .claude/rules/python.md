# Python Rules — mamba-video

> Claude Code: follow these rules for every Python file you create or edit.

## Language Version
- Target: Python 3.9
- No `match` statements (3.10+)
- No `type X = ...` syntax (3.12+)
- Use `typing` imports: `List`, `Dict`, `Optional`, `Tuple`, `Union`

## Type Hints
- Required on ALL function signatures (parameters + return type)
- Use `List[str]` not `list[str]` (3.9 compatibility)

## Docstrings
- Google style on all public functions and classes
- Include `Args:`, `Returns:`, and `Raises:` sections

## Logging
- Use `logging` module, NEVER `print()` for production code
- Logger: `logging.getLogger(__name__)`

## Structure
- Prefer functions and `dataclasses` over classes with methods
- Keep files under 300 lines. Split if longer.
- Absolute imports only

## Memory Safety
- Never load entire model without checking RAM
- Use `torch.no_grad()` for all inference
- Delete large tensors + `gc.collect()` when memory is tight
- Prefer `float16` / `bfloat16` over `float32`

## SSM-Specific Rules
- All SSM operations must work on CPU tensors (never `.cuda()`)
- Use `torch.float32` for discretization math (A, B → Ā, B̄) to avoid
  numerical instability, then cast back to float16 for the scan
- The selective scan loop must be a simple Python for-loop over timesteps —
  do NOT try to parallelize it (that's Phase 5's job, via portable SIMD kernels)
- Always clamp Δ (delta) values to prevent NaN: `delta.clamp(min=1e-4)`
- SSM state dimension `d_state` should default to 16 (not 64) to stay
  within memory budget on the Pentium Gold

## Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

## Testing
- Test files: `tests/test_<module>.py`
- Test functions: `test_<what_it_tests>`
- Use `pytest` fixtures, not `unittest.TestCase`
- Always test with small dimensions first (d_model=64, seq_len=16)
