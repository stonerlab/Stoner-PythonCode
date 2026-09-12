# Column bounds and temporary masks

Reviewed 2026-09-12. This batch changes mask restoration, not statistical
formulae or the interpretation of bounds.

## Findings and changes

- Non-contiguous bounded selections already return original row indices from
  `min` and `max`, including when the column is selected by name or `setas`.
  The row-index TODOs in `mean` and `std` were stale: neither returns an index.
- `min`, `max`, `mean` and `std` previously skipped `_pop_mask()` when a bounds
  callback raised. Their temporary-mask operations now use `try/finally`.
- `_push_mask()` saved the existing mask by reference and then cleared that
  array in place. Consequently even a successful call could lose the original
  mask. It now saves an independent deep copy. This is the only change to the
  shared mask implementation; nested snapshot restoration has direct coverage.
- Bounds continue to replace the active mask temporarily, as before. This
  batch does not change bounds to intersect with an existing mask, change
  callback signatures, or revise weighted statistics.

## Regression coverage

`tests/Stoner/analysis/test_columns.py` covers callback errors in all four
methods, successful mask restoration, original extrema indices for disjoint
rows, integer/name/default column selection, data and role preservation on
callback failure, and nested independent snapshots.

Before the fix, the initial focused run produced 13 failures and seven passes
in 6.38 seconds (`maintenance/runs/20260912-184217-focused-43cbb6b4`). The
failures demonstrated both lost masks on success and incomplete cleanup on
exceptions. The first corrected run, including core tests, passed 33 tests
in 14.06 seconds (`maintenance/runs/20260912-184342-focused-27bcfcfe`).

Final validation:

- Shared `py311-minimal` (Python 3.11.16, NumPy 2.0.2, SciPy 1.14.1):
  35 focused column/core tests passed, 18 warnings, 34.95 seconds;
  `maintenance/runs/20260912-184444-focused-415b50e7`.
- `py314` full serial suite: 363 passed, 592 warnings, 454.76 seconds;
  `maintenance/runs/20260912-184444-serial-a74d8d94`.
- `git -c core.whitespace=cr-at-eol diff --check` passes, accounting for the
  repository's retained mixed line endings.

These are local results. The separately reported Phase 6 CI failure in run
34705333723 remains a follow-up and is not resolved by this validation.
