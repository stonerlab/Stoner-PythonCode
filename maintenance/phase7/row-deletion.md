# Inverted scalar row deletion

Fixed 2026-09-12. `Data.del_rows(row, invert=True)` referenced the undefined
local variable `c`, so keeping a single row raised UnboundLocalError.

The scalar branch now resolves `col` through `range(len(datafile))` and
delegates to the existing list-based inverted selection. This supports
negative scalar indices and rejects out-of-range indices before mutation.
Simply substituting `col` for `c` would have deleted every row for negative
or invalid indices; focused tests caught that before completion.

The tuple-range implementation was already present and works with inclusive
endpoints in either order. Removed its stale TODO after verifying ordinary
and inverted selection. List-index semantics and other deletion branches
were not changed.

## Validation

- Original code: three failing scalar cases, four passing range cases,
  5.76 seconds (`maintenance/runs/20260912-213027-focused-b77513c2`).
- Intermediate variable-only correction: four failures for negative/invalid
  indices, seven passes, 5.47 seconds
  (`maintenance/runs/20260912-213220-focused-c72beacc`).
- Final Python 3.14 core suite: 54 passed, one warning, 14.00 seconds
  (`maintenance/runs/20260912-213257-focused-e87fcf0b`).
- Shared Python 3.11 minimal environment: 11 focused cases passed,
  18 warnings, 7.86 seconds
  (`maintenance/runs/20260912-213308-focused-0d4b9279`).

Regression checks preserve row order, data, masks, metadata, headers, column
roles and the in-place return contract. Invalid indices leave data unchanged.
The final documentation-only wording update followed the runtime tests.
`git -c core.whitespace=cr-at-eol diff --check` passes. The full repository
suite was not repeated for this isolated branch fix; hosted validation remains
outstanding.
