# Folder traversal and recursive metadata slicing

Implemented 2026-09-12 following the maintainer's review.

## Documented traversal bug fix

`walk_groups(only_terminal=False)` previously skipped non-terminal members
and failed to forward the option to recursive calls. It now visits every
level, depth-first in stored group order, with children before their parent.
Breadcrumbs and member order are retained. Matching the existing documented
contract is a bug fix, not an API-breaking change.

The default `only_terminal=True`, existing return behaviour and terminal
replacement behaviour are preserved. Regression tests include a parent whose
last subgroup is replaced, ensuring it is not unexpectedly visited again.

## Backward-compatible addition

`folder.metadata.slice(..., recurse=True)` and
`folder.slice_metadata(key, recurse=True)` now collect metadata through
`walk_groups(..., only_terminal=False)`. Recursion is opt-in; direct-member
slicing remains the default. Existing output formats and defaults are retained,
with no additional group-path columns and no changes to the folder hierarchy.

Key matching, common keys and missing-value handling apply across all visited
members before formatting. Tests cover nested and sibling groups, root members,
wildcards, vector metadata, empty folders, missing keys and numeric outputs.
The method docstrings and user guide describe the traversal and opt-in API.

## Validation

- Before the runtime fix: eight failures reproduced the defects
  (`maintenance/runs/20260912-220051-focused-68fc7e47`).
- Final focused Python 3.14 checks: 13 passed, one warning, 7.16 seconds
  (`maintenance/runs/20260912-220336-focused-21882b30`).
- Shared Python 3.11 minimal environment: 13 passed, 18 warnings, 10.80 seconds
  (`maintenance/runs/20260912-220444-focused-8a1d145e`).
- Fresh cached Sphinx build and audit passed: the same 106 reviewed Windows
  warnings, all five primary classes and 85 dynamic Data methods retained,
  no removed API objects or unexpected warnings. Build and audit evidence:
  `maintenance/runs/phase7-recursive-docs*`.

- Full Python 3.14 serial suite: 396 passed, 592 warnings, 431.69 seconds
  (`maintenance/runs/20260912-220433-serial-fb4d4d36`).
- No tracked plot-cache changes; whitespace diff checks pass.

Hosted CI has not been run for this batch. Changes remain uncommitted.
