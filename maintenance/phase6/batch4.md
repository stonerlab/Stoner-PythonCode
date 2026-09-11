# Phase 6 fourth batch: Data, column roles and vector plotting (2026-09-12)

## Changes

The Data overview now describes the current numerical array, metadata and
properties, with corrected types and markup. The obsolete `patterns` and
`subclasses` entries are removed. Conditional column-index attributes such as
`xcol` remain documented: a populated-data probe confirmed they are available
when roles are assigned, although absent on an empty Data object.

`Setas.__call__` now has valid examples and describes its actual assignment,
reset and return behaviour. `_self=True` returns the same Setas instance, not a
copy. A bare call returns the current role list. The positional example uses
one sequence, rather than multiple positional arguments whose later values
would be ignored.

`plot_xyuv` now documents the real signature and artist return, removes the
nonexistent `zcol` parameter, and explains the two plotting stages. It records
that `save_filename` is forwarded only to the arrow-overlay stage and therefore
does not save when `no_quiver=True`. No plotting implementation was changed.

## Validation

Only these three package docstrings changed; the source AST comparison confirms
identical executable structure. Focused Setas probes verify identity, bare-call
and reset=False behaviour (`batch4-source-check.json`). The full runtime suite
was not repeated for this documentation-only change.

The clean RTD-mode HTML build succeeds with 443 warnings, down from 470. All 27
removed diagnostics are markup warnings; other categories are unchanged. The
Python 3.14.7 / Sphinx 9.1.0 environment matches the preceding batch.

| Warning category              | Before | After |
| ----------------------------- | ------ | ----- |
| Docutils markup               | 311    | 284   |
| Duplicate object descriptions | 76     | 76    |
| Missing autosummary stubs     | 57     | 57    |
| Ambiguous Python references   | 24     | 24    |
| Footnote references           | 1      | 1     |
| Other                         | 1      | 1     |
| Total                         | 470    | 443   |

The build and API audit are recorded in `batch4-after.json`.
All five primary classes and 85 dynamic Data methods remain documented. Two
inventory entries are intentionally removed: `Data.patterns` and `Data.subclasses`
were generated from stale prose, not current runtime attributes. Their explicit
review list and runtime absence check are in `batch4-obsolete-attributes.json`.

The audit's optional `--allow-removed` argument accepts this reviewed list, while
retaining the complete removal list in its report. Unexpected removals still fail
the audit, as do missing primary classes or dynamic methods. An unrestricted
comparison fails on the two stale entries; the exact reviewed list passes.

All 243 plot-cache hashes remain unchanged (`batch4-cache-check.json`). No warning
filters were added. Run the following in the activated documentation environment:

```powershell
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch4-final -w maintenance/runs/phase6-batch4-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch4-final maintenance/runs/phase6-batch4-final-warnings.log maintenance/phase6/batch4-after.json --compare maintenance/runs/phase6-batch3-final --allow-removed maintenance/phase6/batch4-obsolete-attributes.json
```

## Next proposed batch

Correct the remaining package-owned fitting-model markup in `Lorentzian_diff`,
`BlochLaw` and `Ic_B_Airy`, preserving the equations and checking parameter
descriptions against the implementation. Repeat the AST, warning, inventory and
cache checks. Imported-symbol stubs and third-party documentation remain separate
warning categories. Phase 6 remains in progress.
