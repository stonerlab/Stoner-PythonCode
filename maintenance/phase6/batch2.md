# Phase 6 second batch: Kerr helper docstrings

## Changes

Revised only the docstrings for `crop_text`, `defect_mask` and
`defect_mask_subtract_image` in `Stoner/Image/kerrfuncs.py`. Their arguments,
defaults, conditional return values and indentation now follow
`DOCSTRING_STYLE.md`.

The descriptions follow the implementation: an already cropped image is returned
unchanged even with `copy=True`; intensity masks select values strictly between
the thresholds; corner regions use rounded square slice bounds. The optional
intermediate-result dictionaries are described explicitly. The related-method
link targets the documented `KerrArray` method.

These are documentation corrections, not algorithm changes. An AST comparison
after removing docstrings confirms identical executable structure and exactly
three changed docstrings (`batch2-source-check.json`).

## Validation

The full RTD-mode HTML build succeeds in the same Python 3.14.7 / Sphinx 9.1.0
environment as the preceding batch. Its 506 warnings fall to 490: all 16 repeated
markup diagnostics from these three helpers disappear. No warning filters were
added. The related-method link resolves to an existing rendered API target.

| Warning category              | Before | After |
| ----------------------------- | ------ | ----- |
| Docutils markup               | 347    | 331   |
| Duplicate object descriptions | 76     | 76    |
| Missing autosummary stubs      | 57     | 57    |
| Ambiguous Python references   | 24     | 24    |
| Footnote references           | 1      | 1     |
| Other                         | 1      | 1     |
| Total                         | 506    | 490   |

All five primary classes and 85 dynamic Data methods remain documented, with
no lost Python inventory entries relative to the preceding batch. All 243 plot
cache files retain their SHA-256 hashes. Reports are `batch2-after.json` and
`batch2-cache-check.json`. Runtime tests were not repeated for docstring-only
changes; the AST comparison and documentation build validate this batch.

Reproduce in the activated documentation environment from the repository root:

```powershell
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch2-clean -w maintenance/runs/phase6-batch2-clean-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch2-clean maintenance/runs/phase6-batch2-clean-warnings.log maintenance/phase6/batch2-after.json --compare maintenance/runs/phase6-final
```

The comparison path is the preceding batch's HTML output. Generated API stubs
use the unchanged templates; `-E` rereads their current source docstrings.

## Next proposed batch

Fix the adjacent stack docstrings: `KerrStackMixin.crop_text`,
`ImageStackMixin.convert` and `ImageStackMixin.correct_drifts`. Their malformed
return, argument and list sections still generate repeated warnings. Check the
actual stack mutation, conversion and deprecated drift-correction behaviour
before describing it. Retain the same AST, Sphinx, inventory and cache checks.

Phase 6 remains in progress; imported-symbol stubs, duplicate descriptions and
other malformed docstrings are still visible in the warning audit.
