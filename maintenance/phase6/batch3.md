# Phase 6 third batch: stack docstrings

## Changes

Corrected `KerrStackMixin.crop_text`, `ImageStackMixin.convert` and
`ImageStackMixin.correct_drifts` without changing their executable statements.

- Cropping describes the in-place resize, accepted image dimensions, retained
  pixel masks, unchanged image count and return of the same stack. The return
  description also covers MaskStack, which shares the mixin.
- Conversion documents its actual signature, preserved mask, return of the same
  stack, normalisation and error conditions. Removed the nonexistent `image`
  argument and repaired the numbered references.
- Legacy drift correction documents argument forwarding, deprecation and the
  `None` return, rather than implying a chainable result.

## Confirmed runtime follow-up

The shared converter has an existing same-dtype copy defect. A public API probe
using `ImageStack(np.zeros((2, 3, 4), dtype=np.uint8))`, followed by
`stack.convert(stack.imarray.dtype, force_copy=True)`, raises:

```text
AttributeError: 'numpy.ndarray' object has no attribute 'clone'
```

`imagefuncs.convert` first calls `np.asarray` and then accesses `.clone` on that
ordinary array. The revised docstring identifies this as a current limitation,
not an intended API contract. `batch3-copy-probe.json` records the reproduction.
The runtime fix is deferred to a focused source-maintenance batch with regression
coverage for copying, masks and unchanged-dtype conversion.

## Validation

The clean RTD-mode HTML build succeeds with Python 3.14.7 and Sphinx 9.1.0.
Warnings fall from 490 to 470; all 20 markup diagnostics from the three edited
docstrings disappear. No suppression was added.

| Warning category              | Before | After |
| ----------------------------- | ------ | ----- |
| Docutils markup               | 331    | 311   |
| Duplicate object descriptions | 76     | 76    |
| Missing autosummary stubs     | 57     | 57    |
| Ambiguous Python references   | 24     | 24    |
| Footnote references           | 1      | 1     |
| Other                         | 1      | 1     |
| Total                         | 490    | 470   |

The source AST comparison confirms only these three docstrings changed. All five
primary classes and 85 dynamic Data methods remain documented, with no lost
Python inventory entries. All 243 plot-cache files have unchanged SHA-256 hashes.
See `batch3-source-check.json`, `batch3-after.json` and `batch3-cache-check.json`.
The full runtime suite was not repeated for docstring-only changes.

Reproduce from the repository root in the activated documentation environment:

```powershell
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch3-final -w maintenance/runs/phase6-batch3-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch3-final maintenance/runs/phase6-batch3-final-warnings.log maintenance/phase6/batch3-after.json --compare maintenance/runs/phase6-batch2-clean
```

## Next proposed documentation batch

Address the remaining package-owned markup in the Data class overview,
`Setas.__call__` and `plot_xyuv`: malformed literals, an empty literal block and
incorrect indentation. Check descriptions against the implementation and retain
the same source, inventory and cache checks. Imported-symbol stubs, duplicated
descriptions and third-party docstrings remain separate warning categories.
Phase 6 remains in progress.
