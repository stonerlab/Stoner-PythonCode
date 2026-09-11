# Phase 6 documentation batch (2026-09-11)

## Changes

The two class templates now use class-relative autosummary names within their
declared module. The primary-class index declares its current module explicitly
and uses relative names too. This removes redundant-module warnings without
suppressing Sphinx diagnostics. ImageStack is linked from that index to its
existing full reference page and included in the inheritance diagrams.

The package-documentation heading is corrected. The developer guide now links
the repository guidance, maintenance plan and docstring standard, and gives
environment, focused-test and cached-documentation commands.

`maintenance/audit-docs.py` checks the generated inventory against the five
primary runtime classes and 85 public Data methods attached from core, analysis,
fitting and plotting modules. With `--compare`, it also reports lost Python API
entries relative to an earlier build. Missing classes, dynamic methods or prior
entries cause a non-zero exit status. Warnings are classified, not suppressed.

## Validation

Python 3.14.7, Sphinx 9.1.0 and NumPy 2.5.2 in the existing complete `py314`
environment, importing this checkout. Both comparison builds regenerated the
ignored `doc/classes` directory and used `READTHEDOCS=True`. Original generated
pages were retained under ignored `maintenance/runs` paths. The first build with
the existing generated pages gave the same counts as the fresh baseline.

| Warning category              | Before | After |
| ----------------------------- | ------ | ----- |
| Redundant autosummary names   | 1152   | 0     |
| Docutils markup               | 347    | 347   |
| Duplicate object descriptions | 76     | 76    |
| Missing autosummary stubs     | 57     | 57    |
| Ambiguous Python references   | 24     | 24    |
| Footnote references           | 1      | 1     |
| Other                         | 1      | 1     |
| Total                         | 1658   | 506   |

The HTML build succeeds. All five primary classes and all 85 audited dynamic
methods remain documented, with no lost Python inventory entries. All 243 plot
cache files have unchanged SHA-256 hashes. See `clean-before.json`, `after.json`
and `plot-cache-check.json`. No runtime source or example behaviour changed;
the full test suite was not repeated for this documentation-only batch.

To reproduce in an activated documentation environment, set `READTHEDOCS=True`
and run from the repository root:

```powershell
python -m sphinx -b html -E doc maintenance/runs/phase6-final -w maintenance/runs/phase6-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-final maintenance/runs/phase6-final-warnings.log maintenance/phase6/after.json --compare maintenance/runs/phase6-clean-before
```

The comparison directory must contain a previously built `objects.inv`. Omit
`--compare` for a standalone coverage audit. Template changes require fresh
generated API sources, not just `-E`: sphinx-automodapi retains existing stubs.
Archive only the ignored `doc/classes` directory before rebuilding; retain
`doc/plot_cache`. In this session the sandbox denied recreating `doc/classes`,
so generation needed an approved elevated command. The existing baseline
output directory was also unwritable; builds used fresh maintenance paths.

## Next proposed batch

Fix malformed Google-style sections in the Kerr image helper docstrings,
starting with `crop_text`, `defect_mask` and `defect_mask_subtract_image`.
Their indentation errors recur across generated aliases and inherited pages.
Check each description against its implementation, preserve scientific meaning,
and compare fresh Sphinx warning counts and inventories afterwards.

Separately investigate the 57 missing autosummary stubs, which largely concern
imported names, and the duplicate attribute descriptions. These remain visible
in this batch; a successful Sphinx exit is not a zero-warning documentation gate.
Phase 6 remains in progress.

## Second batch

The three Kerr helper docstrings above are now corrected. See `batch2.md` for
the warning comparison, source-only documentation check and next stack-docstring
proposal.

## Third batch

The stack cropping, conversion and legacy drift-correction docstrings are
corrected in `batch3.md`. That report also records a reproduced same-dtype
`force_copy=True` converter defect for a later runtime fix.

## Fourth batch

The Data overview, Setas call semantics and vector-plotting docstrings are
corrected in `batch4.md`. Two obsolete Data attribute entries are explicitly
reviewed rather than retained as misleading API inventory targets.
