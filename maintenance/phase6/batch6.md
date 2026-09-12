# Phase 6: module index imports (2026-09-12)

Classified all 57 missing autosummary stubs using runtime object ownership,
explicit module exports and generated filenames. The complete classification
is in `batch6-stub-classification.json`.

32 entries are incidental imports in five module indexes: core data (16),
plotting functions (7), HDF5 image loaders (4), format decorators (1), and
folder functions (4). None is explicitly exported through those modules'
`__all__`; all are excluded by automodsumm's local-object generation filter.
The installed extension applies that filter during stub generation but not
during table rendering. Explicit `:skip:` lists in `doc/Stoner.rst` align the
affected tables with their generated pages without changing runtime exports
or globally disabling imported-member documentation.

The other 25 entries are explicitly exported fitting functions whose names
differ from their model classes only in capitalisation. This is intentional
API design, confirmed by the maintainer: related purpose, different entity
types. Windows cannot create both default page filenames. Documentation is
normally built on Linux, so these local warnings are a platform limitation,
not evidence that the public API should be renamed or its pages restructured.
A Linux build remains the appropriate check for those pages; this batch does
not claim fresh Linux validation or suppress the local warnings.

## Validation

Archived the ignored `doc/classes` directory under
`maintenance/runs/phase6-batch6-before-classes` and regenerated all API sources.
The clean RTD-mode HTML build succeeds with Python 3.14.7 and Sphinx 9.1.0.
Warnings fall from 438 to 405: missing stubs from 57 to 25, and docutils
warnings from 279 to 278 after removing the incidental table entries.
Other warning categories are unchanged.

`batch6-after.json` confirms no lost API inventory entries, all five primary
classes and all 85 dynamic Data methods documented. `batch6-cache-check.json`
confirms all 243 plot-cache files are unchanged. No package code changed;
runtime tests were not rerun for this documentation-only batch.

Reproduce after archiving the ignored generated API sources:

```powershell
$env:READTHEDOCS = 'True'
$env:MPLBACKEND = 'Agg'
$env:QT_QPA_PLATFORM = 'offscreen'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch6-final -w maintenance/runs/phase6-batch6-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch6-final maintenance/runs/phase6-batch6-final-warnings.log maintenance/phase6/batch6-after.json --compare maintenance/runs/phase6-batch5-final
```

## Next proposed batch

Refresh `README.rst` and `doc/index.rst`: review the project introduction,
features, installation and optional dependencies, supported Python versions,
repository links and navigation against current manifests and the user guide.
Keep detailed API material in its existing pages and verify the rendered
front page with a documentation build. Phase 6 remains in progress.
