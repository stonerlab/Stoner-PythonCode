# Phase 6: README and front page (2026-09-12)

Refreshed the repository README and documentation landing page. The overview
now covers all five primary classes, metadata and masks, the actual in-place
chaining convention, supported Python versions and the maintained stable branch.
Removed obsolete development promises, legacy module names and historical
installation advice. Links use HTTPS and the stable documentation where relevant.

Installation recommends a Conda-based distribution such as Anaconda or Miniforge,
with both phygbu and conda-forge channels. Pip remains an alternative. Optional
features match pyproject.toml; OCR explicitly requires the optional pytesseract
wrapper and separate Tesseract executable. Published packages may lag the source;
no fresh package installation or remote release validation is claimed here.

The Makefile workflow is preserved: `make commit` calls `make -C doc readme`,
which copies root `README.rst` to `doc/readme.rst`. Both files are identical.
The front page now offers direct guide and API navigation, with the complete
README available as Project Overview. The installation guide includes the
marked installation section of the root README, replacing stale master-branch,
Canopy, network-share and PYTHONPATH instructions. The contributor guide explains
the copy step for direct Sphinx builds.

## Validation

- Standalone docutils HTML rendering of README.rst reports no warnings.
- The README's exact Python example executes on Python 3.14.7. Sorting its clone
  preserves the original row order and retains the temperature metadata.
- Clean RTD-mode Sphinx HTML build succeeds with 405 warnings, unchanged from
  batch 6. All warning categories are unchanged; no warnings were suppressed.
- All five primary classes and 85 dynamic Data methods remain documented, with
  no lost API inventory entries.
- The four edited HTML pages have no problematic markup or new broken local
  links. Checked 143 local links, including fragments. The front page's existing
  theme header placeholder (`# `) is recorded separately; it also occurs in the
  baseline and was not introduced here.
- All 243 plot-cache hashes are unchanged. Package code is unchanged and the
  full runtime suite was not rerun for this prose-only batch.

Evidence: `batch7-checks.json`, `batch7-after.json`, `batch7-html-check.json`
and `batch7-cache-check.json`. Build and scratch verification scripts are under
`maintenance/runs/phase6-batch7-*`. The browser URL policy rejected the local
file preview, so no visual browser review is claimed. Generated HTML structure
and local targets were checked directly instead.

Reproduce the documentation checks from the repository root:

```powershell
Copy-Item README.rst doc/readme.rst
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch7-final -w maintenance/runs/phase6-batch7-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch7-final maintenance/runs/phase6-batch7-final-warnings.log maintenance/phase6/batch7-after.json --compare maintenance/runs/phase6-batch6-final
```

## Next proposed batch

Return to local docstring warnings: correct malformed markup in
`Stoner.Image.util._dtype` and the `DiskBasedFolderMixin` overview, checking
documented types and defaults against implementation. Preserve runtime behaviour
and validate with an AST comparison and another documentation inventory audit.
The intentional fitting-model capitalisation remains unchanged; Linux is the
appropriate validation platform for those pages. Phase 6 remains in progress.

## Localhost visual verification follow-up

On 2026-09-12, served `maintenance/runs/phase6-batch7-final` with Python's
HTTP server bound to `127.0.0.1:8765`. The browser successfully opened
`http://127.0.0.1:8765/index.html`. Inspected the rendered front-page layout
and followed Install Stoner, confirming the Anaconda or Miniforge guidance.
No browser policy change was needed. This completes the front-page visual
check previously blocked by direct file navigation. AGENTS.md and the
contributor guide now record localhost HTTP as the preview workflow.
