# Phase 5 second batch (2026-09-11)

## Changes

- Added registered pytest categories and explicit network/OCR markers. All tests
  have a category; no category is excluded by default. `tests/README.md` documents
  subset commands, optional OCR, isolation and warning policy.
- Removed import-time warning filters from documentation, plotting and filtering
  modules and the broad image-stack UserWarning filter. Plotting/filtering tests
  retain scoped strict warning checks. The exact Agg warning is allowed in the
  plotting module; two manual-axes tests allow the specific GridSpec warning.
  Other warnings remain visible. See `batch2-warning-audit.json` for the starting
  warning inventory.
- Folder metadata tests restore working directories; folder-save output uses
  `tmp_path`. Dialog tests share a per-test mock fixture and pass with the loader
  test executed before the dialog test. Plotting tests get fresh data, restored
  options/rcParams and guaranteed figure cleanup.
- Removed OCR execution during collection. The integration test now skips when
  either optional dependency is unavailable and asserts the real fixture's
  recognised field, scale-bar length and derived pixel scale.
- Added fresh-process import coverage without pytesseract, non-OCR image checks
  with dependencies absent, and regression tests for full/field-only text crops.

## OCR defects exposed by the new assertions

The full metadata path discarded its cropped image and recognised the entire
annotated image for every field. The field-only path attempted to call a private
helper as an attached method. Both now pass the text crop directly to the helper.
Each crop uses Tesseract's documented single-line segmentation (`--psm 7`), verified
against the installed executable's `--help-psm`. The real fixture now yields
50 microns, a field of -0.13, and a 189-pixel scale bar.

Availability now checks both the optional wrapper and the executable on PATH.
Explicit OCR requests preserve existing metadata when dependencies are absent.
No dependency declarations were changed; OCR remains optional at runtime.

## Validation

Focused OCR/plotting/filtering validation passed 17 tests. The independently
ordered dialog and folder isolation tests passed. An unavailable-wrapper run of
the OCR integration test produced one explicit skip, rather than a false pass.
Collection with strict marker validation accounts for all 342 tests, including
two network tests and one OCR integration test. `batch2-categories.json` records
the category counts; overlapping categories must not be summed as test totals.

The full serial run passed all 342 tests with 593 warnings in 432.67 seconds.
The two-worker run passed the same 342 tests with 594 warnings in 262.29 seconds.
Neither full run skipped tests; coverage counts were identical, with 79% combined
line/branch coverage. All 337 first-batch tests remain, with five additional OCR
regression cases. All 369 audited fixture/cache files retained their original
hashes. Results are recorded in `batch2-results.json`.

These results are local Windows/Python 3.14 validation. No new remote CI or other
platform result is claimed.
Dependency-version boundary environments and broader cross-platform validation
remain subsequent Phase 5 work.
