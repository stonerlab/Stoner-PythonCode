# Phase 5: test isolation and reliable execution

## First batch (2026-09-11)

The first batch addresses test isolation and the failures carried forward from the
baseline. Dependency compatibility and test categorisation remain later batches.

- Folder `each` tests and the relative-path image test restore their working
  directories through `monkeypatch.chdir`.
- Image round-trip tests and the HDF5 folder test use pytest-managed temporary
  directories. The TIFF round-trip no longer overwrites a tracked fixture.
- The core metadata round-trip also uses a temporary directory and reads the
  actual file just saved, instead of a pre-existing similarly named fixture.
- The ZIP loader opens archive inputs in read mode. Its previous append mode
  required write access to scientific fixtures and caused archive and folder
  tests to fail under restricted write permissions. A regression using the real
  ZIP fixture rejects any open mode that requires writes; it failed before the
  fix.
- Documentation examples begin without leftover figures and close figures even
  on failure. Original exceptions now reach pytest without being replaced by an
  assertion containing a formatted traceback.
- The arbitrary-loader test explicitly adds its helper module's directory to the
  import path and cleans up both registration and module-cache state on failure.
- `clear_routine` now removes entries from the index it is visiting. Previously it
  always edited the loader-pattern index, leaving MIME entries and saver patterns
  behind. Two regression cases cover loader and saver removal, duplicate entries,
  and preservation of unrelated entries; both failed before the one-line fix.
- The baseline runner supports focused selections, enables Python fault reporting,
  and uses a per-run pytest cache as well as its existing temporary directory,
  persistent log, JUnit report and exit-code capture.
- Ordinary runs remove `READTHEDOCS` from the environment. Setting it to `False`
  still enabled the package's presence-based documentation restrictions and
  prevented external image methods from being attached. On this PowerShell,
  assigning a null value through .NET left an empty environment entry; explicit
  removal is required. Documentation runs continue to set it to `True`.

## Environment and dependency scope

Validation uses Windows and the existing Miniforge `py314` environment. The
installed versions and channels checked against `tests/test-env.yml` are recorded
in `environment.json`. The audit found missing `pytesseract` and `tesseract`; these
were installed from conda-forge with `--freeze-installed`, along with their native
dependencies. No dependency declarations were changed. OCR remains optional at
runtime; these packages belong to the comprehensive test environment. Coverage
of their absence is proposed for the next batch.

## Reproduction

Run from the repository root in PowerShell:

```powershell
./maintenance/run-baseline.ps1 -Check focused -TestPaths tests/stoner/tools/test_file.py,tests/stoner/test_FileFormats.py::test_arb_class_load,tests/stoner/image/test_core.py,tests/stoner/test_HDF5.py,tests/stoner/folders/test_each.py
./maintenance/run-baseline.ps1 -Check focused -TestPaths tests/stoner/test_doc_samples.py,-k,STXMIMage
./maintenance/run-baseline.ps1 -Check serial
./maintenance/run-baseline.ps1 -Check parallel
```

Each command reports its directory under ignored `maintenance/runs`. Compact
validation results are retained in `results.json` and `fixture-check.json`.

The isolated arbitrary-loader failure was reproduced before its test setup was
fixed. The focused image/loader selection then passed 45 tests, and the isolated
STXM example passed. A subsequent full run exposed the ZIP write-access and
metadata fixture-write defects; after their fixes, all five focused regressions
passed. Manual example-harness probes also confirmed failure cleanup, directory
restoration and rejection of a stale figure from an earlier test.

The final serial run passed all 337 tests with 612 warnings in 440.95 seconds.
All 73 documentation examples passed. Combined line/branch coverage was 79%.
The two-worker run passed the same 337 tests with 613 warnings in 303.17 seconds,
with identical coverage counts. Neither run skipped tests. All 369 checked
scientific fixtures and plot-cache files retained their pre-run SHA-256 hashes.
The historical disappearance of a runner without a result summary has not
recurred; these runs capture completed results. Its original cause is not
established by this investigation.

## Second batch completed

See `batch2.md` and `batch2-results.json` for the final changes and results:
registered categories, scoped warning policy, meaningful optional-OCR coverage,
and folder/dialog/plotting isolation. Both full runs passed all 342 tests.

## Third batch completed locally

See `batch3.md`, `batch3-environment.json` and `batch3-results.json`. The Python
3.11 lower numerical dependency combination passes all 342 tests serially and
with two workers, with identical coverage counts and unchanged fixture hashes.
The scoped plotting warning fix also passes the Python 3.14 plotting tests.

## Remaining Phase 5 gate

Run the prepared hosted workflows against these changes: the ordinary Python
3.11-3.14/Linux and Python 3.14/macOS matrix, the new lower-dependency Linux job,
and clean wheel/sdist installation probes on Python 3.11 and 3.14. Record their
results before marking Phase 5 complete. Current local results do not establish
new CI or cross-platform success.
