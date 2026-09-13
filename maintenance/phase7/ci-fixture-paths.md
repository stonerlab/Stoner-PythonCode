# Linux CI fixture path correction

Investigated 2026-09-13 at `db24317ed4e2f0ef3dc87a46dcbf62f199e5a230`.

## Cause

The [lower-dependency run](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34724802000)
and all four Ubuntu Python 3.11-3.14 jobs in the
[main test run](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34724802021)
each report the same seven failures and 470 passes. All seven failures are
FileNotFoundError from three new fixture references using `tests/stoner`.
Git tracks the directory as `tests/Stoner`. Windows accepted the case mismatch,
and the macOS job passed, masking the defect until the case-sensitive Linux runs.

Five parametrised stream input cases and the current-position stream test
reference `CoreTest.dat`; the filename/stream parity test references
`tdi2-stream.txt`. Corrected those three literals to the tracked spelling.
No package implementation, fixture contents or workflow policy changed.

## Validation

- Compared path literals in both affected test modules against the exact strings
  from `git ls-files`: HEAD has three mismatches; the corrected files have none.
  This check is case-sensitive even on Windows.
- Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0, pytest 9.1.1: all 55 tests in both
  affected modules pass, one warning, 28.20 seconds. Evidence:
  `maintenance/runs/20260913-094303-focused-9c6c3469`.
- Python 3.11.16, NumPy 2.0.2, SciPy 1.14.1, pytest 9.1.1: the same 55 tests
  pass, 18 warnings, 31.44 seconds. Evidence:
  `maintenance/runs/20260913-094314-focused-ae9d1326`.
- Verified every dependency name in `tests/test-env.yml` is present in both
  existing Conda environments; resolved packages are recorded in
  `maintenance/runs/ci-paths-py314-environment.json` and
  `maintenance/runs/ci-paths-py311-environment.json`.

Reproduce with `maintenance/run-baseline.ps1 -Environment py314 -Check focused
-TestPaths tests/Stoner/core/test_stream_input.py,tests/Stoner/core/test_tdi2_loading.py`,
then repeat with `-Environment py311-minimal`.

The full suite was not repeated locally for these path-only test corrections.

## Hosted closure (2026-09-13)

The correction was committed and pushed as `d5c43882c`. All Linux Python
3.11-3.14 and macOS Python 3.14 jobs, plus final test-result publishing, passed
in [run 34748453258](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34748453258).
The [lower-dependency run](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34748453281)
and [package-validation run](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34748453303)
also passed at the same commit. Phase 7 is complete.

This supersedes the earlier commit/push/hosted-validation boundaries in the
Phase 7 batch notes. The historical intermittent Python 3.12 worker loss remains
an unresolved, documented follow-up; the successful runs do not establish its
root cause. Deferred facility-format extensions and optional structural cleanup
remain outside the completed phase's scope.
