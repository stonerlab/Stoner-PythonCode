# Phase 0 baseline — 2026-09-10

Baseline: `stable`, commit `08ad42f0997a5ced3a630cc05ecde9576295a3b2`, plus the pre-existing working changes recorded in `initial-status.txt` and `pre-baseline.patch`. This is a working-checkout baseline, not an unchanged release-commit baseline.

The existing Miniforge `py314` environment is used in accordance with AGENTS.md. Every dependency named in `tests/test-env.yml` is installed. Exact package builds are recorded in `conda-explicit.txt`; `conda-packages.json` and `versions.json` record the inspected environment. Recreating this environment from the portable test manifest has not yet been verified.

Editable installation succeeded using `python -m pip install --no-deps --no-build-isolation -e .`. Collection found **333 tests** in **32.20 s** with one SciPy ODR deprecation warning.

Initial smoke testing encountered `PermissionError` accessing the existing temporary root `C:\Users\gavin\AppData\Local\Temp\pytest-of-phygbu`. The incomplete run was interrupted and retained in `smoke.log`. `setup-diagnostic.log` records the cause. Subsequent runs use a fresh, explicitly named `--basetemp`; this changes test execution configuration only.

Documentation: `READTHEDOCS=True`, `MPLBACKEND=Agg`, `QT_QPA_PLATFORM=offscreen`, then `python -m sphinx -b html -E doc doc/_build/phase0 -w maintenance/phase0/sphinx-warnings.log`. Build succeeded with **1659 warnings**, using the retained plot cache. These warnings are separate from test results. See `sphinx.log` and `sphinx-warnings.log`.

Full serial run: **333 passed, 612 warnings in 448.47 seconds**, exit 0. Combined statement/branch coverage is **79%** (line coverage 82.49%, branch coverage 70.59%). No tests skipped. The two-worker run also passed **333 tests**, with **613 warnings in 265.50 seconds**, exit 0, no skips, and identical coverage counts. No package behavior or test failures have been fixed in this phase.
## Repeating checks

From the repository root, after installing the checkout in a supported environment:

```powershell
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python -m pip install --no-deps --no-build-isolation -e .
.\maintenance\run-baseline.ps1 -Check collect
.\maintenance\run-baseline.ps1 -Check smoke
.\maintenance\run-baseline.ps1 -Check serial
.\maintenance\run-baseline.ps1 -Check parallel
.\maintenance\run-baseline.ps1 -Check docs
```

Use `-Environment py313` or `-Conda <supported-conda-executable>` when appropriate. Each invocation saves a timestamped report directory under `maintenance/runs`, records its command, duration and exit code, and uses a unique temporary directory. Runs must finish sequentially because existing tests write into shared fixtures and example directories. The runner's collection mode was executed successfully: 333 tests collected in 11.28 seconds.

The Sphinx build generated three previously missing retained cache files: `doc/plot_cache/stitch_int_overlap.png`, `.hires.png`, and `.pdf`. They are generated baseline output, not hand-edited scientific fixtures.

The corrected smoke run completed with **124 passed, 1 failed, 139 warnings in 153.41 s**. Its sole failure was `tests/stoner/test_FileFormats.py::test_arb_class_load`, reporting `Cannot locate a loader function for dummy.ArbClass`. The same case passes in the full serial execution order: investigate import/registry state in phase 5 before treating an isolated failure as a fixed package defect.