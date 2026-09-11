# Phase 5 third batch: supported dependency boundaries

## Scope and environment

The local prefix environment was created from the complete test dependency list,
with Python 3.11 and lower supported numerical dependency minor lines. Its
manifest is now retained as `tests/minimum-env.yml`; all 36 dependency names
match `tests/test-env.yml`. `batch3-environment.json` records the installed
versions, builds, channels and satisfaction of the ordinary test requirements.

The resolved combination includes Python 3.11.16, NumPy 2.0.2, SciPy 1.14.1,
Matplotlib 3.8.4, scikit-image 0.24.0, lmfit 1.3.4 and pandas 2.3.3. Pandas has no
declared lower bound and resolves normally. These are supported minor-line
checks, not validation of every earliest patch or every possible combination.
No runtime dependency requirements have changed. OCR remains optional.

Conda used its libmamba solver to create the environment. This does not imply
that Conda uses mamba's complete download/installation implementation.

## Changes and initial findings

- The PowerShell runner accepts `-EnvironmentPrefix` for a workspace environment,
  retaining named-environment support and the existing `py314` default.
- The initial focused run passed 14 cases and failed three plotting cases:
  Matplotlib 3.8 invokes deprecated Pyparsing aliases, and the strict plotting
  warning policy converted those upstream warnings into failures.
- Plotting tests now narrowly filter alias deprecations from Matplotlib's font
  and math parser modules, plus the exact `parseAll` deprecation reported inside
  Pyparsing's compatibility wrapper. Other warnings remain errors in these tests;
  import-time and documentation warnings remain visible.
- The final focused plotting run passed all five cases on both Python 3.11 and
  Python 3.14. The initial OCR/filtering selection also passed on Python 3.11.
- The new lower-dependency workflow uses micromamba and runs the entire suite
  with two workers on Linux. It does not upload a second set of coverage flags.
- Existing installed-package probes already exercise real loading, packaged
  styles, OCR resources and an image round-trip outside the checkout. Their
  workflow now covers both Python 3.11 and 3.14. No duplicate probes were needed.
- CI policy validation checks the new workflow and package endpoint matrix.
- The real-data Attocube interpolation case is marked `slow`: SciPy 1.14's
  `griddata` calculation exceeded the runner's two-minute diagnostic threshold,
  then completed successfully. The diagnostic does not terminate a test. The
  marker permits quick selections without excluding this case from full runs.

## Validation status

Full serial: 342 passed, no skips, 6856 warnings, 867.38 seconds. Two workers:
342 passed, no skips, 6977 warnings, 509.81 seconds. Both runs exercised all
73 documentation examples, have identical test identities and coverage counts,
and report 79% combined coverage. The counts also match batch 2 on Python 3.14:
13651/16540 lines and 4691/6638 branches covered. The two local runs overlap, so
their durations must not be interpreted as a serial/parallel speed comparison.

All 369 audited scientific fixtures and retained plot-cache files have unchanged
SHA-256 hashes. Detailed counts and log locations are in `batch3-results.json`.
Older plotting dependencies emit substantially more deprecation warnings than
the Python 3.14 environment; those warnings remain visible outside the narrowly
scoped strict plotting tests. There is no evidence here requiring raised minimum
versions or changes to the scientific implementation.

YAML parsing, the 20 pinned-action policy checks and dependency-name alignment
passed locally. Hosted execution of this batch remains pending; previous Phase 4
CI results do not establish success for these changed files. Distribution builds
remain a GitHub Actions responsibility.

Strict marker collection selected one `slow` case from 342. A direct invocation
of the Python 3.14 executable without Conda activation failed in an Astropy native
import; repeating collection through `conda run -n py314` passed. Use the runner
or an activated environment so native-library paths are configured correctly.

## Reproduction

Create a Conda or mamba environment from `tests/minimum-env.yml`, then run:

```powershell
./maintenance/run-baseline.ps1 -EnvironmentPrefix D:/PythonCode/maintenance/runs/envs/py311-minimum -Check serial
./maintenance/run-baseline.ps1 -EnvironmentPrefix D:/PythonCode/maintenance/runs/envs/py311-minimum -Check parallel
```

## Next proposed batch

After Phase 5's hosted validation, resume Phase 6 by grouping the remaining
Sphinx warnings by cause, checking documentation coverage of public and
dynamically attached APIs, and fixing a small evidenced group of documentation
defects. Use the retained plot cache for RTD-mode validation. Preserve scientific
meaning and avoid refreshing generated assets as incidental cleanup.

## macOS CI follow-up (2026-09-11)

At implementation commit `d143d6f21`, macOS job `103432649895` in run
`34650928065` resolved Pyparsing 3.0.4 and pytest-cov 3.0.0. It ran 337 cases
successfully but reported one collection error and a coverage-combine internal
error. The five plotting cases could not import the newer
`PyparsingDeprecationWarning` class. The warning filters now check that the class
exists; older Pyparsing needs no filter for warnings it does not emit.

The older pytest-cov subprocess hook rediscovered configuration from the child
working directory when its default `.coveragerc` path was absent. The new OCR
import subprocess runs in a temporary directory, so it selected statement-only
coverage while its parent used branch coverage from `pyproject.toml`. Both CI
test commands and the local full-suite runner now pass the absolute TOML path.

An isolated Windows environment with those two test-tool versions reproduced
the collection ImportError. After the fixes, the five plotting cases and OCR
subprocess case pass together with two workers and coverage (6 passed). A direct
probe of pytest-cov 3.0's child startup confirmed branch=False with automatic
discovery outside the checkout and branch=True with the explicit config.

Hosted validation of fix commit `6f9c039f4601eacef5e080590dc570d102c4dffa`:
[macOS job 103436035446](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34651990565/job/103436035446)
passed all 342 tests with 157 warnings in 358.28 seconds, and completed coverage
aggregation successfully. The lower-dependency and installed-package workflows
also passed at this commit. Linux Python 3.11, 3.13 and 3.14 passed; Python 3.12
and final matrix reporting were still running when this evidence was recorded.

## Unexpected-dialog guard and CI diagnostics

The replacement Python 3.12 job remained in pytest after the other jobs finished;
its live log showed 144 completed tests (42%), but the terse output did not
identify active cases. The preceding Python 3.12 run passed all 342 cases in
371.35 seconds. An unexpected modal dialog is a hypothesis, not an established
cause of this wait.

Every test now guards the shared file-dialog mode callbacks: unexpected native
dialog requests raise a pytest failure with the test identity and arguments.
The intentional widget tests explicitly override the guard with their existing
deterministic replies. A regression checks that the guard rejects a dialog.
The focused widget and loader checks pass (6 tests). CI now logs individual
test names, the 20 slowest durations and stack dumps after two minutes, allowing
future waits to be distinguished from slow computation or blocked interaction.

## Hosted gate completed (2026-09-11)

Implementation commit `db79eaadeb043b52d1f4ff274fd3197ca6dce8c2` passed the
[full matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34653015465),
[lower-dependency job](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34653015413)
and [installed-distribution checks](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34653015414).
Coverage aggregation and test-result publishing also succeeded.

| Platform | Python | Passed | Warnings | Pytest seconds |
| -------- | ------ | ------ | -------- | -------------- |
| Linux    | 3.11   | 343    | 34       | 204.14         |
| Linux    | 3.12   | 343    | 134      | 837.76         |
| Linux    | 3.13   | 343    | 131      | 299.19         |
| Linux    | 3.14   | 343    | 131      | 185.30         |
| macOS    | 3.14   | 343    | 157      | 571.13         |

The Python 3.12 slowdown is selective, not a uniform multiplier:

| Test                   | Linux 3.11 seconds | Linux 3.12 seconds | Linux 3.13 seconds |
| ---------------------- | ------------------ | ------------------ | ------------------ |
| test_Base_Operators    | 18.03              | 176.07             | 22.64              |
| test_Properties        | 14.67              | 169.78             | 18.65              |
| test_groups_methods    | 7.29               | 85.09              | 9.67               |
| folder_operations.py   | 16.35              | 228.10             | 80.13              |
| test_attocube_scan     | 45.54              | 133.86             | 52.02              |
| test_outlier_detect    | 37.25              | 46.06              | 44.44              |
| image test_funcs       | 20.91              | 21.30              | 22.03              |

The sampled base-operator test was in the OVF loader's NumPy `genfromtxt` call;
another sample showed Attocube image/metadata reconstruction. Neither establishes
how long the sampled operation took. The `execnet` receiver frames are worker
communications, not evidence of a communication deadlock. Set aside the HDF5
locking hypothesis for this investigation; no observed trace implicates it.
No unexpected dialog failed the suite. All sampled tests eventually passed.

Both Linux 3.12 and 3.13 resolved NumPy 2.5.2, coverage 7.14.1 and pytest-cov
7.1.0, with interpreter-specific Conda builds. Those version numbers alone do
not explain the difference. The remaining performance follow-up is a controlled
comparison of the affected tests with and without coverage, then serial versus
two workers, retaining environment and runner details. No folder implementation
change is justified by these samples alone. Phase 5's compatibility gate is
complete; this performance anomaly remains unresolved.
