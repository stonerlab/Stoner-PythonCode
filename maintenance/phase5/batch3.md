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
