# Intermittent Python 3.12 CI worker loss

Investigated 2026-09-12. No loader behaviour changed. The separately approved
CI coverage policy change is recorded below.

## Hosted evidence

[Failed run 34705333723](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34705333723)
tested commit `fb21b974fc7aee830b7c3f70612ab9bbc59d30a7`. Only Ubuntu Python 3.12
failed: worker `gw1` disappeared during `test_attocube_scan`, without a Python
assertion traceback or a recorded exit signal. The result was 347 passed and
one failed in 846.13 seconds. Its last partial diagnostic mentioned
`Stoner/tools/tests.py` at an implausible line number (32680); that incomplete
trace is not sufficient to locate a defect.

The test started at 16:33:08 UTC, the worker was reported lost at 16:34:44,
and pytest continued until 16:43:15. This does not indicate a GitHub job time
limit terminating the test. The configured `faulthandler_timeout=120` dumps
stacks; pytest's `faulthandler_exit_on_timeout` defaults to false and is not
enabled by this repository. There is no recorded OOM, kill signal or fatal
native exception in the available log. Memory pressure and native/runtime
failure remain hypotheses, not established causes.

Three later pytest runs succeeded: `34705673074`, `34705726713` and
[34706013334](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34706013334).
In the last run, Python 3.12 passed all 348 tests in 656.57 seconds, with
Attocube taking 98.68 seconds. Other tests still emitted two-minute diagnostic
dumps. Thus the slow-run symptom persists independently of worker loss.
The failed and latest successful jobs both used Python 3.12.14, NumPy 2.5.2
and coverage 7.14.1. The Attocube loader, its test and the pytest workflow did
not change between those commits.

Downloaded evidence is retained under ignored `maintenance/runs`:

- `phase6-ci-34705333723-failed.log`
- `phase6-ci-34705333723-junit/pytest.xml`
- `phase7-ci-success-34706013334.log`

## Isolation review and local probes

The Attocube test reads the retained real scan fixture and writes HDF5/TIFF
round trips under its pytest temporary directory. The concurrent folder test
does not use those output paths. No shared output-file collision was identified.
The final Attocube operations include image fitting and cubic regridding;
the log does not identify which operation was active when the worker vanished.

Windows Python 3.14 probes passed:

- Attocube alone: 31.19 seconds call time, 36.90 seconds total;
  `20260912-185822-focused-1d6c41e3`.
- Attocube alongside `test_Base_Operators`, two workers with branch coverage:
  two passed in 42.52 seconds; Attocube 31.83 seconds;
  `20260912-185939-focused-d9cdcdc0`.

A named shared `py312` environment was created with mamba from
`tests/test-env.yml`, changing only its Python constraint to 3.12. It contains
Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, coverage 7.16.0 and h5py 3.16.0.
This is not a byte-for-byte replica of the Linux job's environment.
Attocube alone passed in 45.29 seconds (37.93 seconds call time), recorded in
`20260912-190254-focused-6f395167`.

The Python 3.12 two-worker run with branch coverage also passed, but took
285.36 seconds: Attocube 196.76 seconds and the folder test 270.95 seconds
(`20260912-190423-focused-9f299512`). Both diagnostic dumps showed active
`genfromtxt` conversion loops, not a dialog or an HDF5 lock wait.

The same Python 3.12 pair with two workers and no coverage passed in 45.70
seconds: Attocube 37.57 seconds and the folder test 20.24 seconds
(`20260912-190926-focused-b594bbce`). The same tests, environment and worker
count therefore took about 6.2 times longer with coverage. This isolates
coverage overhead as a cause of the observed slowdown, without reproducing
worker loss.

A separate 1,000-row, ten-column `genfromtxt(StringIO(text))` probe compared
plain execution with `coverage.Coverage(source=['Stoner'], branch=True,
data_file=None)` started after imports. Python 3.12 measured 0.0058 seconds
plain and 0.3106 seconds under coverage; Python 3.14 measured 0.0050 and
0.0063 seconds. These single-run timings are indicative, not a benchmark,
but isolate a large tracing overhead even for excluded NumPy code. The
reproduction script is `maintenance/runs/phase7-genfromtxt-timing.py`.

The effect is consistent with the upstream
[Python 3.12 tracing slowdown report](https://github.com/python/cpython/issues/107674).
This correspondence is a hypothesis about the underlying runtime mechanism;
it is not proof of the cause of the intermittent worker death.

## Disposition

Keep this as an intermittent worker-loss investigation, not a confirmed
Attocube loader defect. Do not increase timeouts, skip the test, or change
scientific algorithms based on this log. If it recurs, capture worker exit
status and runner memory/kernel OOM evidence to distinguish termination from
a native crash. Local passes cannot resolve that remote evidence gap.

## Approved coverage policy change

The maintainer approved collecting coverage only on Ubuntu Python 3.14.
`.github/workflows/run-tests-action.yaml` now marks that matrix entry for
coverage, adds coverage arguments and generates XML only there, and gates
both Coveralls and Codacy uploads on the same marker. The obsolete Coveralls
parallel-finish job was removed because only one job submits coverage.

All five Python/OS matrix entries still run the complete test suite with two
workers, retain the diagnostic timeout, and upload JUnit results. Codacy still
does not receive pull-request uploads. This reduces demonstrated tracing
overhead; it is not a claimed fix for the intermittent worker death.

Local validation parsed the YAML and expanded the matrix to confirm five
jobs and exactly one coverage entry. Six Bash checks exercised Linux with
and without coverage and macOS, with both passing and failing pytest exit
codes. Coverage arguments and XML generation appeared only when enabled;
all cases preserved the pytest status and JUnit arguments. Evidence:
`maintenance/runs/phase7-coverage-workflow-check/result.json` and reproduction
script `maintenance/runs/check-coverage-workflow.py`.

`git -c core.whitespace=cr-at-eol diff --check` passes. Hosted workflow
execution and external coverage uploads remain unverified until committed
and pushed. Earlier local runtime tests are recorded above; no runtime code
changed in this policy batch.
