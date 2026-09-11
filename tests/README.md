# Running and selecting tests

Use a supported environment satisfying `tests/test-env.yml`; see `AGENTS.md` for
the Windows/Conda setup. Tesseract and its Python wrapper remain optional runtime
dependencies. The comprehensive test environment includes them to exercise OCR.
The OCR integration test reports a skip if either is unavailable; other image
tests and the tests of missing OCR dependencies still run.

The default `python -m pytest` selection includes every test. Registered markers
allow smaller selections without changing that default:

```powershell
python -m pytest -m "not network and not ocr"
python -m pytest -m "not slow"
python -m pytest -m "image and not ocr"
python -m pytest -m "plotting or gui"
python -m pytest -m documentation
python -m pytest -m ocr -rs
python -m pytest --collect-only --strict-markers
```

Primary categories follow the existing layout: `core`, `analysis`, `formats`,
`folders`, `image`, `plotting`, `documentation` and `tools`. Additional markers
overlap: `gui` identifies widget/dialog tests, `network` identifies URL-loading
tests and the URL-based fitting example, and `ocr` identifies the real Tesseract
integration. Tests of unavailable OCR dependencies do not require `ocr`.
The `slow` marker identifies the real-data Attocube interpolation test, observed
to exceed two minutes with the lower dependency combination. Timings depend on
the environment; this marker does not exclude the test from default or CI runs.

`tests/conftest.py` assigns layout categories; decorators identify additional
behaviour within mixed modules. Add a network marker to new tests that fetch
remote resources, and mark new network examples in the collection hook.

For headless Windows runs with logs, temporary directories, exit codes and fault
diagnostics, use the repository runner:

```powershell
./maintenance/run-baseline.ps1 -Check focused -TestPaths tests,-m,"not network and not ocr"
./maintenance/run-baseline.ps1 -Check serial
./maintenance/run-baseline.ps1 -Check parallel
```

Ordinary tests must run with `READTHEDOCS` absent from the environment. The runner
handles this automatically. Setting it to `False` still enables the package's
presence-based documentation restrictions.

## Isolation and warning policy

- Use `tmp_path` for outputs and `monkeypatch.chdir` for temporary directory
  changes. Treat scientific input fixtures and `doc/plot_cache` as read-only.
- Use fixtures to restore dialog mocks, options and plotting state. A test must
  not depend on a preceding test installing a mock or creating a figure.
- Do not change warning filters during module import. Plotting and filtering
  unit tests treat warnings as errors within their own scope.
- The plotting module ignores the exact Agg non-interactive-display warning and
  Pyparsing alias deprecations originating in Matplotlib's font/math parsers
  (needed for supported Matplotlib 3.8 with newer Pyparsing), including the exact
  `parseAll` warning reported inside Pyparsing's compatibility wrapper.
  Two tests that deliberately manipulate axes outside a GridSpec additionally
  filter that specific layout warning within their fixture scope.
- Numerical warnings, dependency deprecations and documentation-example warnings
  remain visible. Record and investigate them rather than suppressing whole
  categories to obtain a smaller warning count.

## Supported dependency boundaries

`minimum-env.yml` exercises Python 3.11 with the lower supported minor lines of
NumPy (2.0), SciPy (1.14), Matplotlib (3.8), scikit-image (0.24) and lmfit (1.3).
Other dependencies resolve normally, including pandas, which has no declared
minimum. This is a representative combination, not an exhaustive compatibility
matrix or a claim that every earliest patch release was tested. Keep its test
and optional dependencies aligned with `test-env.yml`.

The lower-dependency workflow runs the full suite separately from the ordinary
Python 3.11–3.14 CI matrix. Installed wheel/sdist probes cover Python 3.11 and
3.14, including styles and OCR resources outside the source checkout.

For a local prefix environment:

```powershell
./maintenance/run-baseline.ps1 -EnvironmentPrefix D:/path/to/environment -Check serial
./maintenance/run-baseline.ps1 -EnvironmentPrefix D:/path/to/environment -Check parallel
```
