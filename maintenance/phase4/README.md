# Phase 4: CI and quality tooling — 2026-09-10

Status: **Complete**.

## CI decisions

- The Linux matrix is the hosted compatibility gate for Python 3.11–3.14.
  Python 3.14 also runs on the supported Intel `macos-15-intel` runner.
- The macOS matrix entry adds PyQt6 explicitly because the file-dialog tests
  exercise the supported Qt implementation. The environment also requires
  conda-forge's current `libmagic >=5.48`: an unconstrained mixed-channel solve
  paired conda-forge `filemagic 1.6` with `libmagic 5.36` from `defaults`, whose
  magic database produced invalid character-range errors once PyQt6 was
  present. Keeping the wrapper and native library on the current
  conda-forge stack preserves Python 3.14 macOS coverage.
- MIME detection is optional at runtime. `get_mime_type()` now treats a native
  `MagicError` as unavailable MIME detection and returns `None`, allowing the
  existing filename-pattern loader search to continue just as it does when
  filemagic is not installed. A focused regression test covers this fallback.
- Windows is the primary development and local-test platform, so hosted CI is
  used to exercise the non-development platforms: Linux and macOS. Phase 0
  passed all 333 tests both serially and with two xdist workers on Windows 11
  and Python 3.14.
- A hosted Windows lane was tested on `windows-latest` and `windows-2022`.
  Micromamba repeatedly crashed with access violation `3221225477` while
  applying a successfully solved environment, including with `--always-copy`.
  This rules out package-cache hard links as the direct cause. A separate
  setup-miniconda/libmamba attempt spent more than seven minutes creating the
  environment without reaching pytest. The large conda-forge dependency graph
  makes this a slow, memory-intensive and unreliable hosted gate, so retaining
  it would create known failures and unhelpful notification noise.
- Test jobs use explicit Coveralls flags of
  `run-<python-version>-<runner>`. The finalisation job waits for the complete
  matrix and does not carry forward nonexistent or missing jobs.
- Coveralls uploads run on the four Linux jobs. Its macOS action currently
  installs through a Homebrew tap rejected as untrusted by the hosted runner;
  the redundant macOS upload is skipped so this reporting transport cannot
  overturn a successful 334-test platform result.
- Codacy coverage uses the official action pinned to the commit behind v1.3.0.
  Pull requests skip this upload because repository secrets are unavailable to
  untrusted forks. Codacy's hosted static analysis remains the repository's
  active code-quality service.
- All third-party actions are pinned to full commit hashes. Workflow token
  permissions default to read-only; only the test-result publisher receives
  `checks: write`.
- The repository-level Actions default is `read`, and workflows cannot approve
  pull requests. This was read back from the GitHub API after updating it.
- Documentation CI consumes the intentional plot cache with
  `READTHEDOCS=True` and no longer commits or pushes generated files. Cache
  refreshes use `make -C doc refresh-plot-cache` and enter the repository
  through normal review.
- The Travis webhook is inactive and GitHub Actions has replaced its Python
  matrix, coverage and test-result roles, so `.travis.yml` is removed.

## Quality-tool status

| Tool | Status |
| --- | --- |
| Codacy | Required hosted static analysis and coverage reporting |
| Coveralls | Required parallel coverage reporting |
| pytest/coverage | Required local and CI test gate |
| Prospector | Obsolete; its profile was absent and its Makefile target was removed |
| Pylint | Not a repository gate; existing inline annotations remain historical/local guidance |
| Bandit | Not configured as a repository gate |
| Black | Optional local formatter; retained Makefile helper, not a CI requirement |

Removing Prospector does not remove a working gate: no workflow invoked it,
its referenced `.landscape.yml` profile did not exist, and its generated report
was empty. Codacy continues the hosted static-analysis role.

## Local reproduction

Run these commands from the repository root with the `py314` environment used
for the maintenance baseline:

```powershell
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python maintenance\check-ci-config.py
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python maintenance\check-package-contents.py
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python -m pytest -n 2 --cov-report= --cov=Stoner --junitxml pytest.xml
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 coverage xml
$env:READTHEDOCS = "True"
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python -m sphinx -b html -E doc doc\_build\html
```

The Conda and PyPI deployment workflow is release-only and publishes external
artefacts, so it has no non-publishing local equivalent. Phase 2's package
validation workflow supplies the build, archive and clean-install checks
without publishing.

Both added OCR packages were found on the approved `phygbu`/`conda-forge`
channels. Cross-platform dry-solves repeated the previously recorded slow
solver behaviour and were stopped without a success or conflict result; the
five GitHub Actions jobs are the authoritative hosted validation. Phase 0's
Windows results provide the current Windows validation baseline.

## Remote validation

Implementation commit `99843b7a5dedcd82740ced076138383ab8fec0a3`
passed all required hosted checks:

- [pytest run 34536293951](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536293951): Linux on Python 3.11, 3.12, 3.13 and 3.14; Intel macOS on Python 3.14; test-result publication; and Coveralls finalisation.
- [Package validation run 34536293816](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536293816): source and wheel builds, content checks and clean-install probes.
- [Documentation run 34536324371](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536324371): complete documentation build using the retained plot cache.

The required workflows are green, local equivalents are documented above,
workflow permissions are least privilege, and retired Travis/Prospector files
no longer represent inactive or duplicate gates. These results satisfy the
Phase 4 completion criteria.
