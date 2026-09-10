# Phase 4: CI and quality tooling — 2026-09-10

Status: **In progress pending remote validation**.

## CI decisions

- The established Linux matrix remains the compatibility gate for Python
  3.11–3.14. Python 3.14 also runs on `windows-latest` and the supported Intel
  `macos-15-intel` runner, giving every advertised operating-system family a
  current execution check without multiplying the complete version matrix.
- Micromamba crashed with access violation `3221225477` while linking the
  solved Windows environment on both `windows-latest` and `windows-2022`.
  The Windows matrix entry therefore passes `--always-copy`, avoiding hard
  links between the package cache and environment while retaining the faster
  micromamba solver. Linux and macOS retain the default hard-link behaviour.
- Test jobs use explicit Coveralls flags of
  `run-<python-version>-<runner>`. The finalisation job waits for the complete
  matrix and does not carry forward nonexistent or missing jobs.
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
six GitHub Actions jobs are the authoritative platform validation.
