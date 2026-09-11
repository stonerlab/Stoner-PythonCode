# Stoner maintenance and cleanup plan

This document is the working plan for bringing the Stoner repository onto a clean, reproducible maintenance footing without changing its scientific behaviour accidentally. Update the status and evidence in this file as each phase is completed.

## Objectives

1. Establish a trustworthy Python 3.11-3.14 development and test baseline.
2. Remove ambiguity from Git status, packaging metadata, dependencies, and release artefacts.
3. Separate source and required scientific fixtures from disposable generated content.
4. Modernise validation and documentation incrementally.
5. Make later code cleanup safer by preserving the public data, metadata, loader, and chaining contracts.

## Current baseline

Recorded from the `stable` branch at `08ad42f09` on 2026-09-09:

- `stable` is aligned with `origin/stable`.
- Source version is `0.11.4`; checked-in `dist` artefacts are version `0.10.8`.
- `pyproject.toml` requires Python 3.11 or newer; GitHub Actions tests 3.11-3.14.
- This machine's modern Conda installation is `C:\ProgramData\miniforge3` (Python 3.14.7); other development machines may use `C:\ProgramData\Anaconda3`. Either is suitable when it provides Python 3.11 or newer and the required dependencies.
- `C:\ProgramData\Miniconda3` is intentionally retained at Python 3.6 for LabVIEW 2018. It must be ignored for Stoner development and is out of scope for maintenance.
- The Miniforge ``py314`` environment provides the test and documentation toolchain and is the preferred local maintenance environment; use ``py313`` as the fallback when present.
- The Cygwin-created checkout appears broadly modified to Windows Git because of executable-mode and line-ending differences. Do not treat this as a semantic 691-file change.
- The repository contains about 29,700 lines of package Python, 30 test modules, 78 documentation examples, and 98 sample-data files.

## Status

| Phase | Workstream                 | Status      | Completion evidence                                                                                                                           |
| ----- | -------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Reproducible baseline      | Complete    | maintenance/phase0: 333 passed serial and with two workers; identical 79% coverage; environment manifest and Sphinx warning baseline recorded |
| 1     | Git and checkout hygiene   | Complete    | Approved Windows Git workflow; local fileMode=false/autocrlf=false; disposable clone clean and semantic diff unchanged                        |
| 2     | Packaging and metadata     | Complete    | GitHub run 34519864056: wheel/sdist builds, archive contents and separate clean-install probes passed on Python 3.14/Linux                    |
| 3     | Repository content cleanup | Complete    | `maintenance/phase3` inventory and policy check; 1.11 MB of reviewed output removed; plot cache and scientific fixtures retained              |
| 4     | CI and quality tooling     | Complete    | Linux Python 3.11–3.14 and macOS Python 3.14 green; Windows covered by the Phase 0 local baseline                                             |
| 5     | Tests and compatibility    | Complete    | 343 passed in all five hosted matrix jobs; lower dependencies and installed distributions passed; 3.12 slowdown recorded separately           |
| 6     | Documentation and examples | In progress | Fresh RTD build: warnings 1658 to 470; five primary classes and 85 dynamic Data methods audited; plot cache unchanged                         |
| 7     | Focused source maintenance | Not started | Small reviewed batches with regression tests                                                                                                  |
| 8     | Release readiness          | Not started | Clean-room package and release checklist                                                                                                      |

Statuses should be one of `Not started`, `In progress`, `Blocked`, or `Complete`. Add dated notes and commands beneath a phase when work begins.

## Phase 0: establish a reproducible baseline

### Tasks

- Detect a supported Conda installation (`C:\ProgramData\miniforge3` or `C:\ProgramData\Anaconda3`). Prefer an existing `py314` or `py313` environment after checking every requirement in `tests/test-env.yml`; otherwise create a dedicated environment from that manifest.
- Install the checkout without resolving a second, conflicting dependency set.
- Record Python, Conda, NumPy, SciPy, Matplotlib, scikit-image, lmfit, pytest, and platform versions.
- Run `pytest --collect-only` and record the actual collected test count.
- Run focused smoke tests for `Data`, format loading, plotting, folders, and images.
- Run the complete suite without parallelism first, then compare with the CI-style parallel run.
- Record failures, warnings, skips, duration, and coverage as baseline facts. Do not fix failures in the baseline commit.
- Build the Sphinx documentation once and record warnings separately from test failures.

### Suggested commands

```powershell
C:\ProgramData\miniforge3\Scripts\conda.exe env create -f tests\test-env.yml
C:\ProgramData\miniforge3\Scripts\conda.exe run -n test-environment python -m pip install --no-deps -e .
C:\ProgramData\miniforge3\Scripts\conda.exe run -n test-environment python -m pytest --collect-only -q
C:\ProgramData\miniforge3\Scripts\conda.exe run -n test-environment python -m pytest
```

On Anaconda-based machines, substitute `C:\ProgramData\Anaconda3\Scripts\conda.exe` in these commands. If the named environment already exists, inspect it before deciding whether to update it in place or recreate it. Do not use or alter the LabVIEW Python 3.6 Miniconda installation.

### Completion criteria

- A new contributor can reproduce the environment from repository files.
- Full-suite results and known environmental limitations are recorded here.
- Maintenance work has a reliable before/after comparison.

## Phase 1: Git and checkout hygiene

### Tasks

- Compare `git status`, file modes, attributes, and line endings using the Cygwin Git client that created the checkout and the Windows Git client used by Codex.
- Determine whether all reported changes are mode/line-ending noise or whether semantic changes are mixed in.
- Choose and document the supported checkout/commit workflow for Windows and Cygwin contributors.
- Review `.gitattributes` and repository-local Git configuration. Add explicit text/binary and line-ending rules only after testing them on a disposable clone or worktree.
- Ensure any normalization is isolated in its own reviewed commit, with no source changes.
- Add ignore rules for genuinely local caches only after checking whether they are already tracked.

### Completion criteria

- The agreed Windows Git workflow shows a comprehensible semantic status. Cygwin contributors use a separate checkout; live Cygwin verification remains unavailable on this machine.
- No repository-wide permission or line-ending churn appears in ordinary feature commits.
- The normalization procedure, if needed, is reversible and documented.

## Phase 2: packaging, dependencies, and metadata

### Tasks

- Treat `pyproject.toml` as the primary project metadata and dependency declaration unless a different source of truth is explicitly chosen.
- Reconcile runtime dependencies across:
  - `pyproject.toml`
  - `requirements.txt`
  - `recipe/meta.yaml`
  - `recipe/build-env.yml`
  - `tests/test-env.yml`
  - `doc/docs-env.yml`
  - `doc/requirements.txt`
- Keep Conda recipes and environments as pure Conda specifications as practical. Check `phygbu` and `conda-forge` before using pip, and report a pip-only requirement as a packaging defect.
- Remove or explain duplicate/ambiguous entries such as `python-dateutil` and `dateutil`.
- Classify dependencies as required runtime, optional runtime, test-only, documentation-only, or build-only.
- Keep the confirmed GPLv3 licence consistent in `pyproject.toml`, `LICENSE.md`, `COPYING`, and the Conda recipe. (Completed 2026-09-09.)
- Standardise repository URLs on the current `stonerlab/Stoner-PythonCode` location, retaining redirects only where required. (Completed for searchable source files 2026-09-09.)
- Review supported-Python wording and classifiers so README, package metadata, Conda recipe, documentation, and CI agree.
- Verify package-data configuration for:
  - `Stoner/plot/stylelib/*.mplstyle`
  - `Stoner/Image/tessdata/*`
- Build fresh wheel and sdist artefacts from a clean tree. Inspect their contents rather than relying on the old `dist` directory.
- Install each artefact into a clean environment and run import, loader, plotting-style, and image-resource smoke tests.

### Completion criteria

- Dependency and licence declarations are consistent or their intentional differences are documented.
- Wheel and sdist contain all required non-Python resources and exclude repository-only material.
- Clean installations pass the defined smoke tests.

## Phase 3: repository content cleanup

Do this only after Phase 0 supplies a regression baseline and Phase 2 identifies what packaging actually needs.

### Inventory candidates

- `build/`
- `dist/`
- `.eggs/`
- `.pytest_cache/`
- `.ipynb_checkpoints/`
- `.idea/`
- `.spyproject/`
- Python bytecode caches
- `Stoner.egg-info/`
- generated Sphinx output under `doc/_build/`
- generated API/output material under `doc/pypi-docs/`
- `prospector-report.txt`
- legacy `.travis.yml`
- generated `doc/plot_cache/` images and PDFs
- compiled user-guide outputs such as DVI/PDF files
- test-created Dask lock files and working data

### Tasks

- Produce a tracked-file inventory with size, provenance, reproducibility, and consumer for every candidate category.
- Mark each category as:
  - required source or fixture;
  - intentional generated cache;
  - release artefact;
  - local-only output;
  - obsolete.
- Keep real instrument and facility files that provide unique loader coverage, even when large.
- For large scientific fixtures, document which test or example consumes each file before considering Git LFS, external storage, or removal.
- Decide whether documentation plot caches are intentionally versioned to make builds deterministic. If retained, document their refresh workflow.
- Retain `doc/plot_cache`: these generated graphics are an intentional cache that avoids repeatedly executing expensive examples during documentation builds.
- Remove only reviewed categories, one cohesive batch at a time, and add narrowly scoped ignore rules.
- Measure repository checkout size and package artefact size before and after cleanup.

### Completion criteria

- Every retained generated or large file has a documented reason.
- Local test/build activity does not pollute `git status`.
- Cleanup does not reduce format or documentation regression coverage.

### Confirmed repository policy

- `doc/plot_cache` is intentionally tracked and must be retained. It avoids expensive regeneration of documentation graphics.
- Local `dist` artefacts are not retained; release distributions are built by GitHub Actions.

## Phase 4: CI and quality tooling

### Tasks

- Verify the test workflow on Python 3.11-3.14 and decide whether all supported operating systems need CI coverage.
- Check that test-result, coverage, Coveralls, and Codacy job names and parallel flags correspond correctly.
- Review the coverage-finalisation `carryforward` values against the flags actually emitted by the matrix.
- Replace network-executed installer scripts such as `curl | bash` where a pinned or verified alternative is available.
- Review action pinning and permissions, keeping workflow permissions minimal.
- Decide whether documentation builds should commit generated plot caches directly to the source branch.
- Remove Travis configuration only after confirming it is no longer used.
- Consolidate active lint/static-analysis configuration. Identify whether Prospector, Pylint, Bandit, Codacy, and formatting commands remain supported.
- Provide local commands equivalent to each required CI check.

### Completion criteria

- Required workflows are green and use least-privilege permissions.
- Each CI check has a documented local reproduction command.
- Obsolete tooling is removed without silently dropping a quality gate.

## Phase 5: tests and supported-version compatibility

### Tasks

- Categorise tests as core, format, image, plotting/GUI, documentation example, slow, and external-resource.
- Make tests independent of execution order and clean up temporary files, Dask workers, figures, and GUI objects locally.
- Replace writes into tracked fixture directories with pytest temporary directories where behaviour permits.
- Add explicit tests for package resources installed from a wheel, not just imports from the source checkout.
- Add regression tests for each confirmed maintenance bug before changing implementation.
- Audit warnings and apply targeted fixes or narrow filters; do not globally hide actionable warnings.
- Compare serial and parallel results and document tests that cannot safely use `pytest-xdist`.
- Validate supported NumPy, SciPy, Matplotlib, scikit-image, pandas, and Python boundaries rather than only the newest environment.

### Completion criteria

- The suite is repeatable locally and in CI.
- Temporary outputs do not modify tracked paths.
- Supported Python versions have explicit, recorded results.

## Phase 6: documentation and examples

### Tasks

- Correct stale version, branch, repository, installation, and compatibility statements.
- Fix obvious spelling and cross-reference errors without rewriting established technical meaning.
- Verify that all public top-level classes and dynamically attached `Data` methods appear in the API documentation.
- Run every example covered by `tests/test_doc_samples.py` and identify any examples excluded from automated execution.
- Separate historical generated manuals from the definitive reStructuredText sources.
- Decide which compiled PDFs and plot outputs are release deliverables versus source-controlled caches.
- Add a short contributor section pointing to `AGENTS.md`, this plan, environment setup, tests, and documentation commands.

### Completion criteria

- Sphinx builds without unexpected warnings.
- User guide, README, package metadata, and current behaviour agree.
- Examples execute against the supported stable API.

## Phase 7: focused source maintenance

This phase is deliberately after the baseline and repository cleanup. Avoid broad stylistic rewrites.

### Candidate workstreams

- Resolve documented `TODO`/`FIXME` cases, including MAXIMUS single-region assumptions, the metadata-copy workaround, and incomplete column-indexing behaviour.
- Review dynamic method binding for discoverability, typing, and API documentation while preserving the public chained-operation contract.
- Improve type annotations module by module, beginning at stable public boundaries rather than internal implementation details.
- Reduce duplicated loader/saver logic without changing positive-identification or priority semantics.
- Audit exception handling around optional dependencies and malformed scientific files.
- Review legacy compatibility shims only after tests establish which are still exercised.
- Break unusually large modules or functions into cohesive helpers only where this materially improves comprehension and testability.

### Batch rules

- One behavioural concern per batch.
- Add or identify regression coverage first.
- Preserve metadata, masks, column roles, filenames, loader selection, and return/chaining behaviour.
- Use real sample formats where a defect is format-specific.
- Record the focused and full-suite results in the commit or pull-request description.

### Completion criteria

- Each refactor has a clear behavioural invariant and regression evidence.
- No broad cleanup commit mixes formatting, generated outputs, dependencies, and runtime changes.

## Phase 8: release readiness

### Tasks

- Run tests and documentation from a clean clone using the documented environment.
- Build wheel, sdist, and Conda package without relying on stale local outputs.
- Inspect archive contents, metadata, licences, dependency declarations, and package size.
- Install each package into clean environments and run smoke tests against installed resources and representative sample data.
- Confirm the version source and release/tag process.
- Verify PyPI, Conda, Read the Docs, coverage, and citation/DOI links.
- Produce release notes separating user-visible changes, compatibility changes, fixes, and repository-only maintenance.

### Completion criteria

- Release artefacts are reproducible and self-contained.
- All supported channels expose consistent version, licence, URL, and dependency metadata.
- Any remaining hardware-, platform-, or facility-specific validation boundaries are stated precisely.

## Decisions requiring explicit agreement

Record decisions here before implementing changes that affect repository history, release contents, or supported users:

- Canonical Git client and line-ending/file-mode policy: agreed 2026-09-10, Windows Git for this checkout with repository-local `core.fileMode=false` and `core.autocrlf=false`; preserve existing mixed line endings and tracked executable bits. No repository-wide normalization.
- Whether generated documentation and plot caches remain tracked.
- Whether old distribution artefacts are retained, moved to release storage, or removed.
- Whether large fixtures remain in Git, move to Git LFS, or are replaced by smaller representative fixtures.
- Supported operating systems and dependency-version ranges.
- Whether all current runtime dependencies are mandatory or some become optional extras.
- Whether legacy scripts and compatibility shims remain supported public material.

## Work log

Add concise dated entries as work proceeds. Each entry should identify the phase, files changed, validation performed, and any remaining boundary.

- 2026-09-09: Initial repository survey completed; maintenance plan created. No cleanup or runtime changes performed. Full test baseline remains outstanding because the visible modern Miniforge environments do not yet include pytest.
- 2026-09-09: Audited runtime imports and separated required dependencies from guarded optional features and test/documentation extras. Reconciled PyPI and Conda manifests, corrected GPLv3 metadata, standardised repository URLs, retained `doc/plot_cache`, and removed stale local `dist` artefacts. A completed Conda solve, clean installation checks, and the full test baseline remain outstanding.
- 2026-09-09: Built `Stoner 0.11.4` successfully through the PEP 517 interface into a temporary directory. The wheel contained `COPYING`, `LICENSE.md`, `Stoner/plot/stylelib`, and `Stoner/Image/tessdata`. It also contained tests and pre-existing `__pycache__` bytecode; excluding those from release wheels remains a packaging-hygiene task. No local distribution artefact was retained.
- 2026-09-10: Began Phase 6 documentation cleanup. Updated the user guide to the current loader registry and canonical class locations, corrected British-English spelling and stale API names, and fixed malformed RST. RTD-mode builds now use the retained plot cache and import the checkout rather than an installed Stoner package; local builds regenerate and refresh cached plots. A clean ``py314`` HTML build succeeds with no unreadable images and no external ``class_modifier`` wrapper warnings. Rendered unresolved references in the narrative user-guide pages fell from 233 to zero; 72 unresolved displays remain in the generated API and fitting indexes. Raw Sphinx warnings remain dominated by autosummary, third-party docstrings and inherited-docstring duplication and are not yet a useful zero-warning gate.
- 2026-09-10: Ran the utility and documentation-example tests in ``py314``: all 78 passed, comprising five focused utility tests and all 73 examples. An earlier isolated run made ``STXMIMage_Demo.py`` fail while resolving the dynamically attached ``translation_limits`` method, but the complete repeat passed and no file-dialog hang occurred. Treat this as a possible order/state-sensitive warning to watch rather than a currently reproducible failure.

## Dependency audit evidence (2026-09-09)

The current dependency set is based on imports in the present Python 3 code, not on the accumulated historical manifests from earlier Python 2.7-era releases.

### Required runtime imports

`asteval`, `chardet`, `h5py`, `lmfit`, `looseversion`, `matplotlib`, `multiprocess`, `numpy`, `packaging`, `pandas`, `Pillow`, `python-dateutil`, `scikit-image`, `scipy`, and `statsmodels` are imported unconditionally by reachable package modules.

Although a later concurrency migration was expected, the current `stable`, `origin/main`, and `origin/devel` trees still import and use `multiprocess.Pool` in `Stoner/folders/utils.py`; it remains required until that implementation is changed and tested separately.

### Guarded optional features

- Pretty representations: `tabulate`.
- MIME detection: `filemagic` (the imported module is `magic`).
- TDMS loading: `npTDMS` (the imported module is `nptdms`).
- Image alignment: `imreg_dft`, `image-registration`, or OpenCV.
- Plot styles and 3D visualisation: `seaborn` and `mayavi`.
- Specialist image/facility formats: `fabio`, `rosettasciio`, and HyperSpy.
- Acceleration, OCR, and dialogs: `numba`, `pytesseract`, and PyQt.

### Test and documentation-only packages

The test environment additionally includes pytest/coverage tools and optional packages exercised by format, image-alignment, and documentation-example tests. The documentation environment includes Sphinx and the packages needed to import Stoner and regenerate examples whose cached graphics are missing; it intentionally omits the complete specialist-format stack.

### Removed drift

Direct declarations of `memoization`, `configobj`, `urllib3`, and `dill` were removed because the current code does not import them. `dill` may still be installed transitively by `multiprocess`. `cycler` and `matplotlib-scalebar` were moved from runtime declarations to the test/documentation environments where they are imported by examples.

The uncommon dependencies checked are available as Conda packages from `phygbu` and/or `conda-forge`. No pip-only dependency defect was found in this audit; any future pip-only result must be reported rather than hidden in a mixed Conda/pip environment. A local dry-run of the complete Windows test environment loaded all channel metadata without reporting a missing package, but its solve remained unusually slow and was stopped without a success or conflict result.

## Phase 0 and 1 completion evidence (2026-09-10)

- Phase 0 used the preferred existing Miniforge `py314` environment, verified against every constraint in `tests/test-env.yml`. Editable installation used `--no-deps --no-build-isolation`; package versions, exact Conda builds, collection output and test reports are recorded in `maintenance/phase0`.
- Collection: 333 tests. Focused smoke: 124 passed, one arbitrary-loader failure, 139 warnings, 153.41 s. Full serial: 333 passed, 612 warnings, 448.47 s. Two workers: 333 passed, 613 warnings, 265.50 s. Both full runs have identical 79% combined coverage (82.49% lines, 70.59% branches), with no skips.
- The arbitrary-loader case passes in both full runs but fails in the smaller smoke selection. Carry this possible import/registry order dependency into phase 5. Initial pytest temporary-root access errors were resolved using new per-run directories, without changing tests or package code.
- RTD-mode Sphinx build succeeded with 1659 warnings. It generated three previously missing `stitch_int_overlap` cache files, retained according to repository policy. Warning counts are an observed baseline, not a zero-warning quality gate.
- `maintenance/run-baseline.ps1` reproduces collection, smoke, serial, parallel and documentation checks using isolated report and temporary directories. Its collection mode was executed successfully. The exact environment export is available, but fresh environment creation and Python 3.11-3.13/Linux validation remain separate compatibility work.
- Phase 1: the user approved Windows Git with repository-local `core.fileMode=false` and `core.autocrlf=false`. A disposable clone was clean under these settings; the actual checkout's semantic diff hash was identical before and after applying them. No source, tracked permissions or line endings were normalized. One pre-existing CRLF/LF-only difference remains in `Stoner/formats/utils/__init__.py`.
- `maintenance/.gitignore` excludes only this workflow's generated logs, coverage, temporary directories and large patch snapshots. Reports and environment manifests remain reviewable. Existing source/documentation work was preserved; no commit, push or release was performed.
- Next: resume phase 2 package contents and clean-install validation, before phase 3 removal of generated repository content. Release distributions remain a GitHub Actions responsibility.
### Phase 2 package-content progress (2026-09-10)

Corrected package-data ownership for all seven Matplotlib styles and three OCR resources; disabled implicit namespace discovery and manifest-based wheel data inclusion. The source manifest now excludes bytecode, pytest bytecode temporary files and `doc/_build`, while retaining tests, scientific fixtures and the intentional plot cache. `maintenance/check-package-contents.py` verifies setuptools' resolved inputs and a fresh temporary source manifest; all checks pass. `Stoner.tools.tests` remains an intentional package helper. See `maintenance/phase2/README.md` and `package-contents.json`.

Phase 2 remains in progress: current wheel/sdist builds and clean-install checks require non-publishing GitHub Actions validation of the committed changes under the agreed distribution policy. Existing release workflows were not dispatched because they publish to PyPI/Conda. Do not begin phase 3 removals before these packaging checks are completed. Existing setuptools licence-syntax deprecations remain recorded; no licence change was made.
Prepared `.github/workflows/check-packages.yaml` as a non-publishing phase 2 gate. It builds wheel/sdist, validates archive contents, and installs each into a separate new environment for loader, style, OCR-resource and image round-trip checks outside the source checkout. Workflow/Python syntax and the source-level probe passed locally. Remote execution awaits committing/pushing the accumulated maintenance changes; no CI or clean-installed-artifact success is claimed.
### Phase 2 CI completion (2026-09-10)

Committed the accumulated maintenance changes as `7d84956c5ede5617697cc9a7ed454ff9328a42f5` and pushed branch `codex/maintenance-baseline` with maintainer approval. [Package validation run 34519864056](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34519864056) completed successfully: built `stoner-0.11.4.tar.gz` and `stoner-0.11.4-py3-none-any.whl`, passed both archive-content checks, and passed both fresh installation probes outside the checkout on Python 3.14/Linux. No distribution was published. This supersedes the pending-CI boundary above and completes phase 2.

The existing Python 3.11-3.14 pytest matrix is running separately in [run 34519863932](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34519863932); package-validation success does not imply that matrix has passed. Phase 3 can now begin with the tracked-file inventory and classification required by its task list, while preserving the intentional plot cache and scientific fixtures.

The only pre-existing change excluded from the commit is the LF/CRLF-only difference in `Stoner/formats/utils/__init__.py`. SSH pushes on this machine required a per-command empty configuration because the user's existing SSH configuration requests unsupported `ssh-dss`; no global SSH configuration was changed.

- 2026-09-10 (Phase 6): Recorded the derived docstring conventions as the expected standard in DOCSTRING_STYLE.md and linked it from AGENTS.md. Standardised previously usual formatting and section choices, and required British English prose while preserving exact API names and literals. Updated the existing Stoner and doc/samples docstrings to use canonical headings, corrected malformed or inaccurate documentation, and made no runtime changes.

### Phase 3 completion (2026-09-10)

Created `maintenance/phase3/inventory.json` and `README.md` to classify every cleanup candidate and record the provenance, reproducibility and consumers of all retained scientific and image fixtures of at least 500 kB. The tracked checkout was 94,307,875 bytes before cleanup. Removed 1,110,307 bytes of reviewed local or reproducible output: legacy eggs, notebook and IDE metadata, an obsolete documentation redirect, an empty Prospector report, Dask worker locks, and compiled CHM/DVI/PDF documentation. The retained checkout is 93,197,568 bytes before adding the Phase 3 audit files.

Added root `.gitignore` entries only for local build/test/editor output and reproducible documentation products. `doc/plot_cache` remains tracked as the intentional 18.28 MB deterministic documentation cache. `.travis.yml` remains until Phase 4 makes its explicit CI retirement decision. `maintenance/check-repository-content.py` verifies these policies without modifying the checkout.

The Phase 3 policy check passed with all 243 cached plot files retained, and the Phase 2 package-content check continued to retain all scientific fixtures. The representative format/folder/image selection collected 72 cases after cleanup, matching its pre-cleanup collection; this checkout's pytest runner again exited without a final result summary after beginning execution, an existing Phase 5 test-runner investigation rather than evidence of a cleanup regression. Phase 0's full 333-test baseline remains the completed runtime baseline.

### Phase 4 completion (2026-09-10)

Validated implementation commit `99843b7a5dedcd82740ced076138383ab8fec0a3`. [pytest run 34536293951](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536293951) passed the required Linux Python 3.11–3.14 matrix, Intel macOS Python 3.14, test-result publication, and Coveralls finalisation. [Package validation run 34536293816](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536293816) and [documentation run 34536324371](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34536324371) also passed at the same commit.

Windows remains the primary development and local-test platform, with Phase 0's serial and parallel 333-test Python 3.14 results as its current evidence. Hosted Windows environment creation was removed after repeatable micromamba access violations on two runner versions, including with `--always-copy`, and an impractically slow setup-miniconda/libmamba attempt. Hosted CI therefore checks the non-development Linux and macOS platforms without retaining a known-red notification source.

The macOS environment now uses PyQt6 and a current conda-forge `libmagic >=5.48`. This replaced a mixed-channel pairing of conda-forge `filemagic 1.6` with `defaults` `libmagic 5.36` whose magic database failed after Qt initialisation. The current stack passed the complete macOS suite. Runtime `MagicError` failures now degrade to filename-pattern loader selection, matching the existing behaviour when filemagic is unavailable. Coveralls uploads run on Linux because its macOS Homebrew tap is rejected by the hosted runner; macOS pytest remains required and passed independently.

### Phase 5 first batch complete (2026-09-11)

Isolated working-directory changes and figure cleanup, moved image/HDF5/core metadata round-trips into pytest temporary directories, and made arbitrary-loader import and registry cleanup independent of preceding tests. Fixed two reproduced package defects: `clear_routine` removed entries from the wrong registry index, and ZIP loading requested append access to input fixtures. Added three regression cases using isolated registry entries and the real ZIP fixture.

Corrected `maintenance/run-baseline.ps1` to remove `READTHEDOCS` for ordinary tests: its presence, even with the value `False`, disabled external image methods. Added focused selections, Python fault reporting and isolated pytest caches. The historical unexplained runner exit did not recur; its original cause remains unproven.

Validated all 36 test-manifest requirements in Miniforge `py314` on Windows. Restored the optional OCR packages in that test environment without changing dependency declarations; Tesseract remains optional at runtime. Focused checks passed, including the isolated STXM example and example-harness failure cleanup. Full serial: 337 passed, 612 warnings, 440.95 s. Two workers: the same 337 passed, 613 warnings, 303.17 s. No skips; identical coverage counts and 79% combined coverage. All 73 documentation examples passed and all 369 audited fixture/cache files retained their original hashes. These are local results, not new CI or cross-platform validation.

Evidence and the next-batch proposal are in `maintenance/phase5/README.md`, `results.json`, `environment.json` and `fixture-check.json`. Phase 5 remains in progress. The proposed next batch covers test categories, scoped warning policy, explicit optional-OCR coverage and remaining directory/dialog/temporary-file isolation. Dependency-version boundaries follow separately.

### Phase 5 second batch complete (2026-09-11)

Added registered test categories, including explicit network/OCR selections, and documented commands in `tests/README.md`. Removed import-time warning filters and broad warning-category suppression; plotting/filtering tests retain scoped strict checks with narrowly documented headless/manual-axes exceptions. Folder metadata, folder-save, dialog and plotting tests now restore their local state and use managed output directories.

Replaced the OCR test's silent pass when unavailable with an explicit skip and real recognised-value assertions. New regressions exposed discarded OCR text crops and an invalid field-only helper call; both are fixed, using Tesseract single-line segmentation. Availability checks both the optional wrapper and executable. Missing dependencies preserve ordinary image operations and metadata. No dependency declarations changed.

Windows/Miniforge `py314`: 342 passed serially (593 warnings, 432.67 s) and with two workers (594 warnings, 262.29 s), no skips, identical test identities and coverage counts, 79% combined coverage. All 337 first-batch cases remain with five added OCR regressions. A separate unavailable-wrapper integration run reports one explicit skip. All 369 audited fixture/cache files are unchanged. See `maintenance/phase5/batch2.md` and its JSON evidence files.

Phase 5 remains in progress for supported dependency-version boundary environments and any demonstrated installed-distribution coverage gaps. Current local results do not replace cross-platform CI evidence.

### Phase 5 third batch complete locally (2026-09-11)

Created and verified a complete Python 3.11 test environment using NumPy 2.0.2,
SciPy 1.14.1, Matplotlib 3.8.4, scikit-image 0.24.0, lmfit 1.3.4 and pandas 2.3.3.
The representative lower-minor-line specification is `tests/minimum-env.yml`;
it is not an exhaustive dependency matrix or a test of every earliest patch.
No runtime dependency declarations changed, and Tesseract remains optional.

Resolved strict plotting-test failures with scoped Pyparsing alias-warning
exceptions for Matplotlib's legacy parser calls. All five plotting tests pass
on Python 3.11 and 3.14. Marked the real-data Attocube interpolation test `slow`
after observing its successful calculation exceed the two-minute diagnostic
threshold; it remains included in full runs. Strict marker collection passes.

Full Python 3.11 serial: 342 passed, 6856 warnings, 867.38 s. Two workers:
342 passed, 6977 warnings, 509.81 s. No skips; identical test identities and
coverage counts, matching Python 3.14's 79% combined coverage. All 73 examples
pass and all 369 audited fixture/cache hashes are unchanged. Runs overlapped,
so durations are not a performance comparison. See `maintenance/phase5/batch3.md`
and its JSON evidence files.

Prepared a micromamba CI job for this dependency combination and extended the
existing clean wheel/sdist resource probes to Python 3.11 and 3.14. YAML parsing,
dependency-name alignment and CI policy checks pass locally. Phase 5 remains
in progress solely for hosted validation of the accumulated changes: the ordinary
supported-Python matrix, lower-dependency job and installed-distribution jobs.
After that gate, the next proposed batch is Phase 6 Sphinx warning triage and
public/dynamically attached API documentation coverage.

### Phase 5 macOS CI follow-up (2026-09-11)

The first hosted macOS run of `d143d6f21` exposed two test-tool compatibility
defects: Pyparsing 3.0.4 lacks the imported deprecation-warning class, excluding
five plotting cases, and pytest-cov 3.0's subprocess configuration discovery
mixed statement and branch coverage. Guarded the optional warning class and
passed an absolute coverage configuration in CI and the full-suite runner.
An isolated environment reproduced the import error; the affected six cases
then passed with two workers and coverage. A child-startup probe confirmed the
coverage configuration fix. Hosted macOS job `103436035446` at fix commit
`6f9c039f4` then passed all 342 tests and coverage aggregation (157 warnings,
358.28 s). The lower-dependency and package-validation workflows also passed.
Linux Python 3.11, 3.13 and 3.14 passed; Python 3.12 and final matrix reporting
were still running when recorded. See `maintenance/phase5/batch3.md`.

The replacement Python 3.12 job stalled at 42% in its terse live log. An invisible
file dialog remains an unconfirmed hypothesis. Added a suite-wide guard that
fails unexpected file-dialog requests, while retaining explicit dialog mocks in
widget tests; its regression and focused loader/widget checks pass (6 tests).
CI now reports test names, slowest durations and two-minute stack dumps. Phase 5
remains open pending a completed, diagnostic Python 3.12 run.

### Phase 5 hosted validation complete (2026-09-11)

At commit `db79eaade`, run `34653015465` passed all 343 tests on Linux Python
3.11-3.14 and macOS Python 3.14, including the new unexpected-dialog regression.
Lower-dependency run `34653015413` and installed-distribution run `34653015414`
also passed, along with coverage aggregation and test-result publishing.
Phase 5 is complete. Final timings and run links are in `maintenance/phase5/batch3.md`.

Python 3.12 remains disproportionately slow in folder operations: base operators
took 176.07 seconds versus 22.64 on 3.13, while image functions took 21.30 versus
22.03 seconds. NumPy, coverage and pytest-cov version numbers match between those
jobs. The cause remains unresolved; controlled coverage/worker comparisons are
a separate performance follow-up. The observed stacks do not implicate HDF5 locks.

The next Phase 6 batch will classify Sphinx warnings, audit public and dynamically
attached API documentation, and fix a small evidenced group of references or
directives. Validate in RTD mode using the retained plot cache and record remaining
warning categories explicitly.

### Phase 6 documentation batch complete (2026-09-11)

Corrected autosummary template names and the primary API index, linked ImageStack
to its full reference page, and added contributor environment/test/build guidance.
Fresh RTD-mode builds reduce Sphinx warnings from 1,658 to 506 without suppression.
All five primary classes and 85 dynamically attached Data methods remain in the
API inventory, with no lost Python entries. All 243 plot-cache hashes are unchanged.
Evidence and the repeatable inventory/warning audit are in `maintenance/phase6`.

Phase 6 remains in progress. The proposed next batch fixes malformed Kerr helper
docstrings that produce repeated indentation warnings, starting with crop_text,
defect_mask and defect_mask_subtract_image. Imported-symbol stubs and duplicate
attribute documentation remain separate, recorded warning categories.

### Phase 6 Kerr helper docstrings complete (2026-09-11)

Corrected crop_text, defect_mask and defect_mask_subtract_image documentation to
match the implemented shapes, thresholds, copy behaviour and optional returns.
Only docstrings changed, verified by AST comparison. RTD-mode Sphinx warnings
fall from 506 to 490; all 16 diagnostics from these helpers disappear. API
inventory coverage and all 243 plot-cache hashes remain unchanged. Evidence:
`maintenance/phase6/batch2.md` and its JSON reports.

The next proposed batch covers KerrStackMixin.crop_text, ImageStackMixin.convert
and ImageStackMixin.correct_drifts, retaining source-behaviour and inventory
checks. Phase 6 remains in progress.

### Phase 6 stack docstrings complete (2026-09-11)

Corrected KerrStackMixin.crop_text, ImageStackMixin.convert and correct_drifts.
Documented stack mutation, masks, conversion controls and the legacy None return.
Only docstrings changed (AST verified). Clean RTD-mode warnings fall from 490 to
470; all five primary classes and 85 dynamic Data methods remain documented, with
no lost Python inventory entries and all 243 plot-cache hashes unchanged.
Evidence: `maintenance/phase6/batch3.md` and accompanying JSON reports.

A public ImageStack probe confirms an existing same-dtype force_copy=True defect:
the shared converter accesses ndarray.clone and raises AttributeError. Record a
focused runtime fix with copy/mask regression coverage for Phase 7; no algorithm
was changed here. The next documentation batch targets Data overview markup,
Setas.__call__ and plot_xyuv. Phase 6 remains in progress.
