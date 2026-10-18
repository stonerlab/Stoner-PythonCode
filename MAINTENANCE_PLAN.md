# Repository maintenance

This is the recurring maintenance checklist for Stoner. Keep it focused on current
policy, repeatable checks and actionable follow-ups. Record completed work in
commits, pull requests, release notes and CI runs rather than accumulating dated
reports in the source tree.

The initial repository repair programme is complete. Its Phase 0-7 records are
available in Git history at `0e52ec459`; for example,
`git show 0e52ec459:MAINTENANCE_PLAN.md`. The current release preparation is listed
below. Repository conventions remain in [AGENTS.md](AGENTS.md), with contributor
commands in [the developer guide](doc/UserGuide/developer.rst).

## For each change

- Check branch, remote and working-tree status. Use Windows Git with repository-local
  `core.fileMode=false` and `core.autocrlf=false`; preserve unrelated changes and
  existing line endings. Never normalise the entire checkout as routine maintenance.
- Make cohesive changes with a clear behaviour to preserve. Keep metadata, masks,
  column roles, chained/in-place operations and loader identification intact.
- Use a supported Conda environment verified against `tests/test-env.yml`; use
  `tests/minimum-env.yml` when checking lower dependency bounds. Preserve the
  separate Python 3.6 LabVIEW environment.
- Run focused regressions while developing, then the full suite for shared behaviour
  changes. Use real scientific fixtures where format interpretation is involved.
  Fixture paths must match the capitalisation in `git ls-files`.
- Update docstrings and examples with public behaviour changes. Follow
  [DOCSTRING_STYLE.md](DOCSTRING_STYLE.md), including British English prose.
- Inspect the diff and whitespace, then stage only intended changes. Record tests
  and any limits of validation in the commit or pull request. Check hosted results
  after pushing; local passes do not establish remote or physical compatibility.

## After pushes and before merging

- Check the full hosted matrix: Linux Python 3.12-3.14 and macOS Python 3.14.
  The Windows development environment provides local platform coverage.
- Check the lower-dependency job and installed-distribution jobs, including final
  test-result publishing. Do not treat a successful test step as success for a job
  whose reporting or packaging steps failed.
- Collect coverage on Ubuntu Python 3.14 only. Keep all matrix jobs running the full
  suite. Verify Coveralls and Codacy uploads where applicable; investigate a drop
  in coverage instead of suppressing it.
- Diagnose failures from logs, JUnit results and resolved environment versions.
  Distinguish assertion failures, file-dialog requests, worker loss and slow tests
  before changing code or increasing timeouts.

## Monthly, and when dependencies or CI change

- Review supported Python versions, dependency bounds, deprecations and relevant
  upstream changes. Reconcile `pyproject.toml`, `requirements.txt`,
  `recipe/meta.yaml`, `recipe/build-env.yml`, `tests/test-env.yml`,
  `tests/minimum-env.yml`, `doc/docs-env.yml` and `doc/requirements.txt`.
  Testing/documentation extras may differ intentionally from runtime dependencies.
- Keep Conda specifications primarily Conda-based. Search `phygbu` and `conda-forge`
  before proposing pip-only dependencies; document actual packaging gaps.
- Review pinned GitHub Actions, runner availability, permissions and build tools.
  Update `maintenance/check-ci-config.py` when an agreed CI policy changes.
- Run the CI and repository-content checks described in
  [maintenance/README.md](maintenance/README.md). Review package discovery after
  changing layout, resources or packaging configuration.
- Check source-tree growth. Keep runtime assets, scientific fixtures and
  `doc/plot_cache`; keep local logs, environment snapshots, builds and profiling
  outputs ignored. Remove completed investigation reports from the live tree once
  lasting guidance and regression coverage have been retained.
- Review open bugs and deferred work. Avoid mechanical rewrites, removal of used
  compatibility helpers or blanket annotation work without a demonstrated benefit.
- Refresh Codacy findings against the analysed commit. Triage deleted or relocated
  paths before editing; distinguish real defects from dynamic-API inference and
  intentional import ordering. Fix small issues with focused regressions, and keep
  complexity refactoring separate from release preparation. Store local exports
  under ignored `maintenance/runs` and verify hosted reanalysis after pushing.

## When documentation changes, and before a release

- Build cached HTML with `READTHEDOCS=True`, retaining the warning log. Run
  headless builds with `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen`; importing
  the package can initialise Qt even when plotting uses Matplotlib's Agg backend.
  Run `maintenance/audit-docs.py` against the platform's reviewed warning baseline in
  `maintenance/docs`. New or increased warnings need investigation; never increase
  the ceiling solely to make a build pass.
- Verify all five primary classes and dynamically attached Data methods remain
  documented. Compare inventories with a previous build when changing API indexes;
  permit removed targets only after reviewing whether they are genuinely obsolete.
- Use a case-sensitive Linux build to check intentional function/class names that
  differ only in capitalisation. Do not rename those APIs to work around Windows
  filename collisions.
- Run the documentation example tests when examples or their underlying APIs change.
  Non-cached builds deliberately execute plotting examples and refresh the cache;
  review those generated changes before committing.
- Serve generated HTML alone over localhost for visual checks. Inspect changed
  pages and navigation, then stop the preview when review is finished.

## For each release

- Set the version only in `Stoner/__init__.py::__version__`. Runtime, setuptools,
  Conda and Sphinx derive their versions from it. The Conda build number is a
  separate packaging revision. Commit the change before creating its matching
  `v<version>` tag; never retarget an already published release tag.
- Prepare release notes covering public changes, fixes, compatibility/dependency
  changes and repository-only maintenance. State unresolved platform or facility
  validation limits precisely.
- Use GitHub Actions for all distribution builds. `check-packages.yaml` already
  checks wheel/sdist contents and installs each separately in fresh environments
  on pushes and pull requests. Record successful results for the release commit;
  do not repeat those builds locally or reuse old files from `dist`.
- Review metadata, licences, URLs, dependency declarations and archive size where
  automated checks do not cover them. Check runtime assets including plot styles
  and OCR tessdata. Keep GPLv3 declarations consistent.
- Publish the matching GitHub release only when authorised. Verify the release
  jobs build that tag, reject a version mismatch, and complete PyPI/Conda publication
  and documentation building. Manual dispatch of the deployment workflow also
  publishes packages; it is not a build-only validation shortcut.
- Verify the published versions and metadata on PyPI and Conda, the Read the Docs
  version, and coverage and citation/DOI links. Wheel/sdist smoke tests do not prove
  the Conda package works; check its installed import/version test separately.
- Record release evidence in the release or pull request. Remove completed items
  from the active checklist below rather than retaining another phase report.

## Current release follow-up

[`0.12.0a1`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a1)
is published as a GitHub prerelease from `devel`, tagged at
`4d3187a3611b6812212751aeb30a541e5b1c3e8d`. The release records full validation
evidence; completed preparation details remain in Git history.

[Packaging/publication CI](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39076474024)
passed both deployments. PyPI has the wheel/sdist and the published wheel passed
installed-package smoke checks outside the checkout. Conda has
`noarch/stoner-0.12.0a1-py_1.conda`, with a passing installed import/version test.
[Release documentation CI](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39076473982)
passed with 61 reviewed warnings and zero unexpected warnings. Zenodo archived the
tag as [DOI 10.5281/zenodo.23443114](https://doi.org/10.5281/zenodo.23443114).
Codacy confirmed zero mixed-line-ending findings; remaining complexity/style and
reviewed inference/test-utility advisories did not block the alpha.

- [ ] Activate and build `v0.12.0a1` on Read the Docs, then verify its versioned
  documentation URL. The public API detects the correct tagged SHA but reports
  `active=false`, `built=false`; the version URL currently returns 404. Activation
  requires a signed-in maintainer session, which was unavailable during release
  verification. The successful GitHub documentation build does not publish that
  Read the Docs version automatically.
  The immutable alpha tag predates the headless startup fix below: configure
  `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg` in Read the Docs for that tagged
  build before activating it.
- [x] Verify the Read the Docs `devel` rebuild after fixing build 35216175's Qt
  startup abort. `doc/conf.py` now sets offscreen Qt and Agg for Read the Docs
  before importing Stoner, matching the already validated GitHub docs environment.
  Fresh-process probes pass with backend variables initially unset and confirm
  non-Read-the-Docs builds retain explicit settings. A clean cached Sphinx build
  without externally supplied headless settings passed with 88 reviewed Windows
  warnings, zero unexpected warnings and no removed API objects. All 243 tracked
  plot-cache files are unchanged. The Linux documentation job also passed with
  61 reviewed warnings and zero unexpected warnings.
  [Read the Docs build 35216225](https://app.readthedocs.org/projects/stoner-pythoncode/builds/35216225/)
  succeeded on `05d7cd0d84d4392535799eea56a8511aa9f69fcb`; the published `devel`
  upgrade guide was verified in the browser with the alpha and completed wrapper
  notes (the separate automated HTTP client was blocked with HTTP 403).
  Keep the published alpha tag immutable; this documentation fix is on `devel`.

## Known limitations and follow-ups

- [ ] **Plotting redesign (stages 0-3 complete):** see
  [PLOTTING_REFACTOR_PLAN.md](PLOTTING_REFACTOR_PLAN.md) for the callable Data
  plotting accessor, stonerplots layouts/formatting and bounded pandas delegation.
  API changes are acceptable with aligned examples and documentation. The plan
  records ordinary/scientific plotting and direct stonerplots layout/style
  contexts, aligned examples and local validation. Stages 0/1 are committed as
  `374980c2e`; stage 2 as `ec0d62fa6`, using required stonerplots 1.9.6. Stage 3
  remains uncommitted on `codex/plotting-accessor-stages-0-1`; its full local suite
  passes 1,218 tests. Next is stage 4's bounded pandas convenience evaluation;
  folder pagination/removal remains in stage 5. No push or release
  is part of this batch.

- [ ] **Code complexity remediation (non-blocking for `0.12.0a1`):** the
  complete 2026-10-17 `devel` Codacy snapshot contains 85 complexity-category findings.
  Review and address these in bounded maintenance batches after alpha release
  preparation, favouring cohesive helpers and meaningful regression coverage.
  Do not require complexity cleanup or a zero-finding quality report to publish
  the alpha. Review any independently identified correctness or security defect
  on its actual release impact rather than its complexity score.

- [x] **Native storage wrappers on `devel` (Stages 1-9 complete):** see
  [STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md) for the agreed pandas/xarray
  architecture, validation gates and current handover. Stages 1-8 are complete:
  `Data.data` exposes its actual DataFrame; `ImageFile.image` and `ImageStack.stack`
  expose native Datasets with explicit numerical exports, selection, exclusion,
  ownership and mutation contracts. Final full hosted suites passed 909 tests in
  every supported matrix cell; lower dependencies, installed distributions,
  documentation and all 19 integrated benchmark pairs also passed.
  Stage 9a adds independent `Data.head`/`tail` previews under the recorded wrapper
  contracts and delegation inventory. Stage 9b delegates compatible real numeric
  sorting while retaining numerical dtype/comparator fallbacks and owner commits.
  Stage 9c retains NumPy uniqueness after reviewing its mask, ordering and optional
  index/inverse contracts; dedicated tests and documentation record that decision.
  Stage 9d copies selected preview rows through the existing deepcopy graph, with
  custom copying hooks retaining their established path. Stage 9e's cumulative
  review and installed-wrapper coverage passed on `4525bf7d5`: 1,099 tests in each
  supported hosted cell; 1,098 plus one existing skip with lower dependencies;
  wheel/sdist checks on Python 3.12/3.14; cached docs with zero unexpected warnings;
  and all 19 integrated performance pairs. The migration plan retains exact-run
  links and source fingerprints. Next: separately authorised release preparation
  above; additional wrapper families need independent bounded proposals.

- **MAXIMUS:** image and point-scan readers select the first region and do not
  implement region-to-file mapping or explicitly reject multi-region inputs.
  Multiple stack files do not establish multi-region support. Require authoritative
  facility guidance and complete real multi-region exports with expected axes,
  dimensions and metadata before extending support. Preserve single-region fixtures.
- **SPC:** alternative multiple-x/y layouts need representative exports, a format
  specification and expected results before implementation. Truncation handling
  does not establish support for those layouts.
- **Legacy TDI metadata:** retain the TDI 1.5 deepcopy workaround until the original
  writer's rationale is established. The shared `copy_into` type-preservation fix
  does not justify removing that workaround or changing generic deepcopy semantics.
- **Intermittent Python 3.12 worker loss:** subsequent runs passed; tracing overhead
  explains observed slowdown but not the historical worker death. If it recurs,
  collect worker exit status and runner memory/OOM or native-crash evidence before
  changing loaders, skipping tests or increasing timeouts.
