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

- Check the full hosted matrix: Linux Python 3.11-3.14 and macOS Python 3.14.
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

## Current release preparation

This is the remaining work from Phase 8, not a new requirement to repeat the
completed repository repair programme.

- [ ] Verify hosted tests, package checks and documentation for the release
  preparation commit, including source-based Sphinx versions, tag checks and
  installed-version checks. Historical records have been retired and retained
  maintenance tools and references validated after relocation.
- [ ] Verify hosted Codacy reanalysis of the locally reviewed cleanup.
  Keep the deferred complexity and mixed-line-ending work out of this release batch.
- [ ] Review outstanding release metadata, archive size and service/citation links.
- [ ] Prepare the release notes for the version set in `Stoner/__init__.py`.
- [ ] When authorised, tag and publish; record successful package/documentation
  jobs and verify published versions before closing release preparation.

## Known limitations and follow-ups

- [ ] **Data storage migration on `devel`:** follow
  [STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md) for the agreed pandas/xarray
  architecture, session-sized batches, validation gates and current handover.
  Begin with contract characterisation; backend method simplification and new
  Stoner wrappers follow only after storage migration validation. This is separate
  from the current release preparation above.

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
