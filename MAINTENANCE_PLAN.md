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

## Current release preparation

This checklist prepares `0.12.0a1` from `devel`. Storage migration Stages 1-9 are
complete; complexity remediation is a non-blocking maintenance follow-up.

- [x] Verify hosted tests, package checks and documentation for the release
  preparation commit, including source-based Sphinx versions, tag checks and
  installed-version checks. Historical records have been retired and retained
  maintenance tools and references validated after relocation. Migration Stage 7
  records the alpha's hosted matrix/package results and audited documentation build.
- [x] Verify hosted Codacy reanalysis of the locally reviewed cleanup and triage
  remaining correctness/security findings for release impact. A zero-finding
  Codacy report is not an acceptance requirement for `0.12.0a1`.
  Hosted analysis completed on `7e3dc747d`: **zero mixed-line-ending findings**,
  down from 237 in the first complete migration analysis (595 findings total).
  The remaining 358 findings include 85 complexity, 62 security-category and 47
  error-prone-category findings. Security findings comprise 55 assertions in test
  utilities and seven subprocess advisories in maintenance tools using argument
  lists without a shell. Review found no release-blocking defect: dynamic image,
  Kerr and folder APIs work at runtime; typed-metadata `super` calls deliberately
  target a TypeHintedDict instance; TDI assignments are guarded by container type;
  image operator closures capture each operation in a default argument; frame and
  region owners deliberately resolve parent state without base-owner allocation.
  The focused TDI/folder suite passed 67 tests, and dynamic-API/metadata preflights
  passed. Remaining style, unused-code and complexity work is non-blocking.
- [x] Review outstanding release metadata, archive size and service/citation links.
  Python >=3.12, canonical version `0.12.0a1`, required dependency bounds, GPLv3
  licence files and stonerlab URLs agree across packaging/runtime/test/docs inputs.
  Package discovery retains 103 Python modules and all ten runtime resource files.
  Hosted strict Twine checks and wheel/sdist installation passed on Python 3.12
  and 3.14. Reviewed archive sizes are about 3.26 MB wheel and 46.43 MB sdist;
  OCR resources and retained scientific fixtures explain the largest entries.
  PyPI and Conda APIs still report 0.11.5, with no existing 0.12.0a1. Read the Docs
  stable/latest and the Zenodo citation resolve (DOI 10.5281/zenodo.22734604).
  Coveralls page access returns HTTP 403 to automated requests; CI coverage uploads
  succeeded. Published versions, documentation version and the new citation record
  must be checked again after release. Existing setuptools licence-format
  deprecation warnings are maintenance follow-up; GPLv3 declarations are consistent.
- [x] Prepare the release notes for `0.12.0a1` in `doc/UserGuide/upgrade.rst`.
  Release preparation now includes the completed Stage 9 previews, sorting and
  retained uniqueness contracts. CI package validation checks the description
  with strict Twine validation and reports actual archive sizes for review.
  Updated cached docs passed with 88 reviewed Windows warnings and 61 reviewed
  Linux warnings, zero unexpected warnings, no removed API objects and no missing
  dynamic Data methods. Release-note layout was checked over localhost; the
  temporary browser tab/server were closed. All 243 cached plot files are retained.
- **Final preparation revision:** `7e3dc747d15aaaff36959b1168c52a42826669da`.
  Its newline-normalised runtime fingerprint is unchanged from the completed
  Stage 9 exit: `a958a5809ae60494188e04dda3779f4f64624f6f18fa80ec3aa389a042b5be94`.
  [Package validation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39075488013)
  passed strict Twine, archive contents and installed wheel/sdist checks on
  Python 3.12/3.14. [Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39075487997)
  passed 1,098 tests with the existing pandas 3 lifecycle skip. The final
  [supported matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39075488004)
  passed 1,099 tests in each Linux Python 3.12-3.14 and macOS Python 3.14 cell,
  including final result publication and the configured coverage uploads.
  [Cached documentation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39075706849)
  passed with 61 reviewed Linux warnings and zero unexpected warnings. The
  candidate is ready; only the documentation-only checklist closure follows this
  fully validated revision. Publication will use the GitHub prerelease
  event to run the existing PyPI/Conda deployment and documentation workflows;
  no manual distribution build or upload is required.
- [ ] When authorised, tag and publish; record successful package/documentation
  jobs and verify published versions before closing release preparation.

## Known limitations and follow-ups

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
