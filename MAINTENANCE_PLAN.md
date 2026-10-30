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
  automated checks do not cover them. Check runtime assets including OCR data and dependency-provided plot styles
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

## Planned API work

- [ ] Follow the agreed image accessor sequence: [plot](IMAGE_PLOT_ACCESSOR_PLAN.md),
  then [ROI](IMAGE_ROI_ACCESSOR_PLAN.md), then [fit/flattening](IMAGE_FIT_ACCESSOR_PLAN.md).
  Reuse existing selection code through a general ROI handler; support current
  state/set/clear separately from exclusions. Share Data's API conventions and
  coordinate services, then fit plane/quadratic reference regions and correct the
  whole image, with a separate alpha for each accessor.
  Standalone ImagePlotter is published in 0.12.0a6, with exact-source hosted and
  publication evidence recorded below. Its handover and the next ROI R1 batch
  are in the plotting/ROI plans; ROI and image fitting are not implemented.
- [ ] Follow [RELEASE_0_12_PLAN.md](RELEASE_0_12_PLAN.md): API lock before beta,
  allowing only small compatible additions, then full API freeze/bug fixes only
  for RC. Review cumulative 0.11 breakage, migration and the final version series
  before the lock. Existing alpha tags remain immutable.
- [ ] Follow [STRING_ENUM_PLAN.md](STRING_ENUM_PLAN.md) for public string-valued
  choice arguments. Start with fitting outputs, retain supported string calls,
  and audit unrelated domains in bounded compatibility-preserving batches.
- [ ] Follow [DATA_ACCESSOR_PLAN.md](DATA_ACCESSOR_PLAN.md) for the reviewed
  candidate accessor groups and their ownership boundaries. Fitting is the first
  implemented group; the other groups remain future-work proposals.
- [ ] Implement [FIT_ACCESSOR_PLAN.md](FIT_ACCESSOR_PLAN.md) in bounded batches:
  snake_case model registry, registration through the existing `make_model`
  decorator, `Data.fit`, consumer/folder migration, documentation and validation.
  Settle the explicitly proposed compatibility/default choices before coding them.
  The registry/decorator, `FitOutput`, `DataFitter` and `Data.fit` are implemented
  and locally validated, including folder fitting and internal consumers. The
  documentation/sample/script migration is complete; compatibility warning timing
  and removal policy remain undecided. See the plan handover
  for evidence and the remaining migration decisions.

## Current release follow-up

### Published 0.12.0a6

[`0.12.0a6`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a6)
is published as a prerelease at `3d72b7047994af54307d2234c5e77e79e837fb23`.
It is the ImageFile.plot milestone: Axes-return image/mesh/contour/histogram
drawing, shared Data-style properties/contexts, native/pixel coordinates and
exclusion-aware rendering. Legacy imshow retains its Figure return. Runtime
dependencies are unchanged; ROI and image fitting remain the next milestones.

Every Linux Python 3.12-3.14/macOS Python 3.14 cell passed 1,804 tests in the
[exact-source matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41128446321).
[Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41128446256)
passed 1,803 with one expected skip;
[installed wheel/sdist checks](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41128446205)
and [cached docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41128445576)
also passed before tagging. Coverage checks passed; no gate was waived.

[PyPI](https://pypi.org/project/Stoner/0.12.0a6/) publishes wheel and sdist. The
public wheel passed the installed-package numerical/resource/plotting probes,
including ImagePlotter/property/context checks, outside the checkout. The
[deployment](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41130394817)
passed both registry jobs and the Conda installed import/version test.
`noarch/stoner-0.12.0a6-py_0.conda` is present in the public `phygbu` main-channel
index. [Release docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41130395020)
passed. Published Python/dependency/GPLv3/repository metadata were checked.
Zenodo archived the immutable tag as
[DOI 10.5281/zenodo.23820651](https://doi.org/10.5281/zenodo.23820651).

- [ ] Activate/build `v0.12.0a6` on Read the Docs, then verify its public pages.
  Its API identifies the correct release SHA but reports `active=false`,
  `built=false`. [Version settings](https://app.readthedocs.org/dashboard/stoner-pythoncode/version/v0.12.0a6/edit/)
  require a maintainer session. Manual live GUI/clipboard and physical instrument
  compatibility were not newly validated. The published tag must not move.

### Published 0.12.0a5

[`0.12.0a5`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a5)
is published as a prerelease at `4faff759a4d856bb038ed3ff7cae097da5776935`.
It includes fitting accessors and bulk shortcuts, the model registry/decorators,
`FitOutput`, plotting-owned fit annotations, consumer migrations and the column
schema/all-match consolidation. Runtime dependencies are unchanged from a4.

The exact release source passed 1,762 tests in every Linux Python 3.12-3.14/macOS
Python 3.14 cell of the [test matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41114563912).
[Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41114564000)
passed 1,761 tests with one expected skip.
[Fresh wheel/sdist checks](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41114563983)
and [cached docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41114562835)
also passed before publication. No gate was waived.

[PyPI](https://pypi.org/project/Stoner/0.12.0a5/) publishes wheel and sdist.
The public wheel passed installed-package resource/numerical/plotting checks and
new fitting accessor, enum, folder-shortcut and decorated-model probes from a
temporary installation outside the checkout. The
[deployment](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41115966505)
passed both registry jobs, including Conda's installed import/version test.
`noarch/stoner-0.12.0a5-py_0.conda` is present in the public `phygbu` main-channel
index. [Release docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/41115966448)
passed; coverage checks on the release source passed. Zenodo archived the tag as
[DOI 10.5281/zenodo.23817189](https://doi.org/10.5281/zenodo.23817189).

The maintainer activated Read the Docs; [build 35532428](https://app.readthedocs.org/projects/stoner-pythoncode/builds/35532428/)
passed on the exact release SHA. The version API now reports `active=true` and
`built=true`. The [published documentation](https://stoner-pythoncode.readthedocs.io/en/v0.12.0a5/)
passed checks of 12 pages, 113 fitting links/fragments, six images and 17 fitting
inventory targets. Rendered registration/folder guides were inspected, and the
fitting example's PNG/hires PNG/PDF match the retained release cache byte-for-byte.
Sphinx reported 37 non-fatal warnings, including duplicate API descriptions and
third-party footnotes; two duplicate-object counts exceed the stored Linux baseline.
No checked fitting link or image was broken. There are no remaining publication
gates for this alpha. The tag is immutable; physical instrument compatibility
was not newly tested.

### Published 0.12.0a4

[`0.12.0a4`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a4)
is published as a prerelease at `a96e4bcce70bc96ebfdd68a2efba96f976422ac0`.
It includes callable plotting properties, shorter task-focused guides,
sample/script API updates and the backwards-compatible issue #41 CSV INI fix.
Runtime dependency requirements are unchanged.

The exact release source passed 1,439 tests in each cell of the
[full Linux/macOS matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40413738248).
The first unusually slow macOS attempt was cancelled for diagnostics; a fresh
complete matrix passed, including final reporting. No gate was waived.
[Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40410612257)
passed 1,438 with one expected skip;
[fresh wheel/sdist checks](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40410612452)
and [cached docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40410620541)
also passed. Local Windows validation passed 1,436 tests with three network/OCR
deselections; the affected network example passed separately. Native GenX UI and
interactive mouse/dialog behaviour were not manually exercised.

[PyPI](https://pypi.org/project/Stoner/0.12.0a4/) publishes the wheel and sdist;
the public wheel passed the installed-package probes and plotting property
checks. Conda publishes `noarch/stoner-0.12.0a4-py_0.conda` on `phygbu`, with
successful installed import/version tests and upload in the
[deployment](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40415912602).
[Release documentation CI](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40415912648)
passed. Zenodo archived the tag as
[DOI 10.5281/zenodo.23677984](https://doi.org/10.5281/zenodo.23677984).
Read the Docs has discovered the correct tag/SHA but still needs maintainer
activation and a successful versioned build. The published tag is immutable.

### Published 0.12.0a3

[`0.12.0a3`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a3)
is published as a GitHub prerelease from `devel`, tagged at
`92fd4f3baef3df5ff2f9f5144188edd94ef6de7b`. It includes OffsetPlots, both new
double-axis contexts, the style wrapper, examples/docs and the bounded Pillow 13
TIFF/test-runner compatibility fixes exposed by the first CI candidate.
The required stonerplots version is 1.9.8; all dependency manifests are aligned.

The exact release source passed **1,396 tests with three expected issue #41
failures** in all Linux 3.12-3.14/macOS 3.14 matrix cells and on local Windows
with Pillow 13. The lower dependency suite passed 1,395, with one expected skip
and the same three expected failures. All 82 rendered examples are included.
Evidence: [full matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40203566136),
[lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40203566100),
[fresh wheel/sdist installations](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40203566115)
and [cached docs](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40203566150).
[Tag-triggered matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40204595870)
and tag package/lower-dependency jobs also passed. No gate was waived.

[PyPI](https://pypi.org/project/Stoner/0.12.0a3/) publishes both wheel and sdist
with Python >=3.12 and stonerplots >=1.9.8 metadata; the public wheel passed the
installed-package probe outside the checkout. Conda channel repodata contains
`noarch/stoner-0.12.0a3-py_0.conda`, with a successful installed import/version
test and upload in the
[release deployment](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40204597020).
[Release documentation CI](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/40204596843)
passed. The maintainer activated the tagged Read the Docs version;
[build 35393219](https://app.readthedocs.org/projects/stoner-pythoncode/builds/35393219/)
passed on the exact release SHA, and the
[published plotting guide](https://stoner-pythoncode.readthedocs.io/en/v0.12.0a3/UserGuide/plotfile.html)
was verified in the browser. Zenodo archived the release as
[DOI 10.5281/zenodo.23644220](https://doi.org/10.5281/zenodo.23644220).
There are no outstanding publication gates for this alpha. The tag is immutable.

### Published 0.12.0a2

[`0.12.0a2`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a2)
is published as a GitHub prerelease from `devel`, tagged at
`f995d6068eb7ecacc9a679bfa22b892769984dba`. The plotting redesign, scan-frame
optimisation, validation notes, version and upgrade documentation are merged.
The release records the complete validation and publication evidence; there are
no outstanding publication gates for this alpha. Published tags remain immutable.

The exact release source passed 1,331 tests in every Linux 3.12-3.14/macOS 3.14
matrix cell and on local Windows; lower dependencies passed 1,330 with one
expected skip. Fresh wheel/sdist installations and cached documentation passed.
[Automatic tag validation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39163437419)
also passed all four full-suite cells, including macOS Attocube. Local cached
documentation had 62 reviewed warnings, zero unexpected warnings and no lost API
entries; all 276 tracked plot-cache files were retained.

[PyPI](https://pypi.org/project/Stoner/0.12.0a2/) publishes the wheel and sdist
with Python >=3.12 and stonerplots >=1.9.6 metadata. The published wheel passed
the installed-package smoke checks outside the checkout. Conda publishes
`noarch/stoner-0.12.0a2-py_0.conda` in `phygbu` channel repodata; its installed
import/version test passed in the
[recovery deployment](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39164610037),
which checked out the original release tag and skipped PyPI.

The initial Conda deployment reported success without an archive: its nested
solver omitted `phygbu`, so stonerplots >=1.9.6 could not be resolved, and the
third-party upload action masked the build failure. Deployment now builds/tests
with explicit channels and Python 3.12, rejects missing archives, and supports a
Conda-only recovery from the original release tag. The published tag is unchanged.

The deployment-only fix at `4e9ec76d8` passed the
[full matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39164565387),
[lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39164565381)
and [installed distributions](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39164565319).
[Read the Docs build 35227451](https://app.readthedocs.org/projects/stoner-pythoncode/builds/35227451/)
passed on the exact release SHA; the public
[versioned documentation](https://stoner-pythoncode.readthedocs.io/en/v0.12.0a2/),
release notes, plotting guide and cached images were verified in the browser.
Zenodo archived the release as
[DOI 10.5281/zenodo.23458853](https://doi.org/10.5281/zenodo.23458853).

### Previous alpha

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

- [ ] **Column schema and setas consolidation:** implement the staged
  [follow-on plan](STORAGE_MIGRATION_PLAN.md#column-schema-and-setas-consolidation-2026-10-27).
  CS1's [caller/cardinality audit](maintenance/column-schema-audit.md) and CS2's
  all-match selection, explicit `first=True`, matching precedence and caller
  migrations and CS3's composed schema/shared role logic are complete locally.
  CS4's isolated Index comparison is complete on pandas 2.2.3 and 3.0.5.
  CS5 retains the composed schema and rejects custom Index adoption for this
  consolidation: one fewer owner field brings additional propagation and
  provenance adapters without a consistent speed benefit. CS6 documentation,
  executable examples and local checks are complete: 1,614 current-environment
  passes; lower pandas-line results and overlay details are in the handover;
  Sphinx retains 62 reviewed warnings and no API removals. Final-revision
  hosted matrix/lower-line, docs and applicable package checks remain open.
  Keep CS6 and this parent item open until those results are observed; the
  prototype stays isolated as decision evidence.

- [ ] **Weighted mean normalisation:** investigate the pre-existing formula in
  `Stoner/analysis/columns.py::mean`. With x=[0,1,2], y=[2,4,6], sigma=[1,1,1]
  and setas='xye', `data.mean(1)` returns a mean of 4/3 rather than the
  equal-weight reference 4. HEAD already contains the same
  `(ydata * w).mean(axis=0) / norm` expression. CS6 recorded the reproduction
  under ignored `maintenance/runs/cs6-example/weighted-mean-observation.json`;
  numerical correction and uncertainty-policy review are a separate task.

- [ ] **macOS slow-test diagnosis:** compare JUnit durations and timed-out test
  names before changing scientific code. The a4 source passed the full macOS
  suite; its slowest test was `plot_folder_demo.py` (179 seconds, previously
  125), followed by image functions (99 versus 78 seconds) and the real
  critical-temperature script (81 versus 47 seconds). The cancelled run's stack
  dumps showed image registration and repeated row traversal during splitting
  and fitting, rather than one established deadlock. A separate post-release
  five-minute watchdog bounds individual macOS tests and permits xdist worker
  replacement, retaining failure status and the test name. Local fault injection
  verified a named JUnit error, replacement worker and completion of eight
  healthy tests; hosted validation is still required for that CI follow-up.

- [x] **Scientific script API alignment:** all eleven Python scripts now use
  current plotting/storage interfaces and import-safe entry points where
  standalone. See [the script guide](scripts/README.md). Real-data and controlled
  regressions cover fitting, masks, reductions, GenX handoff and one-point BNL
  import. Local py314 validation passes 1,359 tests with three expected failures
  for issue #41 below. Fresh cached docs have 62 reviewed warnings, zero unexpected
  warnings and no lost API entries; all 276 plot-cache files are retained.
  Native GenX/wxPython and interactive mask/VSM workflows still need user-driven
  validation; host-stub and headless tests do not certify those application UIs.

- [x] **CSV layout options in fitting INI files:**
  The [issue #41](https://github.com/stonerlab/Stoner-PythonCode/issues/41) fix
  forwards explicit CSV layout controls while retaining omitted loader defaults.
  Historical `start_line`/`header_line` numbering remains one-based when
  `data_line` is absent; modern `data_line` selects zero-based INI numbering.
  Modern delimiters override `separator` independently, and native Python layout
  keywords override normalised INI values. The PCAR INI files and fitting guide
  document compatibility and supported detection. The three former expected
  failures now pass. Local py314 validation passes 58 focused tests and 1,436
  broader regressions with network/OCR excluded; the network-loaded fitting
  example also passes separately.
  Historical release results above retain their original expected-failure counts.

- [x] **Plotting redesign (stages 0-6 complete):** see
  [PLOTTING_REFACTOR_PLAN.md](PLOTTING_REFACTOR_PLAN.md) and the
  [release notes](doc/release-notes.rst). Stage 5 is committed as `8d6b49b93`;
  stage 6 fixes as `855f37010` and `33b36ec81` on the pushed feature branch.
  The full Linux 3.12-3.14/macOS 3.14 matrix passes 1,327 tests per job; lower
  dependencies pass 1,326 with one expected skip. Both installed wheel/sdist
  endpoint jobs and cached documentation pass. Local cached docs have 62 reviewed
  warnings and zero unexpected, with no unintended API removals. Included in
  `0.12.0a2`, with final release validation recorded above.

- [x] **OffsetPlots and plotting-consumer audit (25 October 2026):** directly
  expose the stonerplots context through the accessor/package and raise the
  dependency floor to 1.9.7. Add guide/API documentation, a rendered example,
  scientific-state/context integration tests and the installed-package probe.
  Audit runnable samples/scripts and correct guide FFT/semilog calls, cookbook
  vector descriptions and FMR annotation coordinates. Include the previously
  skipped main-guarded FMR workflow in rendered-example tests. See the
  [post-migration handover](PLOTTING_REFACTOR_PLAN.md#post-migration-extension-offsetplots-25-october-2026)
  for validation and the local preview; released in 0.12.0a3.

- [x] **DoubleXAxis/DoubleXYAxis integration (25 October 2026):** expose the
  stonerplots 1.9.8 contexts alongside DoubleYAxis, with aligned dependency
  manifests, guide/API documentation, a cached example and integration tests.
  See the [plotting handover](PLOTTING_REFACTOR_PLAN.md#post-migration-extension-doublexaxis-and-doublexyaxis-25-october-2026).
  Released in 0.12.0a3 with the OffsetPlots changes.

- [x] **Style wrapper and examples (25 October 2026):** `Data.plot.style` now
  accepts SavedFigure-style comma-separated names and `extra=` overrides,
  retaining native dictionaries, mixed lists, files and `after_reset`. Update
  eleven examples and the guide/API documentation. See the
  [plotting handover](PLOTTING_REFACTOR_PLAN.md#post-migration-extension-style-names-and-overrides-25-october-2026)
  for validation; released in 0.12.0a3.

- [x] **Attocube frame metadata cost:** removed the numerical export when rebinding
  the metadata owner in `ScanMetadataMixin._instantiate`, shared with MAXIMUS.
  Regression tests retain frame identity, presentation state, calibration, masks,
  fill values and live common/local metadata. No cache or format coverage changes.
  Local validation passes 1,331 tests on py314 and 261 image/format/HDF5 tests with
  the supported lower dependencies. Pushed revision `9b42bfd74` also passes all
  1,331 tests in every Linux 3.12-3.14/macOS 3.14 hosted job; lower dependencies
  pass 1,330 with one expected skip. Installed wheel/sdist checks on Python
  3.12/3.14, cached documentation and final result publishing all pass. The macOS
  Attocube test passes in 45.787 seconds, versus 201.434 in the previous run;
  these hosted timings are observational rather than a controlled benchmark.
  The 1024 x 1024 metadata-access benchmark
  falls from 16.068 to 0.820 ms and 20,234,538 to 195,855 traced peak bytes; these
  are operation measurements, not whole-loader/macOS timings or process RSS.
  See [the storage handover](STORAGE_MIGRATION_PLAN.md#post-migration-scan-frame-access-2026-10-18)
  for the bounded contract, exact-revision hosted links and documentation-warning
  review. This optimisation is validated independently of the earlier plotting gates.

- [ ] **Code complexity remediation (non-blocking for the alpha releases):** the
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
  links and source fingerprints. Included in the published alpha releases above;
  additional wrapper families need independent bounded proposals.

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
