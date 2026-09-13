# Version source and CI release validation

Reviewed 2026-09-13. The maintainer requested one editable package version and
reuse of hosted packaging evidence instead of repeating local package builds.

## Version flow

`Stoner/__init__.py::__version__` remains the single literal version definition.
The subsequent release-preparation bump changes only that canonical assignment.

| Consumer | Source |
| --- | --- |
| Runtime `Stoner.__version__` and `__version_info__` | Canonical assignment and existing derived tuple |
| Wheel/sdist metadata and filenames | Existing setuptools dynamic `Stoner.__version__` attribute |
| Conda package version | Existing recipe `load_file_regex` reading the canonical assignment |
| Sphinx full/short versions, including Read the Docs | Checkout `Stoner.__version__`, with the short version derived from it |
| GitHub release tag | Must match the canonical version, optionally prefixed with `v` |

Sphinx previously read installed distribution metadata, despite documenting the
checkout. That could label new source with an old installed version or `unknown`.
It now imports the checkout's version. Release documentation previously checked
out `target_commitish`, which could advance past the release; it now checks out
the release ref, matching the package jobs.

All three release jobs run `maintenance/check-release-version.py` before their
build/publish step. It reads the literal using AST without importing scientific
dependencies, rejects mismatched tags and accepts stable/prerelease versions.
Release tag input is passed through a quoted environment variable. Manual
workflow dispatch retains its selected-ref behaviour without requiring a tag.

The wheel/sdist installed-package probe now compares runtime and distribution
metadata versions. The Conda recipe adds an installed import and equality check
between runtime, Python distribution metadata and the rendered recipe version.
Conda's build number remains an independent packaging revision.

The contributor guide documents the version bump, matching tag and release
process. No release, tag, package build or upload was performed locally.

## Hosted evidence and Phase 8 scope

Existing `check-packages.yaml` runs on pushes and pull requests. It builds both
archives and checks contents, then installs wheel and sdist separately into new
virtual environments on Python 3.11 and 3.14, running outside the checkout with
resource and real sample-data probes. The
[run at d5c43882c](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34748453303)
passed. This establishes the existing workflow, not hosted validation of the
new version assertions in this batch.

Release-triggered `build_conda.yaml` builds/uploads Conda and PyPI distributions;
`build-docs.yaml` builds documentation. Record their actual results for a release
commit rather than duplicate those builds locally. Conda import/version tests
are narrower than the wheel/sdist resource and sample-data probes. Remaining
Phase 8 work includes release-commit evidence, uncovered archive size/metadata
review, distribution/service links and release notes.

## Local validation

- Seven regression cases pass on Python 3.14.7 (20.46 s, one warning) and
  Python 3.11.16 (20.32 s, 18 warnings), covering matching prefixed/unprefixed
  versions, prereleases, mismatches, empty tags and dependency-free source reading.
  Logs: `maintenance/runs/20260913-095110-focused-0d7fe3a9` and
  `maintenance/runs/20260913-095215-focused-8f35721e`.
- Setuptools `read_attr` resolves the canonical value from the current checkout.
- Rendered the Conda recipe using its existing regex callback and parsed the
  resulting YAML; package version and the new test command use the source value.
  This is a recipe-template probe, not a Conda build/install test.
- Parsed both release workflows and verified each job has the release-only
  version check with a quoted tag environment variable.
- Evaluated the actual Sphinx configuration with installed metadata mocked to
  `99.99.99`; full/short versions remain the source version and its first two
  components. Probe: `maintenance/runs/check-phase8-version.py`.
- Direct CLI check with the current matching tag succeeds; whitespace checks pass.

The combined release-preparation changes subsequently passed 491 tests on
Python 3.14 and 64 focused lower-dependency checks on Python 3.11. A fresh cached
Sphinx build and inventory audit passed with 106 existing warnings and no new
warnings or lost API entries. No distribution was built locally. Hosted checks
and the Conda installed-version test still require their respective CI runs.

Reference behaviour: [setuptools dynamic metadata](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#dynamic-metadata)
and [GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).
