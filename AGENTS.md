# Stoner repository guidance

## Repository purpose and layout

Stoner is a scientific Python data-analysis package for experimental condensed-matter physics. The main public classes are `Data`, `DataFolder`, `ImageFile`, `ImageFolder`, and `ImageStack`.

- `Stoner/core`: numerical data, masked arrays, metadata, column roles, operators, and the main `Data` class.
- `Stoner/analysis`: transformations, filtering, feature extraction, fitting infrastructure, and physical models.
- `Stoner/formats`: registry-based data and image loaders/savers. Preserve loader identification and priority behaviour when changing formats.
- `Stoner/folders`: hierarchical collections, grouping, bulk operations, and HDF5/ZIP-backed folders.
- `Stoner/Image`: image arrays/files/folders/stacks, Kerr microscopy, masks, and interactive selections.
- `Stoner/plot`: plotting methods and packaged Matplotlib styles.
- `Stoner/tools`: shared decorators, options, file utilities, widgets, formatting, and test helpers.
- `tests`: pytest suite and test data.
- `doc`: Sphinx user guide, API documentation, and runnable examples under `doc/samples`.
- `sample-data`: representative experimental files used to exercise the format loaders.

`Data` deliberately presents a broad, chainable, mostly in-place API. Many methods are dynamically attached from modules by `class_modifier`; search the analysis, fitting, plotting, and core method modules before concluding that a method is missing from the class body.

## Development environment

- The package requires Python 3.11 or newer and CI covers Python 3.11-3.14.
- Use a supported Conda distribution such as `C:\ProgramData\miniforge3` or `C:\ProgramData\Anaconda3`, with Python 3.11 or newer and the required project dependencies. Detect which installation is present rather than assuming one path is available on every machine.
- Ignore `C:\ProgramData\Miniconda3` for Stoner development. It is intentionally fixed at Python 3.6 for LabVIEW 2018 integration and must not be upgraded, repurposed, or treated as the project's test environment.
- Prefer the existing Conda environments named `py314` or `py313` when they are present and contain the dependencies needed by the task. Verify their Python and package versions before use; do not assume an environment is complete solely from its name.
- Existing base or named environments are not guaranteed to contain the complete test dependencies. Use or create an environment from `tests/test-env.yml` before claiming local test success.

Typical validation commands using Miniforge, from the repository root, are shown below. Substitute `C:\ProgramData\Anaconda3\Scripts\conda.exe` on a machine using Anaconda:

```powershell
C:\ProgramData\miniforge3\Scripts\conda.exe env create -f tests\test-env.yml
C:\ProgramData\miniforge3\Scripts\conda.exe run -n test-environment python -m pip install --no-deps .
C:\ProgramData\miniforge3\Scripts\conda.exe run -n test-environment python -m pytest
```

Prefer focused tests while developing, followed by the full suite when practical. Tests involving plotting or widgets may require an appropriate headless Qt/Matplotlib configuration. Do not turn local results into claims about GitHub Actions, Coveralls, Codacy, or physical instrument/data compatibility.

Use the exact fixture path capitalisation recorded by `git ls-files`: the tracked
test directory is `tests/Stoner`, even if Windows displays it as `tests/stoner`.
Windows test passes do not detect case mismatches that fail on Linux. See
`maintenance/phase7/ci-fixture-paths.md` for the 2026-09-13 CI correction.

## Git and checkout caveat

This checkout is created with Cygwin Git. Cygwin may record tracked files as executable (`100755`), while Windows Git reports them as ordinary files (`100644`). This can make essentially the entire repository appear modified with zero inserted or deleted lines.

Before editing or committing:

- Verify the current branch, remote, and status.
- Distinguish mode-only or line-ending noise from semantic changes.
- Do not normalize permissions or line endings across the repository as part of an unrelated task.
- Preserve unrelated changes and generated artefacts already present in the working tree.
- The normal release-oriented branch is `stable`; do not assume another branch without checking.

### Agreed checkout workflow (2026-09-10)

Use Windows Git for this checkout with repository-local `core.fileMode=false` and `core.autocrlf=false`. These settings were approved by the maintainer and tested on a disposable clone. They preserve the existing mixed line endings and indexed executable bits; do not run repository-wide normalization or `git add --renormalize` as routine maintenance. Review `git diff --ignore-space-at-eol` to distinguish semantic changes from the pre-existing one-line CRLF/LF difference in `Stoner/formats/utils/__init__.py`. Stage only intended changes.

Cygwin contributors should use a separate checkout rather than alternate clients in this working directory. Cygwin is not installed here and was not live-tested. See `maintenance/phase1/README.md` for evidence and rollback commands.

## Behaviour and compatibility expectations

- Preserve the metadata dictionary, masked-array semantics, column-role (`setas`) behaviour, and in-place/chained-operation contract of `Data`.
- Format loaders should positively identify supported inputs and reject mismatches with the established Stoner load exceptions. Maintain registration patterns, MIME types, and priority ordering.
- Keep sample-data and regression fixtures when they protect real instrument or facility formats; do not replace format coverage with synthetic-only tests.
- Treat `.ipynb_checkpoints`, `build`, `dist`, and generated documentation as historical/generated content. Do not delete or refresh them unless the task explicitly includes repository cleanup or documentation regeneration.
- Retain the generated files in `doc/plot_cache`. They are an intentional repository cache used to avoid repeatedly executing expensive plotting examples during documentation builds.
- Distribution artefacts are built by GitHub Actions, not locally. Do not retain locally built files under `dist` in the repository or treat an old artefact as evidence that current source has been packaged successfully.

## Packaging and documentation checks

When changing dependencies or preparing a release, reconcile all relevant sources rather than editing only one:

- `pyproject.toml`
- `requirements.txt`
- `recipe/meta.yaml` and `recipe/build-env.yml`
- `tests/test-env.yml`
- `doc/docs-env.yml` and `doc/requirements.txt`

Known consistency issues to verify rather than perpetuate include:

- The project licence is GPLv3; keep `pyproject.toml`, `LICENSE.md`, `COPYING`, and the Conda recipe consistent with it.
- Keep required runtime dependencies aligned across `pyproject.toml`, `requirements.txt`, and the Conda recipe. Testing and documentation environments may intentionally add role-specific dependencies.
- Project URLs should use the current `stonerlab/Stoner-PythonCode` repository. Do not reintroduce legacy personal-repository URLs.
- Package builds should be checked for non-Python runtime assets, especially `Stoner/plot/stylelib` and `Stoner/Image/tessdata`.

Conda recipes and environment files should remain as pure Conda specifications as practical. Search `phygbu` and `conda-forge` for uncommon packages before adding a `pip:` subsection; report any dependency that is genuinely available only from pip as a packaging defect rather than silently mixing installers.

Documentation is built with Sphinx from `doc`. Examples under `doc/samples` are part of the user-facing documentation and test surface, so update examples and prose when public behaviour changes.

Read the Docs sets `READTHEDOCS=True`; that build must consume the retained files in `doc/plot_cache` rather than execute the plotting examples. Documentation builds without that environment flag deliberately execute the examples and refresh the cache.

For visual verification, serve the generated Sphinx HTML directory over localhost rather than opening `file://` URLs in the browser. From the repository root, run `python -m http.server 8765 --bind 127.0.0.1 --directory doc/_build/html` with a supported Python interpreter, substituting the actual build output directory and an available high-numbered port. Serve only the generated HTML directory, not the repository root. Open `http://127.0.0.1:8765/index.html` in the browser, inspect the rendered pages and follow relevant navigation links. Reload after rebuilding. Keep the server available while the user is reviewing the preview, then stop it when no longer needed. This workflow was verified on 2026-09-12 after direct local-file navigation was rejected by the browser URL policy; localhost HTTP worked without a policy change. Visual inspection complements Sphinx warning and API-inventory checks.

For visual verification, serve the generated Sphinx HTML directory over localhost rather than opening `file://` URLs in the browser. From the repository root, run `python -m http.server 8765 --bind 127.0.0.1 --directory doc/_build/html` with a supported Python interpreter, substituting the actual build output directory and an available high-numbered port. Serve only the generated HTML directory, not the repository root. Open `http://127.0.0.1:8765/index.html` in the browser, inspect the rendered pages and follow relevant navigation links. Reload after rebuilding. Keep the server available while the user is reviewing the preview, then stop it when no longer needed. This workflow was verified on 2026-09-12 after direct local-file navigation was rejected by the browser URL policy; localhost HTTP worked without a policy change. Visual inspection complements Sphinx warning and API-inventory checks.

## Docstring standard

Follow [DOCSTRING_STYLE.md](DOCSTRING_STYLE.md) when writing or revising docstrings. It defines the expected Google-style
sections, Sphinx markup, argument and return descriptions, and scientific behaviour documentation. Use British English
spellings for prose while preserving the exact spelling of API identifiers, keyword arguments and literal values.

## Maintenance plan

`MAINTENANCE_PLAN.md` is the working source of truth for repository maintenance and cleanup. Follow its phase ordering, update its status/evidence as work is completed, and do not perform later cleanup phases before the baseline and required decisions are established.
