# Phase 2: package contents — 2026-09-10

Status: **In progress**. Package discovery and source-manifest checks pass; CI-built distribution and clean-install validation remain outstanding.

## Changes

- `pyproject.toml` now declares the resources under their actual packages: seven `Stoner.plot/stylelib/*.mplstyle` files and three `Stoner.Image/tessdata/*` files.
- Disabled implicit namespace-package discovery and implicit inclusion of manifest data in wheels. Resource directories are data belonging to real Python packages.
- `MANIFEST.in` explicitly includes `LICENSE.md`, excludes generated `doc/_build` output, and excludes both bytecode and pytest's temporary `.pyc.<pid>` files. Source tests, scientific fixtures, documentation sources and the intentional plot cache remain included.
- `Stoner.tools.tests` is an intentional package helper and remains included. The old report's statement that a wheel contained tests is not a reason to remove this public helper.

## Verification

Run from the repository root:

```powershell
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python maintenance/check-package-contents.py
```

The check uses setuptools' resolved module/resource discovery and a fresh temporary egg-info manifest. It verified 87 Python modules, all ten required resources, both licence files, retained scientific fixtures, and absence of bytecode and generated Sphinx build output. See `package-contents.json`. The source manifest contains 1351 entries in this working checkout; that count is evidence, not a fixed package-size requirement.

No wheel, sdist or Conda package was built or installed locally. AGENTS.md states that distribution artefacts are built in GitHub Actions. Existing release workflows publish packages, so they were not dispatched merely to obtain a validation build. Complete phase 2 with a non-publishing CI build and clean-install smoke checks from the committed maintenance changes before progressing to phase 3 cleanup.

Setuptools also reports deprecations for the existing licence-table/classifier representation. The licence itself remains GPLv3; a future metadata-syntax update must align the minimum setuptools version across the build declarations.
## Prepared CI gate

`.github/workflows/check-packages.yaml` builds wheel and sdist on Python 3.14 without publishing them. `check-distribution-archives.py` inspects both archives for resources, licences, helper modules and unwanted caches. Each distribution is installed in its own new virtual environment; `check-installed-package.py` runs outside the checkout, rejects accidental source imports, and checks Data construction, a representative TDI loader, Matplotlib styles, OCR-resource readability and TIFF round-tripping.

Python syntax and workflow YAML were checked locally. The probe's API/resource checks passed against the source checkout. These are preflight checks only: the workflow has not run, and clean installed-artifact results remain unverified until the maintenance branch is committed and pushed. No release/publishing workflow was triggered.