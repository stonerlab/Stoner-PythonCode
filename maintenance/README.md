# Maintenance tools

Run these commands from the repository root in a supported Conda environment.
The recurring checklist and current release work are in
[MAINTENANCE_PLAN.md](../MAINTENANCE_PLAN.md). Local output belongs under ignored
`maintenance/runs/`; completed investigation records belong in Git history.

## Tests and source checks

The isolated [storage composition prototype](storage_prototype/README.md) has
acceptance tests and a paired benchmark runner. It remains outside the production
package during the pandas/xarray migration.

```powershell
./maintenance/run-baseline.ps1 -Check focused -TestPaths tests/Stoner/test_Core.py
./maintenance/run-baseline.ps1 -Check serial
./maintenance/run-baseline.ps1 -Check parallel
python maintenance/check-ci-config.py
python maintenance/check-repository-content.py
python maintenance/check-package-contents.py
python maintenance/audit-binding.py maintenance/runs/binding.json
```

The baseline runner finds a supported Conda installation and defaults to `py314`;
use `-Environment` or `-EnvironmentPrefix` for another verified environment.
It isolates pytest temporary files, coverage and logs in a timestamped run directory.
See [tests/README.md](../tests/README.md) for marker and environment guidance.

`check-package-contents.py` inspects setuptools discovery and a temporary metadata
manifest; it does not build a distribution. Its report goes to
`maintenance/runs/package-contents.json`. The binding audit reports signatures,
discovery, documentation and annotations; inspect its JSON for failures and gaps.
It is a diagnostic report, not a pytest substitute or an automatic pass/fail gate.

## Documentation

```powershell
New-Item -ItemType Directory -Force maintenance/runs/docs | Out-Null
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/docs/html -w maintenance/runs/docs/warnings.log
python maintenance/audit-docs.py maintenance/runs/docs/html maintenance/runs/docs/warnings.log maintenance/runs/docs/audit.json --expected-warnings maintenance/docs/expected-warnings-windows.json
Remove-Item Env:READTHEDOCS
```

On Linux use `expected-warnings-linux.json` and add `--require-fitting-functions`.
The reviewed warning ceilings are live configuration: retain them and change them
only after investigating the diagnostics. `--compare <previous-html-build>` checks
for removed API inventory entries; use `--allow-removed <reviewed-json>` only for
explicitly reviewed obsolete targets. Do not suppress warnings by resetting the
baseline. Preserve the committed plot cache in cached builds.

## CI package and release checks

`check-distribution-archives.py dist` inspects CI-built wheel/sdist contents.
`check-installed-package.py <checkout>` runs with each installed distribution from
outside the checkout, checking import provenance, metadata, resources and sample
data. The `Package validation` workflow provides those fresh environments; use its
results rather than building distributions locally.

`check-release-version.py --tag <tag>` compares the proposed release tag with the
literal version in `Stoner/__init__.py`, without importing package dependencies.
Release jobs invoke it before their builds. The Conda recipe separately tests
the installed import and version metadata. Consult the
[developer guide](../doc/UserGuide/developer.rst) for the release process.
