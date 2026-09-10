# Phase 3: repository content cleanup — 2026-09-10

Status: **Complete**. This phase removed only reviewed local or reproducible
outputs and added narrowly scoped root ignore rules. It does not remove
scientific data, the documentation plot cache, or the legacy Travis
configuration, which has its explicit retirement decision in Phase 4.

## Inventory and decisions

`inventory.json` is the machine-readable tracked-file inventory. It records
the size, provenance, reproducibility, classification and consumer of every
candidate category, plus each retained scientific or image fixture of at least
500 kB.

| Classification | Decision | Before cleanup |
| --- | --- | ---: |
| Release artefacts (`build/`, `dist/`) | Ignore; none tracked | 0 B |
| Local build/test/editor output | Remove and ignore | 15.7 kB |
| Obsolete redirect and empty Prospector report | Remove and ignore | 374 B |
| Compiled CHM, DVI and PDF documentation | Remove and ignore; rebuild from Sphinx/LaTeX sources | 1.09 MB |
| Dask worker locks | Remove and ignore | 0 B |
| Generated API stubs, profiling output and nested egg-info metadata | Ignore after validation revealed them | Local-only |
| `doc/plot_cache/` | Retain as intentional generated cache | 18.28 MB |
| `.travis.yml` | Retain pending the Phase 4 CI decision | 1.5 kB |
| Scientific and image fixtures | Retain; each consumer is recorded in `inventory.json` | 35 files at or above 500 kB |

The tracked checkout changed from 94,307,875 to 93,197,568 bytes: a reduction
of 1,110,307 bytes. Package archive sizes are deliberately not measured from
local builds: Phase 2's immutable GitHub Actions validation built and checked
the wheel and sdist without retaining artefacts in this checkout.

## Reproduction and refresh

- The retained plot cache is consumed when `READTHEDOCS=True`. To refresh it,
  use the `refresh-plot-cache` target in `doc/Makefile`; it executes the
  runnable examples and copies the generated figures into `doc/plot_cache`.
  Review the resulting changes before staging them.
- Sphinx output belongs under `doc/_build/`. CHM, DVI and PDF outputs are
  reproducible from the documented Sphinx and LaTeX sources and are ignored.
- The repository check below confirms that removed categories are neither
  tracked nor visible as unignored local output, and that the retained plot
  cache remains versioned.

```powershell
& C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python maintenance/check-repository-content.py
```
