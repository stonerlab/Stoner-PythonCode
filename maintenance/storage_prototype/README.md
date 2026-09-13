# Storage composition prototype

This is batch 3 of [the storage migration plan](../../STORAGE_MIGRATION_PLAN.md).
It is outside `Stoner` and is not imported by production classes. `Table` and
`Stack` exercise [the ownership specification](../../STORAGE_INTERCHANGE.md),
rather than reproducing the entire public API. Retain this experiment until the
production migration has equivalent regression and benchmark coverage.

## Run and reproduce

Use an activated supported Conda environment containing `tests/test-env.yml`
dependencies plus xarray and psutil. The benchmark checks all declared Conda test
dependencies and records runtime versions. For this run, xarray 2026.7.0 was added
from conda-forge to existing Miniforge `py314`; no other installed packages changed.
Prototype dependencies are not yet production release requirements.

```powershell
./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths maintenance/storage_prototype/test_model.py,tests/Stoner/core/test_storage_contracts.py,tests/Stoner/Image/test_storage_contracts.py
C:\ProgramData\miniforge3\Scripts\conda.exe run --no-capture-output -n py314 python -m maintenance.storage_prototype.benchmark --output maintenance/runs/storage-prototype-final.json --repeats 5
C:\ProgramData\miniforge3\Scripts\conda.exe run -n py314 python -m maintenance.storage_prototype.assess maintenance/runs/storage-prototype-final.json
```

`model.py` supplies the experiment, `test_model.py` its acceptance tests, and
`benchmark.py` paired fresh-process measurements. Raw JSON, JUnit and logs remain
under ignored `maintenance/runs/`. The migration handover retains current evidence
and the performance decision; do not add a separate completed phase report.

## Implemented boundaries

- DataFrame values with UUID labels, duplicate displayed headers, explicit masks
  and homogeneous numeric dtype, including zero-column tables. Packages carry
  dtype explicitly because empty frames cannot recover it from column dtypes.
  Schema selection moves roles and masks together.
- Shared exact-name/pattern/position resolution, distinct duplicate matches,
  repeated-selection IDs and absolute error positions in assigned x groups,
  including repeated y and vector roles.
- Detached package import/export, typed metadata and nested-copy independence,
  small-integer fill values, masked integer recovery, NaN distinction, explicit
  native pandas index discard and legacy MultiIndex import.
- Fixed-shape/dtype NumPy, pandas and xarray drafts, rollback, retained-draft
  independence and nested-owner locks. Saved metadata mapping references respect
  locks; nested metadata values remain outside numerical transactions.
- Numeric row appends through pandas concatenation, avoiding conversion of the
  existing table to a temporary MaskedArray on every append.
- Eager xarray stacks with valid rectangles, excluded zero padding, independent
  frame metadata and handles that follow frame identity and fail after deletion.
  Interior frame commits stage typed buffers before writing the validated
  rectangle without copying the entire stack, affecting neither padding nor axes.
- Weighted multi-y coefficient/covariance comparisons using `CoreTest.dat`, plus
  the real annotated image fixture with explicitly supplied physical coordinates.
  The latter checks transport, not instrument calibration or rotation adapters.

The prototype does not implement production loaders, the whole public Data/Image
API, implicit role inference, all concatenation/metadata merge forms, native
xarray convenience import/export, a standalone single-image owner, crop handles,
coordinate transforms or replacement array classes. One-frame stacks exercise
the image intensity/mask boundary. Direct frame edits use fixed-dtype drafts;
whole-stack dtype promotion remains integration work. Calibrated insertion fails
explicitly until a remapping adapter exists. Metadata and mask are implemented
owner-aware interfaces; header/role mutation proxies remain batch 4 work. These
limitations are not claims of production API compatibility.

## Accessor comparison

The module registers `stoner_prototype`, not the proposed production `stoner`
name, only when explicitly imported.

```python
package = table.export_storage()
raw = package.values.stoner_prototype.column("Moment", schema=package.schema)
masked = table.column("Moment")
mean = stack.export_storage().dataset.stoner_prototype.mean()
```

The pandas accessor is longer than the owner call and returns raw values without
the external mask. It receives the schema explicitly and rejects mismatched
column order. Caching schema/metadata in it would weaken ownership. Recommendation:
keep package conversion and owner methods as the initial Data interface; defer
a richer pandas accessor.

The xarray Dataset contains intensity and exclusions, so a selected mask-aware
accessor is useful. For six valid pixels plus twelve valid pixels of value one,
the accessor mean is 1.0; native intensity-only mean over padded storage is 0.75.
Recommendation: consider selected mask-aware xarray operations after production
storage validation. No blanket forwarding or automatic metadata recovery is implied.

## Benchmark interpretation

The recorded Stage 3 results predate the Stage 4 default-backend switch. The
benchmark's `legacy` branch imports the working checkout's `Data`, so rerunning
it after that switch does **not** measure the old array backend. Preserve the
recorded JSON and source hashes as historical evidence. A new integrated
comparison must run its old-backend side from the pre-migration checkout in an
isolated process; do not relabel current Data measurements as legacy results.

Each workload/backend/size runs in a fresh process with one warm-up and five timed
repetitions, rebuilding mutable input outside each timed interval. Both paths use
the same fitting function and seeded data. Tables are 10,000 and 100,000 rows by
eight float64 columns; stacks are 8 x 128 x 128 and 16 x 512 x 512 uint16 pixels.
Row append measures 25 sequential single-row appends; stack insertion adds one
equal-sized frame. Unequal sizes and exclusions are tested separately. Loading
uses `CoreTest.dat` through the existing loader plus a prototype conversion bridge;
this does not establish performance of all scientific file formats.

Timing excludes imports, input preparation and memory instrumentation. A separate
operation measures tracemalloc peak allocations and process RSS sampled every
1 ms. JSON includes absolute sampled RSS peak, its increase from prepared input,
raw samples, worker stderr, source hashes and environment. Short RSS peaks can be
missed; allocator reuse can give zero increase; tracemalloc misses some native
allocations. Neither is a universal memory bound. Do not sum unrelated peaks.

Construction compares full legacy Data initialisation with the narrower prototype;
its speed does not predict complete migrated Data speed. Image reduction includes
the prototype's detached adapter; legacy uses its existing masked reduction.
Conversion workloads isolate detached MaskedArray costs. Source hashes must agree
across workers; exploratory measurements made during edits are not final evidence.

Accepted production-entry tolerances are in the migration handover. They are review
thresholds for these workloads, not promises about arbitrary sizes, platforms,
full APIs or hosted CI. Rerun workloads at later integration gates.
