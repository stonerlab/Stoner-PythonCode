# Column Index ownership experiment (CS4)

This isolated experiment compares the composed CS3 `DataOwner` with an actual
`pandas.Index` subclass installed on `DataFrame.columns`. Production Stoner never
imports this package. The findings and measured results live in
[the migration plan](../../STORAGE_MIGRATION_PLAN.md#cs4-handover-2026-10-27).
[CS5 records the decision](../../STORAGE_MIGRATION_PLAN.md#cs5-decision-2026-10-28)
to retain CS3 composition and reject custom Index adoption for this consolidation.
Retain this prototype as executable evidence; it does not enable a backend,
define production concatenation policy or create an integration backlog.

## Ownership and supported operations

`ColumnIndex` owns an immutable production `ColumnSchema`, reusing the CS2
resolver and CS3 schema algorithms. Its values are flat displayed headers,
including duplicates. Native `get_loc`, equality and hashing keep their exact
pandas semantics; `find_col` explicitly supplies Stoner matching and `first=True`.
Object label dtype is intentional on both pandas versions, rather than the
inferred string dtype used by ordinary pandas 3 indices. Empty axes work.

`IndexOwner` reuses production package validation, masks, typed metadata, numerical
transactions and structural operations. Its `_IndexState` derives schema from
`values.columns`; there is no stored owner schema. `_IndexField` replaces the
axis on header/role edits, and saved views follow the current owner. Immutable
records can be shared by copies; value/mask/metadata copies detach. Native Index,
DataFrame and owner pickle round trips preserve the experimental schema. This
does not change or claim to fix existing TypeHintedDict pickle coercion.

Supported owner operations include renaming, role assignment through the actual
bound Setas facade, selection/reorder/repetition/drop, insertion, clone, package
import/export, fixed-schema numerical/pandas editing, and explicit combination.
The pandas transaction commit adapter additionally checks scientific columns:
native equality intentionally ignores roles and IDs. Metadata-only edits keep
the same value frame and mask, with numerical buffer sharing verified.

Moving schema into the index removes one owner field, but still stores header
text both in records and in the pandas label buffer. Their consistency must be
checked. External masks and typed metadata still need owner lifecycle rules.

## Native pandas boundaries

The executable matrix records 26 native operations for each design and 12 paired
owner operations, including values, labels, ordered IDs/roles, exclusions, typed
metadata, types, sharing and saved views. Both owners use the same deterministic
masked integer package. A real `tests/Stoner/CoreTest.dat` fixture also checks
paired import/export; timings use synthetic narrow and wide packages.

- Native copy/deepcopy, positional selection/take/slicing, duplicate `loc`, drop,
  axis naming and pickle preserve schema on the tested versions with the hooks
  below. Repeated selection retains each original ID once and creates new IDs
  for later occurrences.
- Native rename, insert, horizontal concatenation, reindex and alignment can
  discard schema. Unknown internal constructors may produce a subclass without
  scientific records. Its schema accessor fails; copy/selection/pickle fall back
  to ordinary labels where necessary to keep native label computations usable.
- Series extraction and `to_frame`, and explicit plain export, are lossy export
  boundaries. No identity or role is reconstructed from their header text.
  Reindexing a duplicate source raises the normal pandas error.
- Row concatenation can keep the left scientific schema even when another input
  has conflicting roles or an ordinary axis. A subclass check alone is unsafe.
  These native combinations are unsupported as lossless transformations.
- Re-entry requires explicit expected transformation schema, external mask,
  metadata, dtype and fill value. Missing or conflicting schema is rejected.
  This is a transient proof supplied by the operation, not a second owner store.
  The caller must transport the mask correctly: matching shape alone cannot
  prove cell provenance. Same-shaped value-only permutations remain unsupported
  and undetectable, as in the composed baseline.
- Direct replacement by an ordinary Index is detectable and rejected at owner
  validation/export. Replacing an entire same-shaped frame with an unrelated
  valid experimental axis has the same provenance limitation; callers must use
  explicit owner/package operations.

`combine` defines an intentionally bounded policy for this experiment. Both
inputs must be explicit owners with equal dtype, fill value and typed metadata.
Rows require identical ordered IDs, roles and headers; columns require equal row
counts and remap colliding IDs in input order. Roles and masks travel with values.
Ordinary frames require a new explicit import. This is not a proposal to change
the production Data concatenation API or its metadata merge policy.

## Private pandas dependencies and complexity

The subclass constructs through `_simple_new` and reads `_values`; it overrides
`_view` and `_getitem_slice`. It also overrides public copy, take and indexing,
and supplies custom pickle reduction. `_simple_new` inside pandas can bypass
`__new__`, so it cannot receive missing records automatically. `_shallow_copy`,
`_concat`, selection and copying implementations were inspected and fingerprinted
for each pandas version; they are not a universal metadata propagation protocol.
Further native transform support would require more operation-specific adapters.

There are four experimental classes in addition to the shared production schema
and owner machinery: Index, state, field view and owner adapters. Special
deepcopy/state restoration, pandas commit, native schema-loss checks, a provenance
boundary and combination policy are still necessary. The experiment therefore
reduces independent ownership fields but adds lifecycle and propagation machinery.

## Reproduce

Use a supported Conda environment with the dependencies in `tests/test-env.yml`.
The tools verify all 39 declared specifications, report effective pandas and
pytest-timeout distributions alongside their Conda records, and record interpreter,
NumPy, module paths and SHA-256 hashes of the working production/prototype sources
and inspected pandas hooks. HEAD alone does not identify this uncommitted baseline.

From the repository root (create each output directory before using pytest):

```powershell
$env:MPLBACKEND = 'Agg'
$env:QT_QPA_PLATFORM = 'offscreen'
New-Item -ItemType Directory -Force maintenance/runs/cs4-reproduce | Out-Null
C:/ProgramData/miniforge3/Scripts/conda.exe run --no-capture-output -n py314 python -m pytest maintenance/column_index_prototype/test_model.py tests/Stoner/core -q -o addopts= --basetemp=maintenance/runs/cs4-reproduce/tmp -o cache_dir=maintenance/runs/cs4-reproduce/cache --junitxml=maintenance/runs/cs4-reproduce/pytest.xml
C:/ProgramData/miniforge3/Scripts/conda.exe run --no-capture-output -n py314 python -m maintenance.column_index_prototype.assess --output maintenance/runs/cs4-reproduce/matrix.json
C:/ProgramData/miniforge3/Scripts/conda.exe run --no-capture-output -n py314 python -m maintenance.column_index_prototype.benchmark --output maintenance/runs/cs4-reproduce/benchmark.json --repeats 9 --iterations 50
```

The lower-line run uses Miniforge `py312` plus an isolated wheel overlay under
ignored `maintenance/runs/cs4-overlay`: pandas 2.2.3, pytest-timeout 2.4.0 and pytz
2026.5. Its existing pandas 3.0.5 installation is unchanged. A Conda clone attempt
failed resolving a channel; no complete cloned environment is claimed. Recreate
the overlay with:

```powershell
C:/ProgramData/miniforge3/envs/py312/python.exe -m pip install --no-deps --target maintenance/runs/cs4-overlay pandas==2.2.3 pytest-timeout==2.4.0 pytz==2026.5
$cs4PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = 'D:/PythonCode/maintenance/runs/cs4-overlay;D:/PythonCode'
```

For the lower run, substitute `py312` in the three commands above and use a
separate output directory. Afterwards restore
`$env:PYTHONPATH = $cs4PreviousPythonPath`. The recorded lower tests select the prototype and production
`test_column_schema.py`, `test_column_matching.py`, `test_column_cardinality.py`.
The rest of that environment is unchanged Conda; the JSON reports distinguish
effective overlays from Conda records. This checks the pandas 2.2 line, not every
patch release or every supported Python/pandas combination.

Benchmark scopes: exact/substring/glob/compiled-regex resolution, schema group
reads, role edits through owner views, rename, owner reorder, deepcopy and repeated
selection. Both designs call the shared CS2/CS3 algorithms. Native pandas label
lookup is tested for compatibility, not substituted for domain matching timings.
Widths/heights are 8/4096 and 512/64. Fixtures and imports are outside the timings;
each operation warms first, then nine medians use alternating paired order.
Small operations average 50 invocations per sample; copy/selection use one.
One fresh operation is traced separately for peak Python allocations. Native/RSS
peaks are not inferred from tracemalloc. Schema validation costs are included:
the Index accessor checks that its label buffer agrees with its records.
Measurements are local to these implementations and should not be interpreted as
an inherent performance limit on all possible Index subclasses. Run version
benchmarks sequentially to avoid competing benchmark workloads.

Raw JSON, logs, JUnit and overlays remain ignored under `maintenance/runs/`.
Earlier failing runs are retained as diagnostics; the plan identifies final proof.
