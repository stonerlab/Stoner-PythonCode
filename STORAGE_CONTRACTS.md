# Storage contract inventory

This is the executable baseline for batch 1 of
[STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md), characterised on `devel`
starting at `9d36d0460`. It distinguishes behaviours to retain from observed
defects and API decisions. It is a migration specification, not a promise to keep
every current implementation quirk. The observations below describe the original
baseline; Stage 4 now implements the approved Data changes.

Batch 2 decisions are now specified in
[STORAGE_INTERCHANGE.md](STORAGE_INTERCHANGE.md). References below to decisions
needed in batch 2 describe the original observations; the new specification
settles their target behaviour. The Data contract tests now assert detached
read-only reads, preserved masks/roles through structural edits, distinct duplicate
matches, ordinary negative bounds, absolute error-column positions and masked
scalar results. Empty Data now has shape `(0, 0)`. Stage 6a now asserts a sole
xarray owner for populated standalone ImageFile, detached read-only image/mask
snapshots, direct pixel/mask writes, typed interchange, guarded editing and shared
crop handles. Stage 6b now asserts sole stack ownership, stable frame handles,
calibrated interchange, excluded-zero padding and masked reductions. Stage 6c.2
extends owner storage to Kerr, MaskStack, Attocube and Maximus. Tests in
`tests/Stoner/Image/test_specialised_storage.py` cover stable specialised items,
shared scan headers with frame overrides, HDF5 masks/fill and repeated saves,
detached named-channel views, and Boolean mask-stack transitions and padding.
Stage 6c.1 adds exact calibrated transpose, flips and quarter turns, tested in
`tests/Stoner/Image/test_storage_transforms.py`. Values and exclusions move with
axis coordinates, units and auxiliary dimensions. `ImageFile.T` returns an
independent ImageFile; shared geometry requires a clone and in-place permutations
invalidate saved crop bounds. Stage 6c.3 adds calibrated interpolation, explicit
two-dimensional physical maps, ragged map packing and operand compatibility.
`tests/Stoner/Image/test_storage_interpolation.py` covers numerical/map agreement,
mask propagation, registration shifts, detachment, output-buffer rejection and
atomic rejection of mismatched units. Interpolation excludes positions outside
the source calibration and conservatively expands exclusions for higher orders.
Current consumer integration and validation evidence are recorded in the migration
plan's handover.

Stage 6e removes `DataArray`, `ImageArray`, `KerrArray`, their descriptors and
the old `ImageStackMixin`. `tests/Stoner/core/test_storage_removal.py` audits
production classes/imports and public exports, checks that `Stoner.core.array`
cannot be imported, and verifies native NumPy result types and annotation loss
through ordinary copying. No replacement ndarray subclass is introduced.
Direct Data row/slice reads may have explicit descriptive annotations; subsequent
NumPy operations do not propagate them. Named row indexing and `records._` are
replaced by resolved numerical positions and ordinary structured-field indexing.
Calculated row angles use NumPy; Data's owner-level angle properties remain.
Masked scalar checks use `numpy.ma.is_masked`, preserving the stored raw value.

Real image, TIFF, Kerr/OCR, scan, stack, peak-finding and widget tests continue to
exercise the migrated public owners. Existing TIFF `ImageArray.dtype` metadata
keys remain readable as historical format labels, without importing any removed
class. Useful numerical functions remain ordinary helpers, not storage classes.

## How to use this inventory

The new `test_storage_contracts.py` modules exercise small, distinguishable arrays
where numerical values, masks, metadata and role positions can be asserted
independently. Existing tests retain real instrument/image fixtures and broader
scientific coverage. Do not replace those fixtures with synthetic-only coverage.

Tests named `test_legacy_*` deliberately record current observations. When an
approved migration change corrects one, replace that assertion with the intended
contract and update the corresponding decision below in the same batch. These
tests are not xfails and should not be used to demand reproduction of data loss.

## Retained behaviour and evidence

| Boundary                   | Behaviour to preserve                                                      | Principal test evidence                                   |
| -------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| Column queries             | Exact names before regex search; first string match; positional forms      | core/test_storage_contracts.py                            |
| Query failures             | Missing name: KeyError; bad regex: re.error; high index: IndexError        | core/test_storage_contracts.py                            |
| Query result shape         | String column is 1D; compiled single match is 2D; cell is scalar           | core/test_storage_contracts.py                            |
| Metadata versus columns    | String assignment writes metadata; explicit column assignment edits values | core/test_storage_contracts.py                            |
| Iteration                  | Data iteration yields rows, not pandas column labels                       | core/test_storage_contracts.py; test_Core.py              |
| Role syntax                | Strings, compressed strings, mapping/call forms and role lookup            | core/test_setas.py                                        |
| Role inference             | Multiple y columns, error/vector families, x groups, unset defaults        | core/test_setas.py; core/test_storage_contracts.py        |
| Exclusions                 | Integer value survives masking/unmasking; NaN is not an exclusion          | core/test_storage_contracts.py                            |
| Clone                      | Values, masks, headers, roles and nested metadata are independent          | core/test_storage_contracts.py                            |
| Metadata types             | Preserve explicit strings and native typed metadata on copying             | core/test_copy_into_metadata.py                           |
| Sorting/row deletion       | Values and masks move together; same-object return; roles retained         | core/test_storage_contracts.py; core/test_row_deletion.py |
| Operators                  | + appends rows, & combines columns, // selects, ~ exchanges roles          | core/test_operators.py                                    |
| Pandas values/roles        | Existing real-fixture values, headers and roles round-trip                 | core/test_storage_contracts.py; test_Core.py              |
| Image masks                | Exclusions affect reductions without losing integer pixels                 | Image/test_storage_contracts.py; Image/test_core.py       |
| Unequal stack extents      | Public order is frame/row/column; each item recovers its own extent        | Image/test_storage_contracts.py                           |
| Stack identity             | Frame names, order and metadata survive insertion/access                   | Image/test_storage_contracts.py; Image/test_stack.py      |
| Stack item edits           | Pixel and mask assignments through an item update the stack                | Image/test_storage_contracts.py; Image/test_stack.py      |
| Stack clone                | Pixel, mask and per-frame metadata changes are independent                 | Image/test_storage_contracts.py                           |
| Image dtype conversion     | Scaling rules, same-dtype sharing, explicit copy and mask retention        | Image/test_convert_copy.py; Image/test_core.py            |
| Image crop and dispatch    | Copy options and dynamic image/skimage method behaviour                    | Image/test_core.py; Image/test_listfuncs.py               |
| Kerr and format consumers  | Existing Kerr operations, OCR boundaries and scientific file fixtures      | Image/test_kerr.py; test_FileFormats.py                   |
| Persistence                | Existing metadata save, HDF5 and loader failure/cleanup conventions        | test_Core.py; test_HDF5.py; core/test_loader_cleanup.py   |

Test paths in the table are relative to `tests/Stoner/`. Tracked capitalisation is
`tests/Stoner/Image`, even though Windows may display `tests/stoner/image`.

These are representative boundaries, not exhaustive coverage of every operation
combination. In particular, file-format tests do not establish that every format
round-trips masks or arbitrary metadata. Batch 2 must specify lossless interchange
separately from existing lossy exports.

## Current observations requiring an explicit migration decision

### Selection and role propagation

For headers `['Field', 'Moment', 'Moment error', 'Moment', '2', '[']`:

- `find_col('Moment')` returns 1, while `find_col(re.compile('Moment'))` returns
  `[1, 2, 1]`: duplicate headers collapse back to their first occurrence. Preserve
  first-match string selection, but define compiled-pattern duplicate handling in
  batch 2; positional identities should permit selecting both distinct columns.
- Exact `'2'` selects its named column, and exact `'['` is accepted without regex
  compilation. Missing numeric string `'3'` raises `AttributeError` in this
  baseline; `'-1'` raises `KeyError`. Do not promise working numeric-string fallback
  until it is specified and implemented.
- Integer `-7` on six columns wraps to 5. Decide whether indices beyond the normal
  negative range should instead raise `IndexError`.
- Selecting columns `[3, 0, 2]` from roles `xyey..` moves values and headers but
  produces roles `xye`, not `yxe`. This is a propagation defect; the new schema must
  associate roles with selected columns.
- With roles `xdyexdye`, `_get_cols(startx=4)` returns x=4, y=[6], y-error=[7], but
  x-error=1 rather than absolute column 5. Correcting this requires a regression
  with the intended absolute error-column index.

### Structural edits

The new tests use a cell mask and unequal column values to expose these effects:

- `reorder_columns([2, 0, 1])` returns the original Data and reorders roles/headers,
  but drops the exclusion mask. It also removes the first entry of the caller's
  list. Retain the operation's in-place/chained return; preserve masks and stop
  mutating the caller's selector in the migration.
- `add_column(...)` appends a role but loses the existing mask in the tested path.
  Preserve existing exclusions through insertion/append in the new backend.
- `del_column(0)` retains the remaining mask but clears the remaining roles in the
  tested path. Remaining columns must retain their roles in the new schema.
- `sort(...)` does retain the tested cell mask, headers and roles and returns the
  same object, despite prose describing a copy. The test asserts object identity;
  wrapper work must follow the actual retained contract.

### Array ownership and scalar extraction

Basic Data row slices and `column(...)` write through to their owner; advanced row
selection is independent. Existing ImageArray crop tests also distinguish shared
and independent storage. Preserve these observations as migration inputs, but the
plan permits explicitly replacing unrestricted writable views with conversion and
editing APIs. Settle signatures and the compatibility window in batch 2.

A masked Data scalar returns its fill value without a mask flag; its hidden value
remains stored. Decide the new scalar policy explicitly rather than accidentally
substituting pandas missing-value semantics. The image mask proxy also requires an
array mask before indexed edits on a freshly constructed image whose mask is a
scalar false; the recovery test explicitly initialises that mask.

### Interchange and image padding

Current pandas export turns a masked integer cell into NaN. Import returns an
unmasked NaN and cannot recover the excluded value. Values and roles round-trip
for the ordinary real fixture, but this is not lossless masked interchange.

On the baseline pandas 3.0.5 environment, user metadata is absent after round-trip;
even a subsequent access to the existing pandas metadata accessor does not retain
an assignment from the preceding access. The existing real-fixture round-trip test
does not expose this loss because it does not add a custom metadata field. This
observation is dependency-sensitive and must be rechecked on selected versions.
Its legacy regression is restricted to pandas 3; it does not assert the same
accessor behaviour on older supported dependency lines.
Durable metadata ownership is required; no dependency-specific workaround is
introduced in batch 1.

Unequal image sizes are recoverable from stack items. Stage 6b replaces the
historical unmasked padding with excluded raw zeros. Whole-stack reductions now
ignore padding and user masks; standard error uses the included count per pixel.
`imarray` is a detached read-only snapshot. Edit through stack/item assignment or
the guarded NumPy/xarray transaction interfaces. Regression coverage is in
`tests/Stoner/Image/test_stack_bridge.py` and `test_storage_contracts.py`.

## Validation and resumption

Environment checked against all 36 dependency entries in `tests/test-env.yml`:
Miniforge `py314`, Python 3.14.7, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0,
Matplotlib 3.11.1, scikit-image 0.26.0 and pytest 9.1.1. Existing Conda package
records satisfied the listed minimum versions, including optional test features.
No interpreter, dependency specifications or CI workflows were changed.

Use the existing runner, which activates Conda and sets headless plotting/Qt:

```powershell
# New contract cases only
./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths tests/Stoner/core/test_storage_contracts.py,tests/Stoner/Image/test_storage_contracts.py

# Broader scientific and consumer baseline
./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths tests/Stoner/core,tests/Stoner/test_Core.py,tests/Stoner/Image,tests/Stoner/analysis,tests/Stoner/test_Analysis.py,tests/Stoner/test_FileFormats.py,tests/Stoner/test_HDF5.py,tests/Stoner/folders,tests/Stoner/plot
```

Before additions, core, `test_Core.py` and Image tests passed: 190 tests, 10
warnings. The broader consumer baseline passed 379 tests with 173 warnings in
329.81 seconds. After restricting the dependency-sensitive metadata observation
to pandas 3, the final contract-only run passed all 36 cases with one warning.
Warnings include SciPy ODR deprecation, non-interactive plotting, nptdms dtype
deprecation and existing numerical fallback warnings; none was globally hidden.
The suite of documentation examples and unrelated tools tests was not run in
this focused baseline. Local evidence paths are in the plan's current handover.
The runner stores local logs and JUnit XML under ignored `maintenance/runs/`.
Use Conda activation through the runner: a direct unactivated interpreter probe
did not complete successfully and is not validation evidence.

This is Windows/Python 3.14 baseline evidence, not Python 3.12, hosted CI, xarray
compatibility or migration performance evidence. It is retained as historical
context; use the migration plan for current primitive and consumer validation.
Wrapper expansion remains gated on completing storage validation.
