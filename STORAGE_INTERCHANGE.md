# Storage interchange and ownership specification

Batch 2 of [the migration plan](STORAGE_MIGRATION_PLAN.md), specified on
2026-09-13. This document defines the target contract for the batch 3 prototype
and subsequent migration. Batch 3 exercised their ownership rules in an isolated
[prototype](maintenance/storage_prototype/README.md). Stages 4-6 now implement
the production owners: ordinary Data construction uses pandas storage, while
images and stacks use xarray storage. Exported packages preserve stable identities.
Stage 6e removes the old array subclasses and their storage fallback paths.
Current behaviour remains recorded in [STORAGE_CONTRACTS.md](STORAGE_CONTRACTS.md).
Completion of this specification does not establish backend correctness or speed.

Stage 7 validation is complete. The expanded
[Stage 8 plan](STORAGE_MIGRATION_PLAN.md#stage-8-selection-and-masking-contract)
defines native public storage, explicit per-operation selections and persistent
recoverable exclusions. Stage 8a below settles labels, alignment and direct
mutation rules. Stage 8c/8d now expose actual ordinary backend objects at
`Data.data`, `ImageFile.image` (also `.data`) and `ImageStack.stack`. Numerical
snapshots remain explicit through `asarray()`, `imarray` and `to_numpy()`;
editing contexts remain detached and atomic. Stage 8b implements `where`
for `Data.curve_fit`, `ImageFile.mean` and `ImageStack.average`. The batch notes
immediately below record intermediate snapshot states; the implemented
native-access decisions supersede them. Stage 8f validation is recorded in the plan.

Stage 8c.1 now prepares the Data owner boundary: its authoritative frame and
`edit_pandas()` drafts use displayed headers, while version-1 packages retain
UUID-labelled columns. `Data.data` remains a numerical snapshot until the later
consumer/getter batch. This distinction supersedes the internal-label description
below without changing the lossless package format.

Stage 8c.2 adds the internal `DataOwner.frame` getter, `validate()` and
`replace_pandas(frame)`. The getter returns the stored ordinary DataFrame without
copying it or exporting NumPy values. Replacement requires identical axes and
column dtypes, copies the incoming frame, and retains exclusions, schema, metadata,
fill value and the explicit dtype. It never imports side state from frame attrs.
Use explicit owner operations or package import for structural/dtype changes.
Owner and current Data entry boundaries reject detectable label, shape and dtype
drift with `ValueError`; saved mask/schema/metadata proxies validate on use too.
Validation does not infer same-shaped value permutations or control previously
returned raw objects. New native getter calls and replacements are blocked during
editing contexts. Aborted drafts leave the current frame live; successful numerical
or frame replacements retire it. Reacquire after any owner mutation, including
those which may retain the same frame. These are internal prerequisites:
`Data.data` remains a numerical snapshot until 8c.3.

Stage 8c.3a starts consumer migration with array/column extraction, search, column
iteration and equality. They use explicit owner or masked NumPy conversions,
independently of the public `.data` property's type. `asarray()` retains its
annotated read-only snapshot contract. Search exports once for result assembly;
multi-column results preserve exclusions and raw hidden values in positional
rectangles. Boolean lists/arrays are positional selectors, padded with false when
short. NumPy integer/floating targets work as scalar numerical predicates. Existing
two-item numeric-list interval semantics and equality's independent NaN filtering
remain for compatibility. Rolling windows and remaining consumers still precede
the public native getter switch.

## Stage 8a decisions (2026-09-20; implementation follows in 8b-8e)

This section specifies the native interface now implemented in Stage 8. The
original design batch itself introduced no getter or selection implementation.
The subsequent 8b implementation supplies the operation-selection contract below
for its three initial consumers. Other operations retain their existing interfaces
pending the 8e audit. Fitting uses independent eligible rows per y channel and
publishes result columns/metadata only after every fit and draft update succeeds.

### Operation selection

- The keyword is `where=None`, meaning all observations. A table selector is a
  one-dimensional Boolean array of length `nrows`, or a Boolean Series whose
  unique index exactly equals the table index, including order. Image and stack
  selectors have the full intensity shape; a DataArray must have the same ordered
  dimensions and dimension coordinates. Supplied auxiliary coordinates must also
  match. Reject missing Boolean values, numeric truth values, scalar selectors,
  wrong shapes and mismatched labels with `ValueError` before any mutation.
  Do not silently align, reorder or broadcast. A caller can explicitly broadcast
  a spatial region across frames with xarray before passing it to Stoner.
- Resolve each call independently. There is no stored active selection, named
  selection registry or selection context in Stage 8. Rebuild positional selectors
  after structural changes: an ordinary Boolean array has no provenance, and a
  same-length stale selector cannot reliably be detected. Labelled selectors catch
  label mismatches, not every change to the observations behind those labels.
- Keep `bounds` as the existing fitting row-selector argument, including its
  existing search forms. When supplied with `where`, intersect them. Do not change
  a numeric pair from its historical row-range meaning to parameter bounds.
  For `curve_fit`, introduce `parameter_bounds` to forward optimiser bounds under
  SciPy's `bounds` keyword after resolving the row selection. Consume Stoner's
  selection keywords before backend dispatch. Other fitting engines retain their
  own parameter-constraint interfaces.
- For each y fit, intersect the selection with exclusions and finite-value checks
  for only its required x, y and uncertainty inputs. Resolve multiple y fits
  separately. External uncertainty arrays must match the complete input length
  before selection; select them by the same positions. Reject finite, non-positive
  uncertainties among otherwise eligible observations. Drop non-finite required
  inputs; do not drop a row for an unused channel's mask or NaN.
- `curve_fit` requires more eligible observations than free parameters, even when
  a backend could return a fit with an undefined covariance. Empty/insufficient
  input and invalid uncertainties raise `ValueError`. Compute all requested fits
  before publishing result columns or fit metadata, so a failed later channel
  leaves the owner unchanged. Retain existing output modes and chaining.
- Fitted parameters describe the selected inputs. Explicit model evaluation uses
  the full raw x axis; a stored model column is excluded only for excluded or
  invalid model inputs, not for unselected rows or excluded measured y values.
  Residual columns additionally inherit the measured y exclusion/invalidity.
  Neither evaluation nor fitting overwrites source values or source exclusions.
- The first regional reduction is `ImageFile.mean(where=...)`; also add `where`
  to `ImageStack.average`. Both omit excluded/non-finite values and stack padding.
  An empty image mean returns floating NaN; an empty stack output pixel contains
  raw NaN and is excluded. Existing result metadata/coordinates and weight rules
  remain in force. Unsupported methods must reject `where`, not forward it to a
  backend accidentally. Neighbourhood methods get individual policies in 8e.
  Audited table transforms also reject unsupported `where` before work or mutation.
  Pass a row callback's own `where` argument through `Data.apply(..., _extra={"where": value})`
  rather than as a top-level selection keyword. `Data.select` retains its distinct column-query
  namespace: `select(where=value)` queries a column named `where`, not an operation
  selection. Plotting backend keywords likewise do not imply a Stoner input selection.

### Native values, labels and exclusions

- `Data.data` will return the authoritative ordinary DataFrame, with displayed
  headers as columns. Duplicate headers remain legal: pandas named access may
  return several columns; `.iloc` selects one unambiguously. Stable column UUIDs
  and roles remain in the owner schema and move with owner schema operations.
  Rename retains identity; insertion creates identity; deletion removes it.
- Retain the positional RangeIndex for this interface switch. `.loc` works with
  those row labels; physical x values remain data columns, including repeated
  sweep values. Arbitrary external row indexes remain a later extension, not a
  hidden change to import semantics. Owner row reorder moves exclusions alongside
  values and resets the RangeIndex. Duplicate-header column reorder uses positions
  or stable IDs, never an inferred correspondence between duplicate strings.
- The version-1 lossless package retains its UUID-labelled DataFrame and schema.
  Export translates the public headers to IDs on its detached copy; import restores
  public headers from the schema. Do not change this portable contract merely to
  expose public labels. Preserve metadata, dtype, fill values and raw excluded data.
- `ImageFile.image` and `ImageStack.stack` will be ordinary xarray Datasets with
  `intensity` and Boolean `excluded`. Stacks retain `valid_height`, `valid_width`
  and stable frame IDs. Single-image access is consequently
  `image.image['intensity']`, rather than a DataArray getter which loses the
  co-located exclusion variable. `.imarray` remains an explicit numerical snapshot.
- Keep current image `(y, x)` and stack `(frame, y, x)` dimensions. Maximus and
  Attocube channels retain their existing frame identities, channel metadata and
  common spatial coordinates. Do not merge repeated channels or invent a new
  channel dimension in this switch. Any later Dataset-variable/channel-axis
  redesign requires a separate conversion contract. Physical coordinate maps and
  valid extents survive native access and lossless export.
- `.mask` remains the owner-aware persistent exclusion interface, with true
  meaning excluded. Clearing it preserves raw values/dtype but never makes padding
  valid. Direct pandas/xarray computations do not automatically honour exclusions;
  explicitly select inputs or use a selection-aware Stoner operation. NumPy export
  remains available, with masked export preserving recoverable excluded values.

### Mutation and reference lifetime

Root native getters return the stored object, without a NumPy conversion or a
substitute DataFrame/Dataset. Supported direct edits assign existing values without
changing axes, variable sets, shape or dtype (for example `data.data.iloc[2, 1] = 5`).
For images, existing intensity and Boolean exclusion cells may be edited; valid
extents, frame IDs and calibration are owner-managed. Use the parent object for
assignment, never rely on mutation of a pandas-selected Series writing back.

Direct sorting, index/column renaming, insertion/deletion, dtype changes and
variable replacement are unsupported. Use owner methods for structural operations
or construct a validated lossless storage package and import it as a new owner.
Fixed-schema `edit_pandas`, `edit_xarray` and `edit_numpy` retain detached drafts
and atomic commit/rollback. Whole-native assignment validates before publishing:
matching axes retain identity/roles/exclusions; changed axes require explicit
package import, rather than guessing how side state maps to the replacement.
Native image replacement must contain both required variables and valid geometry.

Ordinary backend objects cannot intercept every change. Detectable schema drift
raises `ValueError` at Stoner entry points, but same-shaped row permutations or
writes through a retained native reference cannot all be detected. Native writes
are not transactional and must not occur during an owner edit context or while
the owner is locked. Locks govern owner APIs, not previously obtained raw objects;
concurrent native and owner mutation is unsupported. Rollback covers the edit
draft, not independent external writes.

Reacquire native references after every owner mutation: an owner may replace its
buffers or backend object even for a fixed-shape update. A retained reference can
then refer to retired state and has no write-back guarantee. Clones and exports
remain independent. Owner frame/crop handles retain their existing live identity
contract; their native getters return current xarray views, not independent raw
stores. Direct mutation of these borrowed views is unsupported: edit through the
frame/crop owner or its transaction. This avoids depending on backend view/copy
behaviour for correctness. Test this contract on lower and current dependencies.

Acceptance examples and current-consumer observations are in
[STORAGE_CONTRACTS.md](STORAGE_CONTRACTS.md#stage-8a-selection-and-native-access-acceptance).

## Authoritative state

`Data` owns one DataFrame of raw values, a same-shaped NumPy Boolean exclusion
mask, an ordered column schema, a typed metadata dictionary and a dtype-compatible
fill value. The frame has an ordinary `RangeIndex(0, nrows)` and ordinary string
column labels containing displayed headers, including duplicates (Stage 8c.1).
Stable column IDs and roles live in the schema; lossless packages translate headers
to UUID labels on export and restore headers on import. No custom pandas index or
role-bearing MultiIndex is used internally. The public frame is authoritative raw
storage; exclusions/schema remain owner-managed and explicit exports are detached.

Each schema record contains `id`, `header` and `role`. IDs are UUID strings, unique
within an object; headers are strings and may repeat; roles are single characters
from `xyzdefuvw.`. `column_headers` and `setas` remain owner-aware interfaces to
this state, including indexed edits. `column_ids` is a read-only ordered tuple.
Production records use canonical UUID text. Boolean column selectors are rejected
explicitly rather than treated as integer positions, including through `find_col`.
Renaming and role assignment never change IDs. Role inference is computed from
the ordered schema, preserving the existing role families and multiple x groups.

`ImageFile` and `ImageStack` own eager NumPy-backed xarray Datasets, plus durable
typed metadata and fill values. Required Dataset variables are `intensity` and
`excluded`, with identical dimensions and shapes; `excluded` is Boolean. Stack
Datasets also contain integer `valid_height(frame)` and `valid_width(frame)`.
Frame records hold unique UUID IDs, display names and independent typed metadata.
These records, rather than an accessor cache or automatic `attrs` propagation,
are authoritative. An image has dimensions `('y', 'x')`; a stack has
`('frame', 'y', 'x')`. `frame` coordinate values are the frame IDs.

Every public mutation validates the complete resulting state before publishing
it. Failed validation leaves the original unchanged. Temporary numerical arrays
and edit drafts are permitted; no mutable secondary store is kept in sync with
the owner. Clones, imports and exports copy values, masks and nested metadata.
Copy failures for arbitrary metadata raise `TypeError` naming the key rather than
silently sharing an uncopyable object. Functions and immutable objects may retain
identity where Python deep-copy semantics permit it.

## Identity, selection and combinations

Rows are positional observations. Repeated, decreasing or hysteretic measurement
x values remain ordinary data columns. Sorting or deleting rows moves masks with
values and recreates the RangeIndex. No persistent row ID is promised. Imports
with meaningful external indexes must explicitly discard them or first materialise
them as value columns; the importer never guesses that an index is the x column.

Selection, reorder and deletion move schema records with values and masks. A clone
retains IDs in its independent namespace. Repeated selection of the same column
retains its ID for the first occurrence and allocates fresh IDs for subsequent
occurrences. Insertion allocates IDs. Column concatenation retains the left IDs
and regenerates colliding right IDs, including concatenating a clone. Row appends
retain the destination schema and apply the existing Stoner column-matching and
padding rules; duplicate headers are matched by occurrence in order, never by a
label-based pandas join. New columns receive new IDs. Existing operator padding
rules are not replaced by image-stack padding rules.

All existing Stoner arithmetic remains positional after Stoner resolves columns.
Internal operations must use position or deliberately normalised indexes;
backend label alignment must never silently drop, duplicate or reorder samples.
Masks undergo the same indexing as values. Elementwise arithmetic combines input
exclusions with Boolean OR; algorithm-specific filtering and uncertainty handling
retain their characterised behaviour. New coordinate-aligned APIs are deferred.

Resolver decisions, replacing the legacy cases in the inventory:

- Exact string names select the first occurrence before any regex compilation.
  Other strings use regex search and select the first match. Numeric strings are
  names/patterns only: there is no integer fallback. An unmatched `'3'` or `'-1'`
  raises `KeyError`; callers use integer positions explicitly.
- Compiled patterns return every matching position in order, including distinct
  duplicate headers: `re.compile('Moment')` resolves to `[1, 2, 3]` in the
  inventory fixture. No-match raises `KeyError`; invalid regex raises `re.error`.
- Integer positions use ordinary bounds `-n <= index < n`; overflow raises
  `IndexError`. Slices use ordinary clipped Python slice rules. Collections
  resolve and concatenate their members in order, retaining repetitions.
- Scalar versus list resolution, `force_list`, row iteration and metadata lookup
  precedence remain as characterised. A string column result is 1D and a compiled
  single-column selection is 2D. These array results are detached snapshots.
- Roles follow selected columns; masks survive insertion and reorder; deletion
  retains remaining roles. Selectors supplied by the caller are not modified.
  The second x group's x-error uses absolute column position 5 in `xdyexdye`.

## Dtypes, exclusions and numerical boundaries

The initial migration supports **homogeneous NumPy numeric storage**: Boolean,
signed/unsigned integer, floating and complex dtypes. DataFrame columns must have
the same dtype. Images retain their existing supported real intensity dtypes and
scikit-image scaling rules; this specification adds no complex image support.
Mixed numeric columns require an explicit `dtype=` conversion on import. Object,
string, categorical, datetime and pandas nullable/extension dtypes are rejected
with `TypeError`; users explicitly convert them before import. Broader mixed-type
Data support is deferred, so `Data.dtype` remains a single NumPy dtype.

Value assignment uses NumPy-compatible numeric promotion across the homogeneous
store when necessary, never pandas object fallback or silent float-to-int
truncation. Invalid non-numeric assignment fails atomically. Explicit requested
dtype conversions follow NumPy conversion semantics and are the caller's choice.
Prototype tests must cover integer overflow, signed/unsigned promotion, empty
shapes and complex inputs; numerical algorithms that require real values reject
complex input explicitly. Promotion and conversions are not intensity rescaling.

Exclusion is distinct from missingness: raw masked integers survive unchanged,
and an unmasked NaN stays unmasked. Stoner numerical adapters create temporary
MaskedArrays, preserving existing NaN handling rather than inheriting pandas'
default missing-value reductions. An all-excluded reduction returns a masked
result where the existing masked operation does so. Fill values affect explicit
filled exports, not stored values. Masked scalar selection returns `np.ma.masked`,
not the hidden value or fill sentinel; the hidden value is available through a
raw export. Setting a scalar to `np.ma.masked` excludes it without overwriting its
value. Ordinary numeric assignment clears the assigned cells' masks, while masked
array assignment copies both values and exclusions. Explicit mask assignment
changes no values. Image padding cannot be unmasked.

## Typed metadata and conflict rules

In-memory lossless interchange carries a deep copy of `TypeHintedDict`, including
explicit type hints and values such as the string `'001'`. It must not reconstruct
this state by stringifying values or reparsing ordinary dictionary assignments.
Owner metadata remains distinct from headers, roles, coordinates and backend attrs.

For Data combinations, copy left metadata then add right-only keys. On collision,
the left value and its type hint win as a unit; do not recursively merge nested
values or attempt ambiguous array equality. This applies to `+`, `&` and their
in-place forms; it is an explicit target policy to test during consumer migration.
Stack-global metadata follows the same rule, while each frame retains its own
metadata without flattening collisions. Coordinate edits do not overwrite a
similarly named metadata key. No automatic metadata-to-coordinate promotion occurs.

The lossless packages below are **in-memory interchange**, not a new file format.
CSV, pandas files, NetCDF and existing Stoner savers retain their individual
documented limitations. Arbitrary Python metadata has no universal portable
encoding. A later lossless file codec must define supported types, type hints,
versioning and rejection of unsupported values; it must not silently fall back
to pickle or strings. That codec is deferred and is not needed for batch 3.

## Conversion signatures and packages

These are target signatures, with keyword-only options as shown. Imports and
exports always detach; there is no public `copy=False` escape hatch. Import
validation errors use `TypeError` for unsupported types and `ValueError` for
inconsistent state or unsupported package versions. File loaders continue to
translate format mismatches through the established Stoner load exceptions.

```python
# Data
d.export_storage() -> DataStorage
Data.from_storage(package: DataStorage) -> Data
d.to_pandas(*, format="legacy", masked="nan") -> pandas.DataFrame
Data.from_pandas(frame, *, mask=None, setas=None, metadata=None,
                 dtype=None, index="require_range") -> Data
d.to_numpy(*, masked=True, dtype=None) -> numpy.ndarray | numpy.ma.MaskedArray
d.edit_numpy() -> context_manager[numpy.ma.MaskedArray]
d.edit_pandas() -> context_manager[pandas.DataFrame]

# ImageFile and ImageStack; each imports its corresponding package rank
obj.export_storage() -> ImageStorage
ImageFile.from_storage(package: ImageStorage) -> ImageFile
ImageStack.from_storage(package: ImageStorage) -> ImageStack
obj.to_xarray() -> xarray.Dataset
ImageFile.from_xarray(dataset, *, metadata=None) -> ImageFile
ImageStack.from_xarray(dataset, *, metadata=None, frame_metadata=None) -> ImageStack
obj.to_numpy(*, masked=True, dtype=None) -> numpy.ndarray | numpy.ma.MaskedArray
obj.edit_numpy() -> context_manager[numpy.ma.MaskedArray]
obj.edit_xarray() -> context_manager[xarray.Dataset]
```

`DataStorage` and `ImageStorage` are proposed public records in
`Stoner.core.storage` and `Stoner.Image.storage` respectively. They are versioned
packages containing independently owned, editable fields; they are not live owner
handles. Validation occurs on import, not merely on record construction.

`DataStorage` version 1 has fields `version=1`, `values` (DataFrame with ID columns
and RangeIndex), `excluded` (Boolean ndarray), `schema` (ordered records described
above), `metadata` (TypeHintedDict), `fill_value` and `dtype` (NumPy dtype). Explicit
dtype is a batch 3 refinement: zero-column frames have no column dtype to recover.
Nonempty frame dtypes must agree with it. Import requires an exact
ordered match of schema IDs to columns, unique IDs, valid roles, supported common
dtype, compatible fill value and matching mask shape. It preserves IDs and types.
Backend operations on package fields do not update other fields automatically.
Callers must transform masks and schema along with values before import.

`ImageStorage` version 1 has `version=1`, `kind` (`image` or `stack`), `dataset`,
`metadata`, `fill_value` and `frames` (empty for a single image). Each stack frame
record contains `id`, `name`, `metadata` and `fill_value`. Valid extents, physical
coordinates and calibration travel in the Dataset. Import validates rank, dimension
order, identical intensity/mask coordinates, Boolean masks, valid positive extents
within padded shape, unique frame IDs and exact frame-record order. Empty stacks
are allowed with no frame records; zero-sized image axes are rejected initially.
Padding must already be excluded; malformed packages are rejected, not repaired.

`to_pandas()` preserves the current two-level `('Headers', 'Setas')` MultiIndex
layout by default. `format="plain"` uses displayed headers, including duplicates.
`masked="nan"` replaces excluded cells with NaN in the export only, promoting its
dtype as needed; `masked="raw"` exposes hidden values without exclusions.
Neither form carries masks, stable IDs, fill values or typed metadata. Exports
issue `UserWarning` when exclusions or user metadata will be lost. Use
`export_storage()` for lossless interchange; automatic attrs are not a substitute.

`from_pandas` accepts flat string headers or the legacy two-level layout, with
explicit `setas` overriding imported roles. Other column-label formats fail until
the caller normalises them. It allocates fresh IDs, defaults masks to all false
and metadata to empty, and copies an explicitly supplied typed dictionary.
The default index must equal `RangeIndex(len(frame))`; `index="discard"` permits
other indexes and discards them positionally. Non-Boolean or mis-shaped masks
fail; NaN never implicitly becomes an exclusion. `Data(frame)` is the convenience
equivalent of this explicit importer with defaults. DataFrame attrs and the legacy
`.metadata` accessor are not read as trusted interchange state.

`to_xarray()` returns a detached Dataset with intensity, exclusions, extents and
coordinates, but no typed owner/frame metadata or per-frame fill values. It warns
when these user metadata dictionaries are non-empty. Native import requires the
same Dataset structure as above; the frame coordinate supplies IDs, and names
default to those IDs. Missing per-frame metadata defaults to empty dictionaries;
supplied frame metadata is keyed by frame ID. Native import uses dtype-default fill
values. Complete names and fill values require the storage package. Reserved
structural fields are not recovered from arbitrary attrs.

`to_numpy(masked=True)` returns a detached MaskedArray of raw values, exclusions
and fill value. With `masked=False` it returns raw hidden values and intentionally
omits the mask. Both omit schema/metadata/coordinates. An explicit `dtype` converts
the export only. Stack exports are padded; masked exports exclude padding.

## Editing and writable-view transition

Direct assignments such as `d[rows, columns] = values` and `image[y, x] = value`
remain supported. `mask` is a small owner-aware Boolean indexing/assignment proxy;
`obj.mask[...] = ...` works even when initially all false. Converting it to an
array produces a detached snapshot. No general MaskedArray emulation is planned.

`d.data`, `column(...)`, row/slice array results, image `.image` and stack
`.imarray` become read-only detached MaskedArray snapshots. Their underlying values
and masks are read-only, and even deliberately making a snapshot writable cannot
alter its former owner. Whole-property assignment (`d.data = ...`, `im.image = ...`)
remains an explicit validated replacement with the existing shape/schema checks.
For images, use `im += other` or `im.image = im.image + other`; augmented writes
through the read-only `im.image` snapshot are rejected. Saved views never write
back. `np.asarray(obj)` yields detached raw values and loses mask
information, consistent with using an unmasked NumPy boundary; numerical Stoner
consumers must use the masked adapter instead.

Editing contexts yield writable detached drafts. On normal exit they validate
and atomically copy the draft into the owner; exceptions roll back. Drafts retained
after exit cannot change the committed owner. NumPy editing includes values and
masks but forbids shape or dtype changes. Pandas editing exposes header-labelled raw
values (Stage 8c.1); it retains the original mask/schema/metadata and forbids axis or dtype
changes. It is for positional cell updates only: users must not sort or permute
values independently of the mask. Structural backend transforms instead require
an explicit storage package with consistently transformed state. Xarray editing
permits intensity/mask cell changes but forbids dimension, coordinate, extent,
variable-set or dtype changes. Padding must remain excluded.

Nested edits on the same owner and public mutations while its edit context is
active raise `RuntimeError`, preventing stale draft commits without relying on
metadata mutation counters. This includes edits through owner-aware metadata,
headers, roles, mask and frame handles. Nested mutable metadata values are outside
the transaction: contexts do not copy back metadata, so such edits cannot be
overwritten by the numerical draft. Contexts are not a thread-safety guarantee.

Stack item retrieval remains an owner-aware ImageFile handle identified by frame
ID: pixel, mask and metadata assignment updates the stack. Insertion/reordering
does not retarget a saved handle; access after frame deletion raises
`ReferenceError`. A frame clone detaches. Image crops requesting sharing use a
bounded owner-aware region handle for direct pixel/mask assignments; raw arrays
still detach. Region handles share parent metadata and the parent's transaction
lock. Nested regions resolve against the same root image. A successful whole-parent
package replacement invalidates saved regions (including same-shape replacement);
later reads or writes raise `ReferenceError`. Failed replacement leaves handles
live. Region clones detach values, exclusions, metadata and sliced calibration.
Transpose/rotation and shape-changing operations on region/frame
handles require a detached clone or an explicit parent replacement.

The compatibility window is the migration branch before its first backend
release: add conversion/editing APIs before moving consumers off array writes,
and migrate internal use sites in batches 4-6. The first backend release enforces
snapshot behaviour immediately and documents replacements; no release promises
arbitrary write-through views. The maintainer's Stage 6e decision supersedes the
earlier proposal to retain standalone array utilities for one release:
`DataArray`, `ImageArray`, `KerrArray` and the old `ImageStackMixin` are removed
before any backend test release. No renamed array subclass or public alias replaces
them. Numerical results are ordinary NumPy arrays; owner methods and ordinary
numerical helpers provide the remaining scientific behaviour. Private `_data`
and `_stack` access is unsupported.

Prepared Data row/slice results may carry `i`, `column_headers` and `setas`
annotations derived from the owner. These describe only the immediate result;
NumPy copying, slicing and arithmetic do not propagate them. Named row indexing,
computed row angles and the `._` alias are removed. Resolve column positions on
Data, compute angles with NumPy, and use normal structured-field indexing on
`records`. Search results and column exports use ordinary positional indexing.

## Image coordinates, calibration and padding

Default y/x coordinates are integer pixel positions, increasing in the existing
array row/column order. Do not flip images to adopt a different Cartesian origin.
Explicit physical coordinates carry units in coordinate attrs; these attrs are
part of the validated coordinate state. Cropping slices coordinates, transpose
swaps axis coordinates, and right-angle rotation permutes/reverses them alongside
pixels and masks. Calibrated interpolation at arbitrary angles retains the
existing physical calibration as two-dimensional `physical_y(y, x)` and
`physical_x(y, x)` coordinate maps with positional dimension axes. Spatial auxiliary
coordinates use the same mapping; scalar coordinates retain their values. Maps
use linear interpolation, with NaN outside the source and corresponding pixels
excluded even if a numerical boundary mode supplies values. Higher-order and
anti-aliased resampling conservatively expand exclusions. Custom warp mappings
must be deterministic. External output buffers are rejected. Calibrated resize
uses scikit-image interpolation instead of the unsupported masked-array resize.

Stacks use top-left valid rectangles and padding to maximum height/width. Raw
padding is deterministic zero of the intensity dtype and always excluded.
Per-frame extraction crops to the valid rectangle. Reduction over the full stack
ignores padding and user exclusions. Frame-dependent field, temperature and time
may be explicit auxiliary coordinates along `frame`; arbitrary metadata remains
in frame records. Differing frame calibration uses auxiliary physical coordinates
with dimensions `(frame, y)` and `(frame, x)`, named `physical_y` and `physical_x`,
with invalid tails NaN; y/x themselves remain pixel positions. Separable image
axes map into those coordinates on insertion and are recovered on extraction.
Incompatible physical units must be explicitly converted before insertion.

The packing mapper requires matching Dataset, intensity, exclusion and coordinate
attrs across input images; it never selects one conflicting description silently.
All images supply the same auxiliary scalar coordinate names and dtypes. These
become frame coordinates and extract back to image scalars. Missing coordinates,
reserved stack names and one-dimensional spatial auxiliary coordinates are rejected
until an explicit y/x map is supplied. Two-dimensional maps pack as `(frame, y, x)`
with NaN padding. Optional `index_y(frame, y)` and `index_x(frame, x)` retain cropped
dimension labels separately from physical maps. Physical coordinate promotion to a
NaN-capable dtype must preserve integer values exactly. Empty stacks retaining
calibration attributes apply those compatibility checks when repopulated.

Legacy positional arithmetic does not align frames or pixels by coordinate labels.
Where both operands carry explicit physical calibration, differing calibrations
raise `ValueError` before positional arithmetic; users explicitly resample first.
An uncalibrated operand follows the calibrated operand's positions. Prototype
coverage must verify frame handle writes and unequal sizes before any production
image replacement.

Registration (`align` and `correct_drift`) explicitly reconciles grids: shifted
values adopt the reference ImageFile's coordinates, or retain the source grid
when given a raw array reference. Exclusions follow the measured numerical shift.
Folder alignment uses detached reference pixels and coordinates and independent
working images; backend failures propagate with the failing image's name before
translation metadata is aggregated.

## Concrete acceptance examples

These are proposed API examples for batch 3 acceptance tests, not runnable examples
against today's package. Imports below assume NumPy as `np` and pandas as `pd`.

```python
d = Data(np.array([[0, 10, 11], [0, 100, 101]], dtype=np.int16),
         column_headers=["Field", "Moment", "Moment"], setas="xyy")
d.mask[1, 1] = True
d.metadata["run{String}"] = "001"
p = d.export_storage()
restored = Data.from_storage(p)
assert restored.column_ids == d.column_ids  # Separate ownership, same identities.
assert restored.column_headers == ["Field", "Moment", "Moment"]
assert restored.setas.to_string() == "xyy"
assert restored.dtype == np.dtype("int16")
assert restored.metadata["run"] == "001"
assert restored[1, 1] is np.ma.masked
restored.mask[1, 1] = False
assert restored[1, 1] == 100
assert d.mask[1, 1]  # Unmasking the import did not alter the source.

# Reorder all package components together, retaining each duplicate's identity.
order = [2, 0, 1]
p.values = p.values.iloc[:, order].copy()
p.excluded = p.excluded[:, order].copy()
p.schema = [p.schema[i] for i in order]
r = Data.from_storage(p)
assert r.setas.to_string() == "yxy"
assert r.mask[1, 2]
assert r.column_ids == (d.column_ids[2], d.column_ids[0], d.column_ids[1])

with d.edit_numpy() as a:
    a[1, 1] = 99  # NumPy masked assignment also clears the exclusion.
assert d[1, 1] == 99
a[1, 1] = 17
assert d[1, 1] == 99  # A retained draft is independent.
try:
    with d.edit_pandas() as frame:
        frame.iloc[0, 2] = 999
        raise RuntimeError("cancel")
except RuntimeError:
    pass
assert d[0, 2] == 11

# Duplicate scan coordinates are observations, never the pandas index.
frame = pd.DataFrame([[0, 1], [0, 2]], columns=["Field", "Moment"], index=[7, 7])
native = Data.from_pandas(frame, index="discard", setas="xy")
assert native.shape == (2, 2)
assert native.to_pandas().index.equals(pd.RangeIndex(2))
```

```python
small = ImageFile(np.ones((2, 3), dtype=np.uint16))
small.filename = "small"
small.metadata["Field"] = 10
large = ImageFile(np.ones((3, 4), dtype=np.uint16))
large.filename = "large"
s = ImageStack([small, large])
q = s.export_storage()
t = ImageStack.from_storage(q)
assert t.shape == (2, 3, 4)
assert t["small"].shape == (2, 3)
assert t["small"].metadata["Field"] == 10
assert t.to_numpy().count() == 18  # Six padded cells excluded.
assert t.to_numpy().mean() == 1.0  # Legacy unmasked padding gave 0.75.
item = t["small"]
item[1, 2] = 77
assert t["small"][1, 2] == 77
assert s["small"][1, 2] == 1
item.mask[1, 2] = True
u = ImageStack.from_storage(t.export_storage())
assert u["small"].mask[1, 2]
u["small"].mask[1, 2] = False
assert u["small"][1, 2] == 77
```

Batch 3 must also reject mismatched masks/schema, duplicate IDs, unsupported
versions/dtypes, non-range native indexes without opt-in, unmasked padding and
stale frame handles. Exercise typed metadata hints/nested independence, fill
values, NaN versus masks, legacy MultiIndex interchange and transaction failure.
Use real fitting/error-column and calibrated image fixtures alongside these small
examples. Do not rewrite the legacy regressions until production behaviour changes.

## Intentional API changes and deferred choices

The first backend release changes writable arrays to snapshots with explicit edit
contexts; masked scalars to `np.ma.masked`; invalid negative positions to
`IndexError`; numeric strings to name/pattern-only resolution; compiled-pattern
duplicate selection to distinct positions; and padded stack reductions to exclude
padding. It fixes role/mask propagation and second-group x-error indexing.
It makes native pandas interchange explicitly lossy, requires explicit index
discard, and rejects unsupported dtypes instead of opportunistic coercion.
Metadata combination becomes explicitly left-biased. Calibrated image arithmetic
rejects mismatched coordinates, and zero-sized image axes are initially rejected.
These changes require migration notes and updated consumer regressions.

Batch 2's ownership and interchange choices are settled as prototype targets.
The following remain open **beyond this specification's gate**:

- Batch 3: measured performance tolerances and whether snapshot/edit-copy costs
  require a narrower internal optimisation. Public ownership must remain intact;
  changes to these decisions require updating this document with evidence.
- Batch 3: compare optional `.stoner` accessor ergonomics against explicit package
  conversion. Accessors must refer to durable state and validate it; arbitrary
  backend transformations receive no implicit lossless-state guarantee. No blanket
  forwarding or new accessor method catalogue is approved here.
- Batch 6's non-separable representation is now implemented as explicit physical
  coordinate maps, with interpolation and operand compatibility described above.
  Physical-unit method arguments remain a separate later API decision.
- Batch 7: choose release number and tested dependency bounds, run Python 3.12
  and hosted validation, and assess Python 3.15 readiness under the plan's policy.
  Portable file codecs, mixed/extension dtypes and lazy execution remain separate
  future work rather than prerequisites for in-memory round trips.

## Upstream basis and validation boundary

The following official references were checked on 2026-09-13. Pandas documents
that Copy-on-Write prevents selected objects from mutating their parents; this
supports explicit owner assignment and detached editing:
[pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).
Its `attrs` facility is experimental, so correctness should not depend on its
automatic propagation:
[DataFrame attrs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html).
Xarray documents automatic coordinate alignment in arithmetic; the positional
Stoner contract therefore requires explicit adapters:
[xarray computation](https://docs.xarray.dev/en/stable/user-guide/computation.html).

These upstream facts motivate the design; all new API and ownership rules above
are Stoner design decisions, not claims that pandas or xarray implement them.
Batch 2 validation was source/contract review, local link and whitespace checks.
Batch 3 tests now exercise these examples' storage boundaries through experimental
Table/Stack APIs; this does not validate the proposed production API names.
The migration handover records current test and measurement evidence.
