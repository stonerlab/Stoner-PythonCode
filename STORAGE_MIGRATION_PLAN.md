# Data storage migration plan

Agreed direction: 2026-09-13. Target branch: `devel`.
Status: Stages 1-9 complete, including the Stage 9e cumulative wrapper exit gate.
Release: [`0.12.0a1`](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a1)
(GitHub prerelease, PyPI and Conda published; release CI passed). Read the Docs
tag activation remains a follow-up in `MAINTENANCE_PLAN.md`.
Target Python support: Python 3.12 and newer; Python 3.11 support ends with this
new version. Python 3.15 support is conditional on its public release and support
from the required dependencies, followed by successful project validation.

This is the working specification for a migration across several separate
sessions. Read this file and [AGENTS.md](AGENTS.md) before starting a batch.
Update the checklist and handover below in the same change as the work. The
original conversation is not required to resume the project.

## Objective and scope

Replace Stoner's custom masked-array storage with pandas storage for `Data` and
xarray storage for `ImageFile` and `ImageStack`, while preserving scientific
behaviour and minimising public API changes. Keep the public Stoner classes and
their domain methods. Some explicit, documented API changes are acceptable.

After the storage migration is validated, simplify existing Stoner methods into
thin wrappers around backend methods where appropriate, and expose selected
backend methods through Stoner wrappers with Stoner defaults, including mutation
and return conventions. Do not combine this later API expansion with the initial
storage replacement.

## Agreed architecture

### Python support policy

The migration targets Python 3.12+; Python 3.11 compatibility is not a requirement
for the new version. Use Python 3.12 as the minimum-version validation target.
Add Python 3.15 to supported-version validation once it is publicly released and
the required dependencies support it. Do not claim Python 3.15 support solely
because the package metadata permits installation on that interpreter.

Apply this policy during migration implementation across package metadata,
Conda recipes, test/documentation environments, CI matrices and contributor/user
documentation. Keep the existing release's interpreter requirements unchanged in
this planning-only update. Recheck dependency availability when expanding support;
the plan does not assert that Python 3.15 or its dependency stack is ready today.

Treat the CI transition as a concrete migration sub-batch, alongside the first
production change that raises the Python minimum, rather than leaving CI on 3.11
until final release validation. Update these verified starting points together:

- `.github/workflows/run-tests-action.yaml`: remove 3.11 from the test matrix;
  retain supported-version/platform coverage and review explicit coverage jobs.
- `.github/workflows/check-minimum-dependencies.yaml` and
  `tests/minimum-env.yml`: move the lower-bound environment, job labels and
  environment names to Python 3.12; resolve compatible lower dependency bounds.
- `.github/workflows/check-packages.yaml`: move installed-distribution endpoint
  checks from 3.11 to 3.12 and keep the upper endpoint aligned with support policy.
- `.github/workflows/build-docs.yaml` and `build_conda.yaml`: review documentation
  and release/build interpreters and referenced environments for consistency;
  an already suitable interpreter need not change.
- `maintenance/check-ci-config.py`: update its hard-coded matrix and endpoint
  expectations together with the workflows, including any intentional coverage
  interpreter changes. Update contributor guidance and maintenance instructions
  describing the old matrix.

When Python 3.15 becomes eligible, review the test matrix, installed-package upper
endpoint, platform/dependency availability and coverage configuration together.
Gate completion on the CI consistency checker and successful hosted jobs; a local
pass alone does not establish the new matrix works. Final batch 7 verifies this
transition and any remaining Python 3.15 readiness work.

### Storage and interfaces

- Use composition: `Data` owns a pandas `DataFrame`; image objects own an xarray
  `Dataset` containing intensity and an explicit exclusion mask. Individual
  variables are xarray `DataArray` objects. Retain `ImageFolder` as the general
  collection abstraction.
- Maintain one authoritative set of values per object. NumPy representations for
  numerical algorithms are temporary conversions with explicit ownership and
  writeback rules; do not maintain competing mutable array and frame stores.
- Keep Stoner column resolution above ordinary pandas indexes. Exact names take
  precedence over pattern matching. Do not make a custom pandas `Index` the
  foundation of query semantics or alter native alignment rules.
- Retain the public `setas` interface. Use an ordered schema with stable column
  identities, displayed headers and roles so duplicate names, renaming and
  reordering are representable. Final schema details belong to batch 2.
- Keep typed metadata and masks as explicit state. Accessors reference durable
  state; accessor instance caches and automatic `attrs` propagation must not be
  the only source of correctness.
- Provide optional pandas/xarray `stoner` accessors sharing the same resolver and
  domain logic. Prefer selected, documented methods over blanket forwarding or
  subclassing pandas/xarray containers.
- Begin with eager NumPy-backed xarray. Defer lazy/chunked execution until mutation
  and numerical boundaries are established.

Conceptual ownership (names are descriptive, not settled API identifiers):

```text
Data                         ImageFile / ImageStack
  DataFrame values             Dataset
  Boolean exclusion mask         intensity DataArray
  ordered column schema          exclusion-mask DataArray
  typed metadata               coordinates, extents and typed metadata
```

## Behaviour to preserve and decisions to resolve

### Columns and roles

Preserve integer/negative indices, exact names, regex/partial matching, slices and
collections through the shared resolver. The current implementation treats a
string pattern as first-match selection but a compiled pattern can return several
columns. Characterise this difference before retaining or explicitly changing it.
Record duplicate-name behaviour, invalid patterns, no-match exceptions, numeric
strings, return shapes and assignment semantics as well as successful lookup.

Preserve string, mapping and callable `setas` forms; repeated y columns;
`x/y/z`, `d/e/f` uncertainties and `u/v/w` components; multiple x groups; and
default inference. Schema edits must follow values through selection, rename,
reorder, insertion, deletion, concatenation and copying. Stable identities do not
recover information lost by arbitrary external pandas transformations.

The existing `to_pandas()` MultiIndex representation (headers and roles) remains
an interchange candidate and needs a compatibility policy. It is not automatically
the canonical storage format: changing a role should not redefine column identity.

### Masks, numerical results and metadata

An excluded value must remain recoverable by unmasking. Do not replace exclusion
masks with NaN or `pd.NA`. Define their interaction with existing missing values,
integer dtypes, fill values, reductions, fitting and uncertainty propagation.
Transform masks with values in every structural operation.

Preserve typed metadata, metadata lookup precedence, clone independence and file
round trips. Specify metadata conflict rules for combinations of objects and
serialisation limits for arbitrary metadata values. Backend-native exports must
state whether they are lossless or lossy and whether changes affect the owner.

### Mutation and public API

Preserve direct Stoner assignment and the existing method-specific in-place and
chained-return contracts. Inventory row iteration, metadata-versus-column
indexing, overloaded operators (including row append with `+`) and dynamically
attached methods before delegating to backend operations.

Unrestricted write-through array views are a candidate for deliberate API change.
Pandas Copy-on-Write conflicts with assuming a selected column updates its parent.
Specify explicit conversion/editing operations and a bounded compatibility policy
for `d.data`, saved slices, NumPy operations and mask edits. Do not recreate the
entire MaskedArray API merely to conceal the storage change. Document changes to
public `DataArray`, `ImageArray` and subclasses rather than silently aliasing them
to semantically different types.

Decide dtype policy, including whether mixed-type frames are supported initially,
what `Data.dtype` means, and how numeric algorithms reject unsuitable columns.
Keep measurement x columns distinct from pandas row identity: repeated and
non-monotonic scan coordinates must not trigger unintended alignment.

### Images and stacks

Use named dimensions provisionally `('y', 'x')` and `('frame', 'y', 'x')`, while
preserving the public positional order and existing physical orientation. Record
crop, transpose, rotation and calibration rules. Store suitable frame-dependent
field, temperature and time values as coordinates without forcing arbitrary
metadata into coordinates.

Retain differently sized stack frames using padded storage plus valid extents and
explicit padding treatment. Preserve per-frame metadata, frame names and ordering.
Verify stack-item writes, mask proxy/drawing operations, crop sharing and clone
independence. Include Kerr classes and format subclasses using `_stack` directly.

Legacy positional arithmetic must not silently become coordinate-aligned
arithmetic. Define explicit alignment for new APIs and reject unintended coordinate
mismatches where appropriate. Preserve image dtype/intensity scaling and numerical
results at scikit-image/SciPy boundaries.

## Work batches and completion gates

Each numbered batch is a bounded workstream, not a requirement to finish a large
migration in one session. Split a batch into named sub-batches in the handover when
necessary. Complete its evidence before moving to dependent work.

- [x] **1. Characterise current contracts.** Add a focused contract inventory and
  meaningful regressions for the behaviours above using the existing backend.
  Reuse current tests and real scientific fixtures. Record the verified environment,
  baseline commands and outcomes. Gate: preserved behaviours and proposed changes
  are distinguishable; no production storage changes.
  See [STORAGE_CONTRACTS.md](STORAGE_CONTRACTS.md) for the inventory, test map and
  explicitly separated legacy defects/API decisions.
- [x] **2. Specify interchange and ownership.** Settle values/mask/schema ownership,
  duplicate-column identity, row identity, dtype policy, metadata conflict rules,
  pandas/xarray import/export and writable-view transition. Define conversion and
  editing signatures. Gate: concrete round-trip examples and an explicit API-change
  list; unresolved choices and their implications are recorded.
  See [STORAGE_INTERCHANGE.md](STORAGE_INTERCHANGE.md) for target signatures,
  versioned in-memory packages, acceptance examples and deliberate API changes.
- [x] **3. Prototype and measure.** Exercise an isolated composition prototype with
  duplicate columns, multi-y fitting/errors, masked integers, unequal frame sizes
  and stack-item edits. Compare accessor ergonomics. Measure time, peak memory and
  conversion costs against the current backend for loading, column operations,
  fitting, row appends, stack insertion and image reductions. Gate: correctness
  evidence and agreed performance tolerances before production replacement.
  - [x] **3a. Isolated prototype and correctness evidence.** See
    [the prototype guide](maintenance/storage_prototype/README.md).
  - [x] **3b. Paired time/memory measurements and accessor comparison.** All 19
    workload/size pairs meet the proposed thresholds below.
  - [x] **3c. Agree performance tolerances.** Accepted on 2026-09-13 by the
    maintainer's instruction to proceed to stage 4 after reviewing the results.
- [x] **4. Migrate Data storage primitives.** Implement construction, schema/resolver,
  `setas`, masks, selection, direct assignment, copying and structural edits using
  the approved ownership model. Establish shared conversion/result-wrapping helpers.
  Gate: primitive contracts and interchange tests pass; any temporary compatibility
  layer has a stated removal condition.
  - [x] **4a. Interchange primitives.** Production `Stoner/core/storage.py` provides
    schema/package validation, shared resolution and detached NumPy/pandas
    conversions, now used by Data's default backend.
  - [x] **4b. Owner construction and mutation.** Integrate authoritative frame,
    mask/schema state, role/header interfaces and explicit editing/conversion APIs.
    - [x] **4b.1. Frame owner primitive.** Internal `DataOwner` owns a detached
      package, stable schema interfaces, masks and transactional numerical edits.
    - [x] **4b.2. Connect Data.** Replace construction/descriptors, retain full
      Setas syntax, and establish the bounded adapter for remaining legacy consumers.
      Ordinary and explicit construction use the owner. Setas syntax and the
      bounded constructor/read adapters are covered by the core gate.
  - [x] **4c. Structural edits and primitive gate.** Integrate selection, insertion,
    deletion/reordering and copies with retained contracts; document any temporary
    consumer adapter and its removal condition before moving to batch 5.
- [x] **5. Integrate Data consumers.** Migrate numerical analysis, fitting, plotting,
  loaders/savers, folders and dynamically attached methods in focused sub-batches.
  Preserve load identification, priorities and established Stoner load exceptions.
  Gate: retained Data APIs and real-fixture results pass, with documented intentional
  differences and updated examples.
  - [x] **5a. Analysis and fitting.** Explicit masked numerical boundaries, owner
    assignments and transactional outlier callbacks; preserve source masks and IDs
    when storing fits/residuals. Deduplication explicitly groups complete row keys.
  - [x] **5b. Loading and saving.** Remove the construction/loading array backend;
    loaders initialise the frame owner directly. Preserve real-format recognition,
    header defaults and typed metadata; savers export once per numerical operation.
  - [x] **5c. Folders, plotting and examples.** Numeric-only Data metadata output,
    frame/array alternatives for text, direct shape/dtype access, fixed-width header
    edits, rectangular peak selections and migrated runnable examples.
  - [x] **5d. Consumer gate.** Full local suite passed: 619 tests, including real
    fixtures and runnable examples; evidence and remaining platform limits below.
- [x] **6. Migrate image storage and consumers.** Start with ImageFile construction,
  masks and numerical adapters, then stack storage/extents and item ownership, then
  Kerr classes, drawing, transforms, folders and format subclasses.
  **Mandatory exit gate:** the old storage classes and their compatibility/fallback
  implementations are removed, and the full test suite passes on that same final
  code state. Both conditions must be verified before marking Stage 6 complete or
  entering Stage 7. Earlier passing runs with legacy classes present do not count.
  Image/stack contracts, scientific results and lossless interchange remain covered.
  - [x] **6a. Image ownership boundary.** Establish validated interchange and
    transactions, then connect ImageFile construction and public editing interfaces.
    - [x] **6a.1. Package and owner foundations.** Production ImageStorage/Frame
      records, image and ragged-stack validation, detached NumPy/xarray exports,
      typed metadata copies and guarded numerical transactions are implemented.
      ImageFile and ImageStack now use these owners.
    - [x] **6a.2. Connect ImageFile.** Migrate construction, mask/drawing proxies,
      direct pixel assignment and method adapters together. Retain detached public
      snapshots and preserve shared-crop ownership through bounded region handles.
      - [x] **6a.2a. Shared-region prerequisite.** Internal bounded and nested
        handles write pixels/masks through the parent, share transaction guards,
        preserve sliced calibration, detach clones and reject stale handles.
      - [x] **6a.2b. Public integration.** Connect the descriptor, constructor,
        mask/drawing proxies and method adapters to the owner and region handles.
  - [x] **6b. Stack ownership.** Stable frame handles, insertion/reordering/deletion,
    per-frame metadata/fill, calibration mapping and excluded-zero padding.
    - [x] **6b.1. Stable frame ownership.** Internal StackOwner and FrameOwner
      preserve saved handles across insertion/reordering, retire deleted IDs,
      share transaction locks and expose frame-backed ImageFile pixel/mask/metadata
      editing. Structural operations validate before publishing; extraction maps
      physical coordinates onto the valid image rectangle.
    - [x] **6b.2. Packing and public integration.** Implement calibrated packing
      without dropping native attributes, then replace ImageStack's legacy arrays,
      extents and item construction with the stack owner and frame handles.
      - [x] **6b.2a. Calibrated packing.** Separable physical axes map to padded
        frame coordinates; scalar calibration maps to frame coordinates. Matching
        native attributes are retained, incompatible attributes/units or coordinate
        sets fail atomically, and precision-losing physical coordinates are rejected.
      - [x] **6b.2b. Public ImageStack integration.** Replace the legacy arrays,
        extents and item construction, preserving public stack and folder contracts.
  - [x] **6c. Image consumers.** Coordinate-aware crop/transpose/rotation, explicit
    representation for non-separable transforms, Kerr and private-stack consumers.
    - [x] **6c.1. Exact coordinate permutations.** Crop/slice plus transpose, T,
      horizontal/vertical flips and CW/CCW quarter turns preserve values, masks,
      axis coordinates/units and auxiliary coordinate dimensions. In-place
      geometry changes invalidate crops; shared frame/region changes require clones.
    - [x] **6c.2. Specialised consumers.** Migrate Kerr/MaskStack and
      Attocube/Maximus private-array consumers onto the image/stack owners.
    - [x] **6c.3. Interpolation and operand alignment.** Define non-separable
      coordinate representations, migrate interpolating geometry and enforce
      calibrated operand compatibility. Physical-unit method arguments remain an
      explicit later API choice; current bounds and indices are positional pixels.
  - [x] **6d. Image gate.** Update intentional-change contracts and run the complete
    image/stack/format/scientific-example gate after switching the public owners.
  - [x] **6e. Remove legacy storage classes before any test release.** Added by
    maintainer instruction after the 6d gate. Authoritative owner replacement is
    not sufficient: remove the old array classes, their imports, descriptors and
    fallback storage paths rather than shipping them as snapshot adapters.
    - [x] **6e.1. Data return contracts and consumers.** Replace DataArray-based
      data/slice/row/role snapshots with explicit pandas/NumPy results. Keep schema,
      column roles, row identity, exclusions and fill state on the new owner or
      explicit interchange package. Migrate threshold/interpolation return values,
      column insertion, constructors, operators and format consumers; document
      intentional changes to array-specific attributes such as slice `.i`/`.setas`.
    - [x] **6e.2. Delete legacy Data storage.** Remove `Stoner/core/array.py`, its
      public exports, the old Data descriptor and obsolete `_data` fallback paths.
      Move any still-needed numerical behaviour into ordinary helpers or owner
      methods. Do not retain the implementation under a renamed compatibility class.
    - [x] **6e.3. Delete legacy image storage.** Move ImageArray/KerrArray numerical
      behaviour into ordinary numerical helpers or owner-backed image methods,
      then remove the ndarray subclasses, old ImageStackMixin storage implementation
      and remaining private-array/descriptor fallback paths. Preserve scientific
      image operations, loaders, masks and presentation behaviour through the owners.
      Earlier instructions in this plan to retain standalone array utilities are
      superseded by this removal requirement; useful functions may remain.
    - [x] **6e.4. Stage 6 exit gate.** Verify no runtime imports, exports, constructors,
      inheritance or fallback paths use the removed classes. Update public API docs,
      examples and intentional-change contracts; run focused consumer checks and the
      complete suite after removal. Record the audited code state, removal evidence,
      full-suite command and successful result together in the handover. Preserve
      historical evidence and fixtures without importing removed production classes.
      Both class removal and the full-suite pass are verified on the code state
      recorded below. Stage 7 may proceed; release readiness still requires Stage 7.
- [x] **7. Validate migration and prepare compatibility release.** Run appropriate
  full-suite, platform, dependency, documentation and installed-package checks;
  repeat representative benchmarks. Reconcile dependency specifications, including
  xarray and tested pandas bounds. Python 3.12 is now declared as the minimum
  throughout packaging, environments, CI and documentation; Python 3.11 has been
  removed from the new version's support matrix. Validate that revised matrix.
  Validate Python 3.15 when publicly released and supported
  by the required dependencies; record readiness separately if it remains pending.
  Document changed APIs and the replacement of
  custom array classes. The maintainer selected `0.12.0a1` as the migration's
  initial alpha version; the canonical package version has been updated accordingly.
  Gate: all retained contracts pass or have reviewed, documented changes; hosted
  validation is distinguished from local evidence. Publishing is a separate action.
  **Prerequisite:** Stage 6e legacy-class removal is complete, including for a test
  or prerelease build. A passing suite with the old adapters present is not enough.
  - [x] **7a. Supported environments and dependency declarations.** Python 3.12-3.14;
    pandas >=2.2 and xarray >=2024.10 aligned across runtime, Conda, test and docs
    specifications. The lower-bound job pins those backend minor lines.
  - [x] **7b. Platform and installed-package gate.** Verify the full hosted matrix,
    lower-bound suite, wheel/sdist contents and clean installs on the final code.
  - [x] **7c. Documentation gate.** Build cached HTML, audit against existing warning
    ceilings, check primary API coverage and inspect rendered migration guidance.
  - [x] **7d. Integrated performance gate.** Compare production against the isolated
    pre-migration checkout using the accepted time/allocation thresholds.
  - [x] **7e. Compatibility notes and future Python readiness.** Alpha upgrade notes
    describe removed classes, snapshot/edit semantics and the new dependencies.
    Python 3.15 remains separately pending: as of 2026-09-20 its latest official
    release is [3.15.0rc2](https://www.python.org/downloads/release/python-3150rc2/),
    with final release scheduled for 2026-10-01. No 3.15 compatibility is claimed.
- [x] **8. Expose native storage through the public attributes.** After Stage 7
  storage validation, establish the native public interface before simplifying
  wrappers. The maintainer's
  target is a pandas DataFrame at `Data.data` and native xarray objects at
  `ImageFile.image` and `ImageStack.stack`, exposing the actual backend storage
  without first rendering it as a NumPy snapshot. Settle masking and ownership
  semantics first; Stage 9 then simplifies wrappers against this validated interface.
  - [x] **8a. Specify selections, exclusions and native ownership.** Characterise
    current mask/bounds consumers, write the contracts below into
    STORAGE_INTERCHANGE.md and add acceptance examples to STORAGE_CONTRACTS.md.
    Resolve native object types, axis labels/identity and allowed direct mutations
    before switching attributes. Gate: each unresolved choice below has a concrete
    decision, worked examples and a testable failure/return contract.
  - [x] **8b. Implement operation selections.** Add one shared selection resolver
    and integrate a bounded first batch: background fitting and regional reductions.
    Reconcile existing fitting `bounds` and forwarded backend keyword arguments
    before choosing the public selection keyword. Keep this usable with current
    storage so selection correctness can be checked independently of native exposure.
    Gate: selected-fit/full-data correction and recoverable exclusions pass without
    changing global selection state or overwriting stored measurements.
  - [x] **8c. Expose the DataFrame.** Switch `Data.data` to the actual native table,
    implementing the agreed labels, exclusions, schema and mutation contract.
    Migrate affected Data consumers in bounded batches; retain explicit NumPy
    interchange and deliberate masked-array export. Gate: named/positional access,
    duplicate headers, roles, structural edits, retained references and rollback pass;
    reading `.data` does not construct a NumPy export or a substitute DataFrame.
    - [x] **8c.1. Owner labels and lossless package boundary.** Store displayed
      headers on the authoritative owner frame, including duplicates; retain
      UUID-labelled version-1 packages and restore headers on import. Keep stable
      schema IDs, exclusions, dtype, fill and typed metadata through edits/copies.
      Pandas transaction drafts now use displayed headers. The public `Data.data`
      snapshot contract is unchanged pending consumer migration.
    - [x] **8c.2. Native owner validation and replacement.** Define and test the
      owner-level native getter, detectable schema-drift rejection, fixed-schema
      DataFrame replacement and retained-reference behaviour against 8a. Preserve
      transaction guards and rollback; do not claim detection of same-shaped
      permutations or concurrent writes through retained native references.
    - [x] **8c.3. Data consumers and public getter.** Move affected core/numerical
      consumers to explicit conversions in bounded sub-batches, then switch the
      public property once its dependent contracts pass. Keep masked exports,
      duplicate-name resolution and mutation/return conventions explicit.
      - [x] **8c.3a. Numerical extraction, search and equality.** Move `asarray`,
        `column`, `search_index`, `search`, `columns` and the equality operator off
        public `.data` reads. Retain explicit NumPy result contracts, verify a
        test-only native getter, and preserve masks through rectangular searches.
      - [x] **8c.3b. Rolling-window boundary.** Migrate rolling windows with
        explicit masks, wrapping/centre-exclusion geometry and iterator-lifetime
        rules before moving on to selection/splitting and remaining consumers.
  - [x] **8d. Expose image and stack storage.** Switch `.image` and `.stack` to the
    chosen native xarray objects. Define the relation to numerical `.imarray` and
    the representation of channel data, exclusions, valid extents and coordinates.
    Preserve frame/crop ownership, coordinate maps and Maximus/Attocube shared grids.
    Gate: regional calculations, native coordinate selection, stack/frame edits and
    lossless export pass on real calibrated, multichannel and ragged fixtures.
  - [x] **8e. Complete consumer and compatibility migration.** Audit remaining
    fitting, analysis, plotting, image filters, folders, loaders and examples.
    Give each selection-aware operation an explicit contract; adapt only consumers
    required for this interface switch. Document `.mask` compatibility, public
    attribute types, direct editing restrictions and selected/full-data result scope.
    Gate: no consumer relies accidentally on implicit masked-array behaviour;
    broad wrapper simplification and unrelated backend conveniences remain Stage 9.
  - [x] **8f. Native-interface exit gate.** Run the complete suite, supported/lower
    dependency CI, installed-package checks and documentation audit on the final
    interface. Repeat integrated benchmarks against the recorded legacy comparison
    and Stage 7 production baseline; measure native reads and selection allocations.
    Gate: no implicit NumPy export on native access, scientific/ownership contracts
    pass, performance meets the accepted limits or has an explicit reviewed decision,
    and the intentional API changes are documented. Do not weaken tests or silently
    accept regressions merely because pandas/xarray behaves differently.

### Stage 8 selection and masking contract

Stage 8 implements this contract through the documented supported operations. The
motivating workflow is to fit a background using part of a measurement, then evaluate and apply that model
over the complete measurement. Original values stay available throughout; clearing
a temporary selection must never require reconstructing or restoring values.

| Concept              | Meaning and proposed rule                                                     |
| -------------------- | ----------------------------------------------------------------------------- |
| Stored values        | Complete native measurements, including recoverable excluded values.          |
| Persistent exclusion | Boolean state: true means excluded, retaining the corresponding raw value.    |
| Operation selection  | Boolean region: true means included in this operation; default is all.        |
| Missing/invalid data | NaN, missing readings or invalid inputs handled by the operation's policy.    |
| Stack padding        | Outside the valid frame extent; never becomes a measurement by clearing flags. |

The effective input is the operation selection intersected with non-excluded,
valid observations inside the stored extent. Clearing an exclusion reveals the
retained measurement; it cannot recover a value originally missing or invalid.
Keep exclusions distinct from missing-value sentinels and statistical weights.
Do not replace stored excluded values with NaN or treat zero weights as a universal
substitute for selection. Temporary xarray `.where()` results may contain NaN and
change dtype; they are calculation inputs, not replacements for the raw store.

**First public selection form.** Stage 8a chooses the explicit per-operation Boolean
selector `where=region`. Start with
positional Boolean arrays and native labelled Boolean Series/DataArrays. Require
exact shape/identity or explicitly defined coordinate matching; reject ambiguous
duplicates, missing/extra labels and unintended alignment or broadcasting. Resolve
selection before calling numerical backends. A native predicate such as
`(frame['field'] < -5) | (frame['field'] > 5)` should be directly usable after 8c.
Pandas and xarray calls made directly by the user retain their native semantics:
they do not automatically apply Stoner exclusions or an active selection.

**Rows, cells and channels.** Preserve existing cell-level exclusions, rather than
reducing all masks permanently to whole-row flags. For a fit, combine exclusions
and required-input validity across the actual x, y and uncertainty columns used by
that fit. An excluded unrelated channel must not discard an otherwise usable row.
For multiple fitted y channels, document whether fits resolve inputs separately.
An image-region selection may intentionally apply to several channels/frames only
through a documented dimensional rule; do not infer that from incidental broadcasting.

**Input selection versus output scope.** Fit parameters come from selected inputs;
model evaluation on the complete x coordinates is a separate explicit step. A
background correction can therefore update all observations without refitting or
changing the fitting region. Transform methods must state whether they update only
selected positions or the complete result, and how omitted positions retain their
values. Preserve established in-place/chaining contracts unless a change is recorded.
Reject empty or insufficient fitting inputs with a clear error and no partial update.
Specify reduction results for empty regions instead of inheriting backend accidents.

**Regional image algorithms.** A mean, fit or histogram can select independent
observations. Convolution, smoothing, interpolation and morphology need separate
rules for neighbourhoods, boundaries and excluded pixels. In 8e, document support
per method; reject unsupported selection combinations rather than pretending that
dropping pixels or inserting NaN preserves the algorithm's geometry.

**Optional conveniences, after explicit selection works.** Named regions and a
scoped selection context may be added as bounded follow-ups. A context must restore
prior state on normal exit and exceptions, define nested intersection and precedence
with explicit arguments, and never change the native storage getter. Decide whether
saved selections are frozen observations or predicates re-evaluated on current data.
Structural edits must transform saved selections by stable identity or invalidate
them explicitly; matching array length alone does not make a stale region valid.
These conveniences are not required to close the initial Stage 8 interface gate.

### Stage 8 native-access decisions and acceptance examples

Stage 8a resolves the following design questions in
[STORAGE_INTERCHANGE.md](STORAGE_INTERCHANGE.md#stage-8a-decisions-2026-09-20-implementation-follows-in-8b-8e).
The list remains the implementation review checklist; it is not a list of open
design choices. Worked examples and failure contracts are recorded in
[STORAGE_CONTRACTS.md](STORAGE_CONTRACTS.md#stage-8a-selection-and-native-access-acceptance).

- **Native representation:** `Data.data` is an actual DataFrame. Choose DataArray
  versus Dataset for `.image` and `.stack`, including where exclusion variables and
  multichannel values live. Prefer a cohesive Dataset when several variables must
  transform together; demonstrate the single-channel ergonomic cost before deciding.
  Do not recreate custom numerical subclasses or return a disguised snapshot.
- **Labels and identity:** current Data storage uses opaque column IDs and a
  positional row index. Decide how native headers, duplicate labels, row labels,
  column roles and stable identity coexist so ordinary named indexing is useful.
  Cover rename, reorder, insertion, deletion, selection and duplicate-label sorting.
- **Mutation and alignment:** a raw DataFrame cannot notify Stoner of every edit to
  an external mask/schema. Define a supported direct-edit subset and the owner
  operation or explicit edit/replacement route for other changes. Same-shaped
  permutations can desynchronise side state without an obvious shape change;
  demonstrate how the chosen architecture prevents this, or document the unsupported
  operation and provide a safe alternative. Do not promise automatic interception
  or detection that standard pandas/xarray objects cannot provide.
- **Reference lifetime:** specify what `native = obj.data` and equivalent image/
  stack references see after owner edits or whole-object replacement. Settle
  copy-on-write differences across supported pandas versions, shared crop/frame
  writes, locking, clones and failed assignments. Direct native operations are not
  implicitly transactional; explicit edit contexts must retain their promised rollback.
- **Compatibility and conversions:** decide whether `.mask` remains an alias for
  persistent exclusions and keep its true-means-excluded polarity unambiguous.
  Preserve lossless package interchange, typed metadata, raw values and fill state.
  Retain `to_numpy()` when a numerical consumer requires it, without mandatory
  conversion where the consumer already accepts a DataFrame or xarray object.

Acceptance coverage must include:

1. Fit two disjoint background regions, evaluate on the full coordinate axis and
   subtract across the full measurement. The fitted subset agrees with an explicit
   native selection, including uncertainties and exclusion of invalid required inputs.
2. Exclude an extreme value, calculate a regional statistic, remove the exclusion
   and recover the exact raw value and original dtype. Genuine NaN remains missing.
3. Fit one channel when an unrelated channel contains excluded cells; preserve
   duplicate headers and selection identity through supported schema operations.
4. Perform a calibrated image-region reduction with a dead pixel and apply the
   resulting background to the complete image. Preserve physical coordinates,
   channel alignment and integer raw values until an explicit dtype-changing result.
5. Select across an unequal-sized stack without including padding; retain live
   frame/crop behaviour and reject incompatible or stale selectors before mutation.
6. Exercise direct value edits, explicit structural edits, whole-native replacement,
   invalid replacement, retained references and detached exports. Verify the stated
   ownership contract on both lower-bound and current pandas/xarray versions.
7. Check scientific fits/uncertainties and image geometry against existing real
   fixtures; measure native getter and selected-operation costs without assuming
   that Boolean indexing is zero-copy.

- [x] **9. Simplify and extend Stoner wrappers.** After the Stage 8 native interface
  and its masking/mutation contracts pass validation, audit existing methods for
  backend delegation and select new backend methods worth exposing. Deliver small,
  independently tested wrapper batches as below.
  Gate: a predictable documented API with equivalent retained scientific behaviour.
  Review labelled row access through the pandas index and `.loc` exposed in Stage 8;
  add Stoner conveniences only where useful. Define label versus position selection,
  duplicate-label behaviour, result type and mutation semantics consistently with
  that interface. Distinguish selecting rows by index label from looking up named
  columns within a row; do not recreate either through a custom array class.
  This is an API enhancement batch, not additional Stage 6 removal work.
  - [x] **9a. Inventory delegation boundaries and add row previews.** Add only
    `Data.head(n=5)` and `Data.tail(n=5)` using pandas positional row selection and
    the existing clone/owner selection path. Retain exclusions, raw values, schema,
    dtype/fill, typed metadata and concrete class; return independent objects.
  - [x] **9b. Review numeric row sorting.** Characterise the existing `Data.sort`
    permutation contract, then delegate its ordinary numeric-key path only where
    pandas reproduces duplicate-header resolution, ties, NaNs, reverse ordering
    and exclusion movement. Retain custom-comparator behaviour and numerical paths
    whose semantics do not match. Document the decision and validate independently.
  - [x] **9c. Review distinct-column results.** Characterise `Data.unique`, including
    exclusions, sorted order, NaNs/complex values and optional index/inverse outputs.
    Compare pandas distinct-value operations before choosing any delegation; retain
    numerical paths where order, mask or return contracts differ. Do not extend
    tolerance-based duplicate merging as part of this batch.
  - [x] **9d. Review preview-copy allocations.** Profile the existing `head`/`tail`
    clone-before-trimming path and consider a shared independent row-subset result
    boundary. Copy only selected observations where this preserves concrete class,
    ancillary attributes, typed metadata aliases/cycles, raw values/exclusions,
    schema IDs/roles and index semantics. Measure against the 9a implementation;
    keep the existing path if an equivalent bounded adapter is not demonstrated.
  - [x] **9e. Cumulative wrapper exit review.** Review the complete 9a-9d diff and
    confirm the documented delegation/retention decisions. Validate the final
    revision with supported/lower dependency CI, installed distributions, cached
    documentation audit and integrated benchmarks before recording Stage 9's exit
    decision. Keep publication separate; propose further wrapper families as
    independent work rather than automatically expanding the backend API.

### Follow-on wrapper policy (batch 9)

For each wrapper, record the backend operation, input resolution, mutation default,
copy semantics, alignment, mask/schema/metadata handling and return type. Preserve
method-specific Stoner expectations; do not assume all methods mutate or return
`self`. For new methods, choose explicit defaults consistent with comparable
Stoner methods and document differences from pandas/xarray.

Delegate only where semantics match or a small explicit adapter makes them match.
Keep domain-specific numerical implementations when backend behaviour differs.
Use shared helpers for selection, state propagation and result wrapping. Test
wrapper behaviour at these boundaries rather than duplicating backend tests.
Check name collisions with existing and dynamically attached methods. Avoid
automatic exposure of the entire backend API and accidental changes after backend
upgrades. Lazy execution and broader coordinate-aware APIs remain separate later
proposals, not implicit scope for this batch.

### Stage 9a delegation inventory and wrapper contracts (2026-10-17)

This inventory identifies bounded candidates in the current core methods and
owner operations; it does not authorise blanket forwarding or declare Stage 9 done.

| Family | Backend candidate | Decision |
| --- | --- | --- |
| First/last rows | `DataFrame.head` / `tail` | Add explicit `Data.head` / `tail` wrappers in 9a; neither name collides with current core, dynamic analysis/fitting/plotting or inherited methods. |
| Existing row sorting | `DataFrame.sort_values` | 9b delegates compatible real keys after positional resolution; complex/other float widths/non-native byte orders and comparators retain numerical paths. |
| Distinct column values | pandas `unique` / `factorize` / `drop_duplicates` | 9c retains the existing NumPy wrapper: sorted masked results, complex/NaN groups, flattened first indices and shaped inverse outputs do not match raw pandas contracts. |
| Predicate selection | pandas indexing / `query` | Retain current `select` contracts: column-query keyword namespace, combined predicates and numerical result wrapping differ. |
| Duplicate reduction | `DataFrame.drop_duplicates` | Retain current tolerance/weighted-merge behaviour; exact equality is not an equivalent replacement. |
| Rename/reorder/insert/delete | pandas structural methods | Retain owner operations so exclusions, IDs and roles move atomically with values. |
| Scientific reductions, fitting and image operations | pandas/xarray calculations | Retain explicit selection, exclusions, validity and geometry adapters; raw backend calls alone do not preserve these contracts. |
| Labelled row conveniences | pandas index / `.loc` | Stage 8 exposes native access with a positional RangeIndex; retain that policy. General labels and duplicate-row-label APIs require a separately defined contract. |

For both new wrappers:

- **Backend and input resolution:** call `DataFrame.head(n)` or `tail(n)` on the
  validated native frame and obtain its row positions. Accept integer counts
  (including NumPy integers), reject Boolean/non-integer counts, and offer no
  arbitrary forwarded keywords. Five is the default; zero selects no rows;
  positive oversize counts select all rows; negative counts omit the opposite end.
- **Mutation/copy and return:** never mutate the source; return an independent
  object of the same concrete class, preserving ancillary instance attributes.
  Stage 9a used `clone` plus owner `take(rows=...)`; 9d optimises standard copying
  through the independent row-copy boundary below. Custom clone/deepcopy hooks
  retain the original path. These remain detached previews, not lazy views.
- **Alignment:** positional rows only, in original order; no label alignment,
  broadcasting or column-name lookup. Reset the result to a positional RangeIndex.
- **Exclusions/schema/metadata:** do not drop excluded rows or materialise NaNs in
  place of recoverable measurements. Apply the same positions to raw values and
  exclusions, retain dtype/fill and complete column schema (including duplicate
  headers, roles and IDs), and deep-copy typed metadata with cycles/aliases intact.
- **Boundaries:** reject detectable native schema drift and active editing
  contexts before creating a preview. `where` and `inplace` are unsupported and
  rejected by the explicit signatures. No new coordinate or labelled-row API.

### Stage 9b sorting contract and delegation decision (2026-10-17)

- **Backend/input:** Stoner resolves column positions, exact names (first duplicate
  occurrence), compiled patterns (all matches), repeated keys and the existing
  no-key x/y/z-role default. A temporary key frame has unique positional column
  labels; `DataFrame.sort_values(kind="stable", na_position="last")` determines
  the row permutation for native-byte-order Boolean/integer and float32/float64
  values. It reads raw keys, without applying exclusions or replacing them by NaN.
- **Retained numerical paths:** use existing `np.lexsort` for complex values,
  float16/other float widths and non-native byte orders. In both pandas 2.2.3 and
  3.0.5, exploratory comparisons found complex-NaN permutation differences and
  unsupported multi-key float16 factorisation. Do not cast keys merely to make
  pandas accept them. The existing two-row comparator still receives raw numerical
  rows through `cmp_to_key`.
- **Ordering/alignment:** first key is primary; ascending ties preserve input
  order, with real NaNs last. Reverse the complete ascending permutation for
  `reverse=True`, including ties and NaNs, rather than use pandas descending order.
  No label alignment, implicit broadcasting or row filtering is introduced.
- **Mutation/copy/return:** commit through existing owner `take(rows=...)` and
  return the same Data object. Move raw values and exclusions together; preserve
  dtype/fill, complete column schema/IDs/roles and authoritative typed metadata
  identity. Reset the positional row index. Owner commits retire native frame
  references, which callers must reacquire. This is an in-place method; callers
  explicitly clone when an independent sorted result is wanted.
- **Failure boundaries:** validate native state and reject active editing before
  backend/comparator work. Selector/comparator failures do not commit a permutation.
  The public signature remains `sort(*order, reverse=False)`; no `where`, `inplace`
  or arbitrary backend keywords are added. Only the ascending real-key permutation
  calculation is delegated; owner selection still performs its numerical export.

### Stage 9c distinct-value contract and review decision (2026-10-17)

- **Backend/input:** retain `np.unique(datafile.column(col), return_index,
  return_inverse)`. This is already a thin numerical wrapper with an explicit
  masked column snapshot. Resolve positions, names, patterns, slices and sequences
  through existing Stoner column selection. Exact duplicate names select the first
  occurrence; compiled patterns can select multiple columns.
- **Ordering/alignment/results:** deduplicate individual selected elements in
  row-major flattened order, not whole rows. Ordinary finite values are sorted.
  Return a one-dimensional MaskedArray; enabled options append ordinary integer
  arrays in `(values, first_indices, inverse)` order. First indices address the
  flattened selection; inverse indices retain the selection's shape, including
  `(rows, 1)` for a one-column list selector. No native label alignment is added.
- **Exclusions/missingness:** ordinary excluded finite observations share a masked
  group, retaining raw measurements only in the owner. Do not expose masked
  payloads as categories or drop exclusions. NaNs, complex-NaN equivalence and mixed
  masks/NaNs follow the existing NumPy masked-array behaviour; mixed NaN/mask cases
  can yield multiple masked entries. This review does not redefine or normalise
  missingness, nor alter tolerance-based duplicate finding or weighted merging.
- **Mutation/copy/metadata:** no mutation and no Stoner result wrapper. Values,
  masks, dtype/fill and optional index arrays are detached, writable numerical
  outputs; they do not carry Stoner schema/metadata. Owner values, exclusions,
  schema IDs and authoritative typed metadata remain unchanged. Numerical reads
  during an editing context see committed state until the draft is published.
  Detectable native drift still fails through the numerical column boundary.
- **Delegation decision:** pandas `unique` preserves encounter order and reads raw
  excluded values as categories; `factorize` has a different missing-value/index
  convention and no NumPy-shaped inverse contract; `drop_duplicates` concerns
  labelled rows rather than this flattened masked result. Preserving all of these
  contracts would require a new masking/order/remapping adapter where the existing
  numerical implementation already fits. Retain it for every dtype and flag
  combination; do not add a data-dependent pandas fast path or blanket forwarding.
  The signature remains `unique(col, return_index=False, return_inverse=False)`.

### Stage 9d preview-copy boundary (2026-10-17)

- **Selection/backend:** keep the existing pandas `head`/`tail` positional count
  rules. Their canonical RangeIndex defines a slice; do not allocate a full-table
  NumPy export or positional index array to trim the result. The shared private
  `_copy_rows` adapter lives in `Stoner/core/storage_bridge.py` and serves only
  previews in this batch, not predicate selection or unrelated structural methods.
- **Independent numerical state:** copy the selected frame and exclusions, reset
  the result RangeIndex and seed the standard deepcopy memo with those two leaves.
  `_TableState.__deepcopy__` honours that memo for its frame/mask instead of making
  unconditionally separate copies. Dtype/fill, headers, stable IDs and roles remain
  under the existing owner/schema validation; no NumPy storage is introduced.
- **Object graph:** use the existing Data deepcopy implementation so it registers
  the new instance, owner and state before copying typed metadata. Preserve concrete
  class, ancillary instance attributes, self-references and shared metadata/attribute
  aliases. Full raw frame/mask snapshots stored in metadata or ancillary fields
  retain their contents, independently of the selected owner leaves. The owner
  consumes its numerical memo entries before copying that ancillary graph, matching
  the original clone-then-trim policy. User arrays/views and metadata still follow
  normal deepcopy semantics and can dominate cost; they are not silently cropped.
  Result frames retain the previous canonical structural-copy policy: flat schema
  headers with no axis names, empty native DataFrame attrs and default duplicate-label
  flags. These backend details are not imported into the result as Stoner metadata.
- **Custom hooks and failure boundaries:** subclasses or instances with custom
  deepcopy hooks, and subclasses overriding `clone`, keep the existing full-clone
  plus owner-take path. Do not bypass their transformations. Active edits and native
  drift fail before copying; copy failures leave the source and retained references
  unchanged. No return, mutation, selection or direct-native-write policy changes.
- **Performance scope:** measure complete previews, including metadata cycles,
  against the captured 9a wrapper using the same current copy foundation. Retain
  the independently captured pre-change baseline as context. This optimises the
  authoritative numerical store; it does not promise zero-copy results or bounded
  cost for arbitrary user metadata or custom copying hooks.

## Session workflow

1. Read this plan, its handover and applicable repository guidance. Verify branch,
   remote and working-tree changes; preserve unrelated work.
2. Select the next incomplete batch or its recorded sub-batch. Read only the
   relevant implementation/tests; do not repeat the initial whole-package survey.
3. Explain the concrete scope, risks and validation for that session. Implement only
   that cohesive slice when implementation is requested; planning tasks do not
   authorise beginning production migration.
4. Run the relevant checks from a verified supported environment. Update examples
   and docstrings when behaviour changes. Keep reusable evidence with tests/tools;
   avoid accumulating completed session reports in the live tree.
5. Update checkboxes and the handover with changed paths, exact useful validation
   commands/results, decisions, limitations and the next bounded step. Record
   completed history in commits/PRs rather than growing this handover indefinitely.

Suggested resumption request:

> Read STORAGE_MIGRATION_PLAN.md and implement the next bounded batch on devel,
> starting from its handover. Update the plan with validation and the next step.

## Source map and upstream references

Initial inspection was on `devel` at `812690bf0`; these are starting points, not
claims that a future checkout has unchanged implementations.

- [Column resolver and roles](Stoner/core/setas.py),
  [DataArray slicing/descriptors](Stoner/core/array.py),
  [Data properties](Stoner/core/property.py),
  [indexing and iteration](Stoner/core/interfaces.py),
  [operators](Stoner/core/operators.py).
- [Pandas import](Stoner/core/data.py),
  [to_pandas and core methods](Stoner/core/methods.py),
  [typed metadata and existing pandas accessor](Stoner/core/base.py).
- [ImageFile/ImageArray](Stoner/Image/core.py),
  [mask and drawing proxies](Stoner/Image/attrs.py),
  [stack storage](Stoner/Image/stack.py), [Kerr classes](Stoner/Image/kerr.py).
- [Core tests](tests/Stoner/core), [image tests](tests/Stoner/Image),
  [analysis](Stoner/analysis), [formats](Stoner/formats),
  [public examples](doc/samples).
- [Pandas extension guidance](https://pandas.pydata.org/docs/development/extending.html):
  composition/accessors and extension dtype alternatives.
- [Pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html):
  pandas 3 mutation and array-export constraints.
- [DataFrame attrs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html):
  experimental status and limited propagation guarantees.
- [Xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html),
  [data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html)
  and [computation](https://docs.xarray.dev/en/stable/user-guide/computation.html):
  composition, metadata and automatic coordinate alignment.

Upstream references informed the initial investigation; recheck version-dependent
behaviour during the prototypes rather than assuming the latest documentation
matches the selected environment.

## Current handover

- **Stage 6 foundation:** `Stoner/Image/storage.py` implements ImageStorage version
  1 for eager image and stack Datasets, plus typed Frame records. Imports validate
  dimensions, real numeric/Boolean intensities, coordinate state, canonical frame
  IDs and record order, integer extents, compatible fills and excluded-zero padding.
  Native xarray exports warn about omitted typed metadata; package copies preserve
  nested metadata and hints. Ordinary arrays and uncalibrated image packages can be
  packed into ragged stacks with fresh IDs. Mixed integer promotion must be lossless.
- **Internal owner:** `Stoner/Image/storage_owner.py` owns a detached package and
  provides guarded metadata/mask interfaces, direct numerical writes and atomic
  NumPy/xarray editing. Drafts cannot change shape, dtype, coordinates or extents,
  nor write/unmask padding. Retained drafts and snapshots cannot alter the owner.
  Failed edits release the lock. Nested metadata values are not overwritten by
  numerical commits. No automatic backend alignment is used for pixel writes.
- **Shared-region prerequisite (6a.2a):** `ImageOwner.region(rows, columns)` uses
  non-empty unit-step positional slices and supports nested regions. Handles read
  current parent state and commit only their rectangle; metadata and transaction
  locks are shared with the parent. Exports and `clone()` detach, retaining sliced
  coordinates, coordinate attributes, raw masked values, fill and typed metadata.
  Successful parent package replacement invalidates saved regions and proxies
  with ReferenceError, even for the same shape. Failed replacement leaves them
  live. Shape/dtype/calibration changes through a region are rejected. Stack
  regions require the future stable frame-handle boundary first.
- **Public integration (6a.2b):** `Stoner/Image/storage_bridge.py` connects populated
  standalone ImageFile construction/loading, the ImageArray descriptor and dynamic
  method/property adapters to one xarray owner. No persistent `_image` or parallel
  `_metadata` remains. Public image/data/mask array reads detach; direct image/mask
  indexing and transactional drawing commit through the owner. NumPy conversion
  returns detached raw values and does not expose pointers into temporary arrays.
  Shape/dtype/size reads do not export pixels. Interchange/editing factories are
  available on ImageFile; an empty no-argument loader placeholder acquires an owner
  when populated. Explicit zero-sized image inputs are rejected.
  Crops support shared region handles and detached clones; rectangular slices
  preserve coordinates. Fixed-shape numerical commits keep regions live, whereas
  whole-parent replacement invalidates them. Use `im += other` or whole-property
  replacement instead of writing through `im.image`; edit metadata on the image,
  not its snapshot. Developer documentation and intentional-change tests cover this.
- **Compatibility consumers:** legacy stack extraction explicitly retains stack
  ownership, excludes stale `_fromstack` attributes from saved public attributes,
  and stores independent typed metadata dictionaries when importing owned images.
  This prevents metadata proxies from pointing at an unrelated standalone owner.
  Attocube TIFF metadata import uses guarded `import_all`; its background subtraction
  explicitly replaces the calculated values instead of editing an exported view.
  Interactive mask/crop selectors attach to the owner, scripted events resolve
  that owner, and imshow retains its actual figure/axes references. This preserves
  the runnable ImageFolder mask-selection example across the snapshot boundary.
  Numerical working arrays preserve those display-handle references while copying
  the remaining metadata, so reading an image does not open cloned pyplot windows.
- **Stage 6 boundaries:** ImageStack, Kerr, MaskStack, Attocube and Maximus use
  the new owners. The temporary LegacyImageStack class has been removed.
  Specialised native imports require their own storage adapter. Calibrated image
  packages and stack physical coordinates now pack and extract through the mapper.
  Non-separable image coordinates now use explicit y/x maps and pack into padded
  frame/y/x maps. Cropped pixel labels use separate index_y/index_x coordinates.
  Exact transpose, flip and quarter-turn coordinate mappings are implemented.
  Interpolating calibrated geometry and coordinate/operand compatibility now use
  the Stage 6c.3 adapters. Named-channel scan exports are detached analysis views;
  canonical owner storage remains frame/y/x.
- **Stage 6 focused validation:** the image suite passed **121 tests** in
  `maintenance/runs/20260913-154242-focused-4606f659`; its additional Attocube case
  exposed the exported-view subtraction fixed above. The final Attocube/public
  owner regression gate passed **11 tests, one warning**, in 55.77 seconds
  (`maintenance/runs/20260913-154535-focused-6d8f7953`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths tests/Stoner/test_FileFormats.py::test_attocube_scan,tests/Stoner/Image/test_storage_bridge.py`.
  Coverage includes real loaders/savers, masks/drawing, stack write-through,
  detached reads/clones/interchange, calibration slicing, shared crops, rollback
  and competing-write guards. The scripted mask-selection example plus public-owner
  regressions passed **13 tests, three warnings**, in 15.94 seconds
  (`maintenance/runs/20260913-160457-focused-d98c91e4`); reproduce with
  `./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths tests/Stoner/test_doc_samples.py,tests/Stoner/Image/test_storage_bridge.py,-k,'ImageFolder_mask_select or storage_bridge'`.
  The final loader-handoff refinement preserves filename/public attributes from
  the prepared input array rather than a snapshot of the placeholder. Its complete
  TIFF round-trip and public-owner gate passed **13 tests, three warnings**, in
  7.48 seconds (`maintenance/runs/20260913-161610-focused-60f661ef`).
- **Stage 6a full validation:** **681 passed, 708 warnings**, in 352.96 seconds,
  using two pytest workers and coverage (**77%** overall), in
  `maintenance/runs/20260913-161258-parallel-31f73f86`. Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  The full gate covers Data/analysis/fitting, folders, real formats, image/stack/Kerr
  compatibility and runnable examples, including scripted selection. The small
  loader filename refinement was made while this run was active and independently
  verified by the final 13-test gate above. Warnings include existing SciPy/Pillow
  deprecations, headless plotting, deliberate lossy exports and numerical-domain
  warnings; no blanket suppression was added. Whitespace/repository-content checks
  pass and all 243 cached plots remain. This completes Stage 6a, not all of Stage 6.
- **Dependency boundary:** xarray is now a required runtime import and is declared
  consistently in pyproject.toml, requirements.txt, recipe/meta.yaml, the test and
  documentation Conda environments and doc/requirements.txt. The build-tools-only
  environment needs no runtime dependency. The active environment supplies xarray
  2026.7.0 from conda-forge. Setuptools discovery and the fresh source manifest
  check passed (`maintenance/runs/package-contents.json`), including runtime assets
  and scientific fixtures; no distribution was built or installed. Python support
  and lower-bound/platform validation remain Stage 7.
- **Completed:** batches 1-5. `Data(...)`, `Data.load`,
  `from_storage` and `from_pandas` use a single authoritative DataFrame, Boolean
  exclusion mask, stable column schema and typed metadata. No persistent `_data`
  or parallel metadata store exists. Construction and loading now use the frame
  owner from the outset; `_constructing` and the legacy constructor/load wrappers
  have been removed.
- **Consumer integration:** numerical reads use selected columns or explicit
  masked arrays where needed; shape/dtype checks do not export data. NumPy's
  acceptance of a DataFrame does not transport the separate exclusion mask.
  Avoid repeated whole-table conversions inside loops. Outlier callbacks edit a
  transactional draft. Fitting preserves source values, masks and IDs, handles
  result column zero, and avoids evaluating a curve when no result is requested.
  Binning returns a centre plus signal/error/count triple per y column, preserving
  full headers. Deduplication groups complete keys and returns the reduced values.
- **Formats and folders:** loaders keep their registry identities and priorities.
  TDI default headers, Lake Shore unit-bearing headers, EasyPlot schema expansion,
  PinkLib headers and CSV metadata updates work through the owner. Constructor
  header padding/truncation retains legacy behaviour; subsequent header edits are
  fixed-width. Folder `output="data"` requires numeric metadata; text/object values
  use `output="frame"` or `output="array"`. Automatic non-numeric metadata selection
  returns a DataFrame. Numeric mixed columns are explicitly normalised at this
  boundary. Image alignment and Attocube consumers use these metadata outputs
  without changing image storage.
- **Retained result interface:** `_storage_array`/`_storage_slice` supply detached,
  read-only DataArray results for public role and row-index access, with no implicit
  write-back. Analysis consumers no longer infer their backend from `.data`.
  These result wrappers are compatibility interfaces, not an alternative storage
  backend. Standalone DataArray remains under the announced compatibility policy.
- **Intentional changes:** writable array views become snapshots; masked scalar
  indexing returns `np.ma.masked`; normal negative bounds apply; numeric strings
  are names/patterns; compiled patterns retain distinct duplicate positions;
  second-x-group error positions are absolute. Empty Data is `(0, 0)`.
  Reordering always moves headers/roles. Public examples use owner assignments
  and exact-width header edits. General Boolean column selectors remain invalid;
  analysis `result=True` is explicitly translated to append at its API boundary.
- **Stages 1-5 validation:** **619 passed, 705 warnings**, in 262.39 seconds using
  two pytest workers and coverage on the final implementation
  (`maintenance/runs/20260913-143927-parallel-8c85a410`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  This includes primitive contracts, analysis/fitting, plotting, real loader/saver
  fixtures, folders, image consumers, runnable examples and maintenance checks.
  Warnings include SciPy ODR deprecation, headless plotting, deliberate lossy
  exports and numerical model-domain warnings; no blanket suppression was added.
  Whitespace and repository-content checks passed; all 243 cached plots remain.
  New regressions cover fit masks/IDs, callback rollback, multi-key deduplication,
  tolerant duplicate search without source mutation, binned counts/headers,
  rectangular indexed reads/writes and LaTeX dictionary headers. The focused
  regression/example gate passed **10 tests**, five warnings, in 7.86 seconds
  (`maintenance/runs/20260913-143840-focused-649b86ee`). The bin-count fixture uses
  positive edges to avoid changing the pre-existing non-positive edge policy.
- **Dependencies/platforms:** Miniforge py314, Python 3.14.7, NumPy 2.5.2, pandas
  3.0.5 and SciPy 1.18.0. Existing specifications and CI are unchanged. Apply the
  agreed Python 3.12 metadata/CI transition together in Stage 7. Hosted CI, other
  Python/platform versions and integrated performance remain unverified.
- **Historical performance:** the accepted Stage 3 JSON remains
  `maintenance/runs/storage-prototype-final.json`, with 19 passing paired
  workloads and matching source hashes. Its `legacy` branch now imports migrated
  Data if rerun here. Use a separate pre-migration checkout for future comparisons.
- **Stage 6b.1:** `Stoner/Image/stack_owner.py` provides an internal StackOwner
  and FrameOwner. Integer positions resolve to stable IDs; duplicate display names
  remain distinct. Saved frame, mask and metadata handles survive insertion and
  reordering; deleted IDs cannot be reused. Whole-stack replacement preserves live
  frame IDs but invalidates saved crop bounds. Numerical transactions share the
  stack lock, detach retained drafts and preserve per-frame metadata/fill and raw
  masked values. Deletion trims padding, including deletion of the last frame.
  `StackOwner.image(selector)` binds an ImageFile to a frame, including shared
  crops, drawing, mask writes and typed metadata replacement/import. Existing
  calibrated packages support frame extraction and reordering; insertion rejects
  coordinates or native attributes requiring the pending packing mapper. Public
  ImageStack still uses its legacy backend; this completes the bounded foundation
  batch, not all of Stage 6b.
- **Stage 6b.1 validation:** **692 passed, 708 warnings**, in 348.69 seconds,
  with two workers and 78% coverage on Miniforge py314
  (`maintenance/runs/20260913-163048-parallel-b5bf13c4`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  This includes 11 new frame-owner regressions, real formats, scientific examples
  and maintenance checks. The earlier focused owner/region/bridge gate passed
  72 tests in 8.17 seconds (`maintenance/runs/20260913-162716-focused-200c2882`);
  the final full run additionally covers rejection of native attribute loss.
  Whitespace and new-module syntax checks pass; all 243 cached plots remain.
  Hosted CI, other platforms and integrated stack performance remain unverified.
- **Stage 6b.2a:** `Stoner/Image/storage_coordinates.py` maps image physical axes
  into `physical_y`/`physical_x`, with NaN tails and positional stack dimensions.
  Scalar coordinates (including acquisition times) become frame coordinates and
  extract back to scalars. Dataset, intensity, exclusion and coordinate attrs must
  agree across inputs; units are not implicitly converted. Unsupported auxiliary
  spatial coordinates, reserved names, incomplete scalar coordinate sets and
  precision-losing integer axes raise before insertion publishes any state.
  Packing/extraction retain masks, raw values, typed metadata and per-frame fill.
  StackOwner insertion now uses this mapper and retains stack-only attributes,
  saved frame handles and crop bounds. Empty calibrated stacks retain attribute
  compatibility checks when repopulated. Differing scalar coordinate dtypes must
  be explicitly normalised before packing. Public ImageStack still uses its legacy
  backend; its switch remains the next bounded batch.
- **Stage 6b.2a validation:** **702 passed, 708 warnings**, in 358.13 seconds,
  with two workers and 78% coverage on Miniforge py314
  (`maintenance/runs/20260913-164206-parallel-a803b015`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Ten new calibration regressions cover round trips, detachment, crop/handle
  write-back, attribute conflicts, unsupported coordinates, integer precision and
  empty calibrated stack reinsertion. The earlier focused gate passed 54 tests,
  two warnings, in 8.47 seconds
  (`maintenance/runs/20260913-164106-focused-db8775ba`). Whitespace and syntax checks
  pass; all 243 cached plots remain. Hosted CI, other platforms and integrated
  stack performance remain Stage 7 validation boundaries.
- **Stage 6b.2b integration:** `Stoner/Image/stack_bridge.py` connects public
  ImageStack construction, folder insertion/replacement/deletion, sorting, slicing,
  clone, conversion and interchange to StackOwner. No persistent `_stack`, `_sizes`,
  `_names` or per-frame `_metadata` store exists on ordinary stacks. Public items
  are ImageFiles bound to stable FrameOwners; `_fromstack` is specialised-only.
  `imarray` and private compatibility `_stack` reads are detached read-only results.
  Direct frame/pixel/mask writes and guarded NumPy/xarray transactions remain live.
  Folder bulk methods operate on detached items before committing their results,
  allowing shape changes while retaining frame IDs; fixed-shape commits keep crop
  handles live. Public sort/reorder preserve saved handles. Reductions exclude user
  masks and padding; standard errors count included samples at each pixel.
  Calibrated reductions preserve matching coordinates and reject conflicting ones.
  Direct frame dtype/shape changes require a clone followed by parent assignment,
  or a whole-stack conversion. Non-broadcastable masks now raise instead of being
  silently resized. Kerr/MaskStack and Attocube/Maximus explicitly use the retained
  specialised backend, to be migrated together with their private consumers in 6c.
  ABC registration retains their public ImageStack family checks without creating
  a second storage owner. Cloning an editing stack copies committed state with a
  fresh, unlocked owner. Uncalibrated reductions also retain native variable attrs;
  structural array assignment rejects attributes it cannot preserve.
- **Stage 6b completion validation:** the full local suite passed **709 tests,
  708 warnings**, in 355.79 seconds, with two workers and 76% coverage
  (`maintenance/runs/20260913-165856-parallel-d7c48581`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Final lock/attribute refinements made while that run was active passed the
  focused public-stack/owner/calibration/conversion gate: **41 passed**, one
  warning, in 25.67 seconds (`maintenance/runs/20260913-170203-focused-0bbdbaf9`).
  The final specialised-family registration and public stack/Kerr gate passed
  **17 tests**, one warning, in 17.36 seconds
  (`maintenance/runs/20260913-170333-focused-d4a82194`). The full-run count predates
  the final two added regressions; it is not a claim of a later full rerun.
  Syntax and whitespace checks pass, and all 243 cached plots remain. Stage 6b
  is complete; specialised consumers/transforms and the integrated performance,
  platform and hosted-CI gates remain Stages 6c/6d/7. No commit or push requested.
- **Stage 6c.1:** `Stoner/Image/storage_transforms.py` performs exact coordinate
  permutations on detached packages. Transpose/swapaxes rename spatial dimensions
  and reorder intensity/exclusion axes; flips and CW/CCW reverse the appropriate
  coordinate sequences alongside pixels. Axis units travel with the source axis,
  scalar coordinates remain scalar and auxiliary spatial coordinates follow their
  dimension. Typed metadata, raw masked values, fill values and native attrs remain
  intact. `image.T` now returns an independent calibrated ImageFile through the
  same adapter. Clone-returning properties preserve source state; explicit in-place
  permutations invalidate crop bounds and reject shared-frame/region targets.
  Arbitrary-angle `rotate` retains its existing radians API and still rejects
  calibrated input pending an interpolation/transform representation. No physical
  units are inferred and existing crop/index arguments remain pixel positions.
- **Stage 6c.1 validation:** **719 passed, 708 warnings**, in 283.06 seconds,
  with two workers and 77% coverage on Miniforge py314
  (`maintenance/runs/20260913-171705-parallel-ee8bd243`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  The initial full run exposed a separate generated-property route for `T` and
  was stopped after that regression was isolated. The corrected property now
  delegates to `transpose(_=None)`. The corrected focused transform/bridge/core
  gate passed **50 tests**, five warnings, in 17.54 seconds
  (`maintenance/runs/20260913-171619-focused-a8b33055`) before the successful full
  rerun. Eight new regressions cover exact value/mask permutations, units and
  auxiliary coordinates, inverse transforms, argument rejection and handle
  lifetimes. Syntax/whitespace checks pass and all 243 cached plots remain.
  Hosted CI, other platforms and integrated performance remain Stage 7 boundaries.
- **Stage 6c.2:** all four specialised consumers now use the public stack owner.
  Kerr image snapshots retain KerrArray methods without a second pixel store;
  stack items retain KerrImageFile type. Cropping text publishes one package,
  field metadata follows reordering, and mask switching uses pixelwise Boolean
  operations without mutating the source. MaskStack preserves nonzero truth
  conversion and excluded-zero ragged padding. The temporary LegacyImageStack
  class and its ABC registration have been removed.
  ScanMetadataMixin stores shared headers in owner metadata, with stable frame
  proxies reading common defaults and writing local overrides. Attocube/Maximus
  loaders and HDF5 writers use public items and names. Repeated HDF5 saves update
  pixels, exclusions, fill values and effective metadata; older files without
  exclusion datasets remain readable. Arbitrary native coordinate serialisation
  is not added to those format-specific files.
  Following the maintainer's common-grid observation, explicit
  `to_xarray(format="channels")` exports provide named variables and exclusions.
  Attocube retains measured PosX/PosY as 2D coordinates. Maximus exposes its named
  detector, header axes and frame-axis values with recorded units and orientation;
  the existing single-detector/multi-region loader limitation is retained.
  These are detached analysis views: canonical storage and editing remain
  frame/y/x, and export_storage remains the typed, lossless interchange path.
- **Stage 6c.2 validation:** the full local suite passed **725 tests, 708 warnings**
  in 321.65 seconds with two workers and 77% coverage on Miniforge py314
  (`maintenance/runs/20260913-175750-parallel-f8dfde4a`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  This includes real Attocube and Maximus fixtures and the scientific examples.
  Final review then fixed Boolean inversion of ragged padding in denoise_thresh;
  inversion now visits only each frame's valid extent. The final specialised/Kerr
  gate passed **15 tests, one warning**, in 17.31 seconds
  (`maintenance/runs/20260913-180325-focused-94e4a03e`), including that added
  regression. The full-suite count predates this last fix and test. Reproduce the
  final gate with `./maintenance/run-baseline.ps1 -Check focused -Environment py314
  -TestPaths tests/Stoner/Image/test_specialised_storage.py,tests/Stoner/Image/test_kerr.py`.
  Syntax and whitespace checks pass; all 243 cached plots remain. Hosted CI,
  other platforms and integrated performance remain Stage 7 boundaries.
- **Stage 6c.3:** `Stoner/Image/storage_interpolation.py` carries calibration
  through rotation, rescale, resize, warp, translation, shift, zoom, affine and
  gridimage operations. Registration computes its shift once, propagates exclusions
  through that geometry and adopts the reference grid, preserving the existing
  numerical convention. A raw-array reference retains the source grid.
  Physical and auxiliary coordinate maps use linear interpolation with retained
  units; dimension axes become positional pixels. Outside-source coordinates are
  NaN and excluded regardless of intensity boundary mode. Exclusions expand
  conservatively for higher interpolation orders and resize anti-aliasing.
  Calibrated resize uses scikit-image rather than unsupported masked-array resize.
  Custom warp mappings must be deterministic; external output buffers are rejected.
  Typed metadata, native attrs and fill values are retained. Shared geometry
  requires a clone and in-place transforms invalidate earlier crop handles.
  ImageStorage accepts y/x maps, StackOwner packs them as frame/y/x with NaN
  padding, and extraction preserves cropped pixel labels via optional index axes.
  Arithmetic and numerical-method ImageFile operands with calibration require
  identical coordinates and units before conversion; mismatches fail atomically.
  Uncalibrated images, raw arrays and scalars retain positional semantics. No
  implicit xarray alignment, unit conversion or physical-unit arguments are added.
  Registration is explicit grid reconciliation and is exempt from the arithmetic
  compatibility check. Folder alignment clones cropped inputs, sends detached
  reference pixels/coordinates rather than an owned image through worker bookkeeping,
  and propagates a named original exception before aggregating translation metadata.
  The maintainer reports earlier intermittent translation_limits failures; the
  generic missing-metadata symptom alone does not identify their original cause.
- **Stage 6c.3 focused validation:** the transform/interpolation/packing gate passed
  **32 tests, one warning**, in 6.94 seconds
  (`maintenance/runs/20260913-181413-focused-70912aed`). The first full gate exposed
  two earlier contracts that intentionally rejected maps and calibrated rotation;
  it was stopped and those assertions were replaced by preservation/detachment
  checks. The corrected storage/bridge/interpolation gate passed **60 tests, two
  warnings**, in 8.22 seconds (`maintenance/runs/20260913-181746-focused-ee207ca4`).
  The next full run passed 739 tests but failed the STXM example with missing
  translation_limits. Folder alignment now clones region inputs and reports
  worker errors directly; reference-grid reconciliation also allows subsequent
  calibrated XMCD arithmetic. The corrected interpolation/stack-align/STXM gate
  passed **17 tests, seven warnings**, in 35.41 seconds
  (`maintenance/runs/20260913-182736-focused-5c8c69b3`). Three successive STXM runs
  in one process passed in 69.68 seconds, with 19 warnings
  (`maintenance/runs/20260913-182832-focused-6eb4eb47`), using the same test file
  three times with `--keep-duplicates -k STXMIMage_Demo`. This does not establish
  the original cause of the historically intermittent symptom. A regression
  asserts propagation of the original backend exception and its cause.
- **Historical Stage 6d image gate (superseded as the Stage 6 exit):**
  **741 passed, 708 warnings**, in 328.86 seconds,
  with two workers and 77% coverage on Miniforge py314
  (`maintenance/runs/20260913-183023-parallel-bee43c85`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  This final run includes all image/stack/format/scientific examples, the STXM
  example and the original-error propagation regression. Syntax and whitespace
  checks passed; all 243 cached plots remained. Legacy classes were still present,
  so this run does not satisfy the subsequent removal requirement. Hosted
  CI, other platforms, installed-package/documentation builds, dependency bounds
  and integrated performance remain Stage 7 validation, not claims of this gate.
- **Historical scope update after Stage 6d:** the maintainer identified the active
  `core.data` import of `core.array.DataArray` and required actual removal before
  a test release. Data snapshots, the old data descriptor, ImageArray/KerrArray
  adapters and ImageStackMixin still existed then. This reopened Stage 6 for 6e;
  the removal and validation below now satisfy that requirement.
- **Stage 6e implementation:** deleted `Stoner/core/array.py`, DataArray,
  ImageArray, KerrArray and the old ImageStackMixin implementation, together with
  their exports, legacy descriptors and fallback consumers. Ordinary helpers in
  `core/numerical.py` and `Image/numerical.py` return native NumPy masked arrays;
  there is no replacement ndarray subclass. Data schema and row identity remain
  on the pandas owner, and images/stacks use their xarray owners. Immediate Data
  numerical results can carry explicit descriptive annotations, but NumPy copies,
  slices and arithmetic do not propagate them or write through to the owner.
  Consumers now resolve row positions explicitly, including peak selections and
  row callbacks. Image operations preserve metadata through numerical work and
  commit through owners. Real TIFF, Kerr/OCR, mask, alignment, stack and scientific
  example coverage is retained. The historical TIFF `ImageArray.dtype` metadata
  key remains readable without importing or reconstructing the removed class.
- **Stage 6 removal audit:** `tests/Stoner/core/test_storage_removal.py` checks
  absent public exports and the deleted module, scans runtime syntax for removed
  class references and custom NumPy storage subclasses, and verifies native
  numerical result types, detached/read-only snapshots and absent legacy stores.
  Public guides, examples, STORAGE_CONTRACTS.md and STORAGE_INTERCHANGE.md reflect
  the removal. Undefined-name (`ruff --select F821`) and whitespace checks pass;
  all 243 retained plot-cache files remain present.
- **Stage 6 exit passed locally:** **743 passed, 709 warnings**, in **301.57 seconds**,
  with two workers and **80% coverage**, using Miniforge py314 (Python 3.14.7,
  NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0 and xarray 2026.7.0).
  Command: `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Evidence: `maintenance/runs/20260913-222027-parallel-3816bd0e` (`run.log` and
  `coverage.xml`). This full run follows actual class removal and all consumer
  fixes, with no added skips or expected failures. Audited state is branch `devel`,
  HEAD `870ff1ce8` plus the uncommitted migration changes. SHA-256 for the sorted
  250 Python files under `Stoner`, `tests` and `doc/samples` is
  `e703a0e9023b3f428c781a1427f2404fc6913f9c05053c520c7ce2da86881e07`:
  hash each relative POSIX path, NUL, file bytes with CRLF normalised to LF, NUL;
  exclude `__pycache__`. Plan/prose updates do not change this runtime/test state.
- **Historical Stage 6 documentation boundary (resolved in Stage 7):** public source documentation was updated,
  but the attempted cached Sphinx build could not write an autosummary source
  stub under the execution filesystem restrictions. No successful complete build
  is claimed. Repeat in a suitable isolated documentation checkout during Stage 7;
  generated API stubs and the retained plot cache were not refreshed here.
- **Lower-bound CI correction:** run `34784426200` failed during imports because
  `tests/minimum-env.yml` omitted the now-required xarray dependency while package
  installation deliberately used `--no-deps`. Added xarray to that environment
  and extended `maintenance/check-ci-config.py` to check that all declared runtime
  dependency names are represented in the lower-bound environment. This corrects
  environment completeness; it does not establish new tested dependency bounds.
  Following maintainer clarification, also applied the Python 3.12 minimum to
  package/Conda metadata, test environments, CI matrices and current documentation.
  The CI policy check passes and a simulated missing-xarray environment is rejected.
  Stage 7 subsequently verified the revised lower-bound environment; see below.
- **Stage 7 dependency and package evidence (2026-09-20):** commits `a60a167db`
  and `8addfd091` align Python 3.12 support and declare pandas >=2.2 / xarray
  >=2024.10 consistently. The final runtime commit `c36dac690` passed the
  [lower-bound suite](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/35525413747):
  **743 passed, one skipped, 6835 warnings**, 234.49 seconds, using Python 3.12.14,
  NumPy 2.0.2, pandas 2.2.3 and xarray 2024.10.0. The
  [package gate](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/35525413842)
  passed wheel/sdist archive checks and separate clean installs on Python 3.12
  and 3.14. These are hosted results, not inferred from the Windows environment.
- **Stage 7 hosted platform gate:** the complete
  [test workflow](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/35525413740)
  passed at `c36dac690`: Linux Python 3.12, 3.13 and 3.14, macOS Python 3.14,
  and both test-result publishing jobs. Combined with the lower-bound and package
  results above, this closes the supported-platform gate for the alpha.
- **Stage 7 Windows gate:** the final runtime/test state at `c36dac690` passed
  **744 tests, 709 warnings**, in **317.84 seconds**, with two workers and
  **80% coverage** on Miniforge py314. Command:
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Evidence: `maintenance/runs/20260920-182249-parallel-cb1c14e2`.
- **Stage 7 performance:** the first complete integrated run passed 14/19 pairs,
  exposing excessive copies in row appends, stack insertion and frame edits.
  `c36dac690` adopts newly allocated row buffers after validation, packs uniform
  uncalibrated stacks without unpacking every frame, avoids redundant image
  exports and integer precision conversions, and commits validated frame-sized
  buffers without copying the whole stack. Calibrated/ragged insertion retains
  the general mapper. A new regression protects input/draft detachment, masks,
  fill values and saved frame identities; the focused gate passed **58 tests**.
  The final benchmark `maintenance/runs/stage7-integrated-final.json` passed
  **19/19 pairs** with five repetitions and separate allocation measurements.
  Reproduce using `maintenance/benchmark-integrated-storage.py --legacy
  maintenance/runs/stage7-legacy --output maintenance/runs/stage7-integrated-final.json`
  in activated Miniforge py314. Legacy checkout: `870ff1ce8`; production:
  `c36dac690`. Both execute the public API in isolated processes; the runner
  verifies import paths and hashes all production Python sources. Production hash:
  `c40b7f9baea2bac09de3ae07e717587d8f76aba882e832c8bf962aad70a85fc8`.
  Large row appends measured 60.679 -> 61.440 ms, stack insertion 244.505 ->
  11.529 ms, and frame edits 0.423 -> 3.222 ms; all meet the accepted absolute/
  relative time and allocation limits. These are measured workloads, not universal
  speed guarantees or a waiver of the Stage 3 thresholds.
- **Stage 7 documentation:** the isolated cached HTML build and existing Windows
  warning audit passed with **88 reviewed warnings, zero unexpected warnings**,
  all five primary classes documented and no missing dynamic Data methods.
  Evidence: `maintenance/runs/stage7-doc-output/` and `stage7-doc-audit.log`.
  Removed internal owner imports from public autosummary tables and qualified
  ambiguous Kerr return types; warning ceilings were not increased. Browser checks
  over localhost covered the index, alpha upgrade notes, image guide and Data API
  navigation. All 243 retained plot-cache files are unchanged. The earlier hosted
  [documentation build](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/35524678798)
  also passed; local audit additionally validates the subsequent warning fixes.
- **Stage 7 completion:** all five sub-gates passed on 2026-09-20. The final runtime
  and tests are committed at `c36dac690`; subsequent changes are the documentation
  heading correction and this validation/maintenance record. Python 3.15 readiness
  remains explicitly pending its final release and dependency availability, rather
  than an unverified supported version. No package tag or publication was created.
- **Stage 8 planning update:** the maintainer identified temporary operation scope
  as the main masking workflow: fit a selected background, then apply it to the
  complete measurement. Stage 8 now separates raw values, persistent exclusions
  and per-operation selections, with explicit native-mutation and alignment rules.
  Batches 8a-8f and their acceptance cases are specified above. Named/scoped selection conveniences
  are optional follow-ups after the explicit per-operation interface works.
- **Stage 8a complete (2026-09-20):** audited shared fitting assembly, full-axis
  model evaluation, result masks, stack reductions and owner editing boundaries.
  Recorded normative decisions and worked acceptance cases: `where` selects inputs;
  legacy `bounds` remains a row selector; `curve_fit(parameter_bounds=...)` handles
  optimiser limits; fits resolve exclusions per required channel. The target native
  objects are a header-labelled DataFrame and cohesive xarray Datasets. Positional
  row indexes and existing channel/frame geometry remain; arbitrary row labels and
  channel-axis redesign are deferred. Fixed-schema native edits are supported under
  the documented lifetime rules; structural edits use owners or package import.
  Two regression cases protect disjoint background fitting/full-axis correction and
  exact excluded integer recovery. Focused consumer tests: **11 passed, 1 warning
  in 5.94 s**, Miniforge `py314`; the warning is existing SciPy ODR deprecation.
  Evidence: `maintenance/runs/20260920-185014-focused-7f823ce2/pytest.xml`.
  `git diff --check` passed. This is design/characterisation validation, not proof
  of future `where` or native getters; no production behaviour changed in 8a.
- **Stage 8b complete (2026-09-20):** `Stoner/core/selection.py` validates strict
  positional/labelled Boolean selectors. `curve_fit` resolves independent eligible
  inputs for each y channel, intersects legacy row `bounds`, accepts optimiser
  `parameter_bounds`, checks uncertainties and sample counts, and stages all result
  columns/metadata before publication. Model masks now depend on model inputs;
  residual masks additionally depend on measured y. Fit statistics use the selected
  observations and correctly weighted residuals. Full-axis model evaluation remains
  explicit unless result columns are requested, preserving restricted-domain fits.
  `ImageFile.mean` and `ImageStack.average` compose selection, exclusions, finite
  values and valid extents; empty outputs are NaN (excluded for stack pixels).
  Added fitting/image selection regressions and user-guide examples. Final focused
  validation: **42 passed, 1 warning in 13.50 s** at
  `maintenance/runs/20260920-190227-focused-11842fe6`. Full parallel suite:
  **762 passed, 708 warnings in 295.90 s; 80% coverage** at
  `maintenance/runs/20260920-185920-parallel-62a76d9a`. The full run began before
  the last two regression additions and restricted-domain evaluation refinement;
  the final focused run verifies those changes. Both used Miniforge `py314` on
  Windows. These results do not establish lower-bound/hosted CI or performance
  validation for Stage 8; those remain the 8f exit gate. `git diff --check` passed.
  New test paths retain canonical `tests/Stoner/Image` and `tests/Stoner/analysis`
  spelling in Git. Public storage getter types remain unchanged; other fitting
  engines and neighbourhood operations remain for the 8e consumer audit.
- **Stage 8c.1 implementation (2026-10-16):** `Stoner/core/storage_owner.py`
  separates private header-labelled `_TableState` from UUID-labelled version-1
  `DataStorage` interchange. There remains one authoritative DataFrame; transient
  package views relabel columns without a NumPy export, and public package exports
  deep-copy values and side state. Construction, numerical commits, replacement,
  selection, insertion and copies restore displayed headers, including duplicates.
  Header edits validate every schema record before updating frame labels; roles
  and stable IDs remain separate. `edit_pandas()` drafts now support named raw
  value edits, retaining masks/schema and the existing fixed-axis/dtype checks.
  `_TableState.__deepcopy__` preserves explicit metadata types, nested aliases and
  cycles rather than reparsing strings. The first focused run caught that copying
  regression; it was corrected before the passing rerun. New regressions are in
  `tests/Stoner/core/test_storage_native_labels.py`; existing copy/TDI regressions
  remain intact. Ownership/contracts, developer guidance and maintenance follow-ups
  reflect this boundary. Public `Data.data` remains the read-only numerical snapshot;
  no native getter or automatic schema-drift detection is claimed by this batch.
  The starting checkout is `devel` at `721ee3db7`, with the existing uncommitted
  8a/8b changes preserved. No commit, push, tag or publication is part of this batch.
- **Stage 8c.1 validation:** the final focused gate passed **272 tests, five
  warnings in 30.75 s** (`maintenance/runs/20261016-222235-focused-0732bdc0`).
  Command: `./maintenance/run-baseline.ps1 -Check focused -Environment py314
  -TestPaths tests/Stoner/core,tests/Stoner/analysis/test_operation_selection.py,tests/Stoner/analysis/test_storage_consumers.py`.
  The subsequent full suite passed **776 tests, 708 warnings in 334.53 s**, with
  two workers and **80% coverage**
  (`maintenance/runs/20261016-222344-parallel-b19a098b`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Both runs use Miniforge py314: Python 3.14.7, NumPy 2.5.2, pandas 3.0.5,
  SciPy 1.18.0, xarray 2026.7.0 and pytest 9.1.1. Undefined-name checks pass for
  changed Python files (`python -m ruff check --select F821 ...`); CRLF-aware
  whitespace checks pass (`git -c core.whitespace=cr-at-eol diff --check`), without
  normalising the checkout. All 243 tracked plot-cache files remain unchanged.
  Conda and runtime report uncertainties 3.2.2, but its installed Python distribution
  metadata reports 0.0.0, causing `pip check` to flag lmfit's requirement; no clean
  pip-metadata check is claimed and the environment was not modified by this batch.
  The maintainer subsequently supplied xarray in py312; its verification and focused
  results are recorded below. Lower-bound dependency and hosted CI, installed-package
  checks and repeated integrated benchmarks remain Stage 8f gates, not claims of
  these Windows runs.
- **Stage 8c.1 Python 3.12 validation:** after the maintainer supplied xarray,
  Miniforge py312 was verified as Python 3.12.14, NumPy 2.5.3, pandas 3.0.5,
  SciPy 1.18.1, xarray 2026.7.0 and pytest 9.1.1. The same focused command above,
  substituting `-Environment py312`, passed **272 tests, five warnings in 47.62 s**
  (`maintenance/runs/20261016-223251-focused-0d8931fd`). This validates the focused
  owner/consumer boundary on Python 3.12 with current dependencies; it is not a
  pandas 2.2/xarray 2024.10 lower-bound run or a full Python 3.12 suite.
- **Stage 8c.1 documentation:** cached Sphinx HTML and the reviewed Windows
  warning/API audit pass: **88 reviewed warnings, zero unexpected warnings**,
  all five primary classes documented, no missing dynamic Data methods and no
  removed API inventory objects relative to `maintenance/runs/stage7-doc-output/html`.
  Final output: `maintenance/runs/stage8c1-doc-clean-output`; warning log and audit:
  `maintenance/runs/stage8c1-doc-clean-warnings.log` and
  `maintenance/runs/stage8c1-doc-clean-audit.json`. Build with `READTHEDOCS=True`,
  `MPLBACKEND=Agg`, `QT_QPA_PLATFORM=offscreen`, and Miniforge
  `conda run --no-capture-output -n py314 python -m sphinx -b html -E
  maintenance/runs/stage8c1-docs-clean/doc maintenance/runs/stage8c1-doc-clean-output
  -w maintenance/runs/stage8c1-doc-clean-warnings.log`.
  The isolated source snapshot copies current tracked `Stoner`, `doc`, `sample-data`,
  `scripts` and `README.rst` files, including working-tree edits. An earlier broad
  copy included stale untracked autosummary stubs and omitted referenced assets;
  the corrected tracked-source snapshot avoids both without modifying originals.
  Direct Python Sphinx startup exited without diagnostics; `conda run` succeeded.
  The developer guide's new example and Data API navigation were checked over
  localhost; its HTML is byte-identical between that preview and the final clean
  output. The temporary server and browser tab were closed after review.
- **Stage 8c.2 implementation (2026-10-16):** `DataOwner.frame` returns the
  authoritative ordinary DataFrame without a copy, substitute frame or NumPy
  export. `replace_pandas(frame)` requires identical axes and column dtypes,
  validates before publication, detaches incoming values, and retains stable IDs,
  roles, exclusions, dtype, fill and typed/nested metadata. Frame attrs do not
  provide Stoner side state. Failed replacement leaves the old frame live;
  successful value/frame commits can retire it. Always reacquire after owner edits.
  `Stoner/core/storage.py::_validate_table` shares validation between UUID-labelled
  version-one packages and header-labelled owner state without copying values.
  Owner reads/writes, saved exclusion/schema/metadata proxies and current Data
  bridge/length/Setas entry boundaries reject detectable labels, shape and dtype
  drift with `ValueError`. There is no inferred correspondence for duplicate-name
  reorders or detection guarantee for same-shaped permutations/cosmetic annotations.
  Getter calls and replacement reject active drafts; both editing contexts
  revalidate the owner before committing and release their lock after failure.
  Previously obtained native references cannot be intercepted: independent writes
  are outside draft rollback, and concurrent native/owner mutation is unsupported.
  `tests/Stoner/core/test_storage_native_owner.py` adds 36 cases for these contracts,
  including no-copy/native-access instrumentation, current Data entry guards,
  retained references, empty axes, raw integer/mask recovery and recursive metadata.
  The initial focused run caught metadata-proxy attribute lookup recursing during
  deepcopy reconstruction; unsupported attributes now raise before owner lookup.
  The first full run then exposed the proxy's shared use by image/stack/frame
  owners, which have no DataFrame validator. Data read checks now live on the
  Data-only `_DataMetadata` proxy; the shared image proxy retains its existing
  read behaviour. A new image/stack-owner regression and the real PNG/EDF/HDF5
  autoload test cover this boundary. Final rerun evidence follows below.
  No regression test was weakened. Existing 8a/8b/8c.1 work is preserved on `devel`
  at `721ee3db7`; this batch does not commit, push, tag or publish.
- **Stage 8c.2 focused validation:** after the shared-proxy correction, the expanded
  py314 core/selection-consumer/real-image gate passed **309 tests, five warnings
  in 60.36 s** (`maintenance/runs/20261016-224859-focused-3056d114`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check focused -Environment py314
  -TestPaths tests/Stoner/core,tests/Stoner/analysis/test_operation_selection.py,tests/Stoner/analysis/test_storage_consumers.py,tests/Stoner/test_FileFormats.py::test_ImageAutoLoad`,
  substituting `py312` for that interpreter. Verified py314 versions remain Python
  3.14.7, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, xarray 2026.7.0 and pytest 9.1.1;
  py312 remains Python 3.12.14, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, xarray
  2026.7.0 and pytest 9.1.1. The final expanded py312 gate passed **309 tests, five
  warnings in 70.72 s** (`maintenance/runs/20261016-225022-focused-d1a762fb`).
- **Stage 8c.2 lower-dependency validation:** created an isolated pure-Conda
  environment from the existing `tests/minimum-env.yml` at
  `maintenance/runs/stage8c2-minimum-env`, without modifying py312/py314 or dependency
  manifests. Verified Python 3.12.15, NumPy 2.0.2, pandas 2.2.3, SciPy 1.14.1,
  xarray 2024.10.0 and pytest 9.1.1. The same expanded focused gate passed
  **308 tests, one skipped, 16 warnings in 54.19 s**
  (`maintenance/runs/20261016-225333-focused-45494112`). The existing
  `test_legacy_pandas_metadata_loss` skips because it characterises the pandas 3
  accessor lifecycle; no new skip or expected failure was added. All new native
  owner/reference tests run on both pandas 2.2 and 3.0. Reproduce with the same
  focused command and
  `-EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env` instead of
  `-Environment py314`. Environment creation used
  `conda env create --prefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env
  --file tests/minimum-env.yml`; its log is `maintenance/runs/stage8c2-minimum-env-create.log`.
  This is focused lower-bound evidence, not the full hosted minimum-dependency gate.
- **Stage 8c.2 full-suite exit:** after the shared-image-proxy correction, the
  final runtime/test state passed **812 tests, 711 warnings in 501.39 s**, with
  two workers and **80% coverage** on Miniforge py314. Evidence:
  `maintenance/runs/20261016-225023-parallel-d80fed1b` (`run.log`, `pytest.xml`,
  `coverage.xml`, `result.json`). Reproduce with
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314`.
  Undefined-name checks pass on the changed Python files with
  `python -m ruff check --select F821`; CRLF-aware whitespace checks pass with
  `git -c core.whitespace=cr-at-eol diff --check`. No permission/line-ending
  normalisation was performed, and all 243 tracked plot-cache files remain
  unchanged. Full lower-bound/hosted CI, installed-package checks and repeated
  integrated benchmarks remain Stage 8f work. These local test timings are not
  controlled performance measurements. Stage 8c.2 is complete; public `Data.data`
  remains the numerical snapshot pending the 8c.3 consumer migration.
- **Stage 8c.2 documentation:** cached HTML built from current tracked sources in
  `maintenance/runs/stage8c2-docs`, preserving original generated stubs and caches.
  `maintenance/runs/stage8c2-doc-output` passes the Windows warning/API audit:
  **88 reviewed warnings, zero unexpected warnings**, all five primary classes
  documented, no missing dynamic Data methods and no removed API objects compared
  with Stage 8c.1. Evidence: `stage8c2-doc-warnings.log` and
  `stage8c2-doc-audit.json` under `maintenance/runs`. Use the 8c.1 build/audit
  commands with these paths, `conda run -n py314` and the same headless/cached
  settings. Browser review over localhost verified the new owner-lifetime prose
  and Data API navigation; the temporary tab/server were closed after review.
- **Stage 8c.3a implementation (2026-10-16):** the first core numerical-consumer
  batch migrates `asarray`, `column`, `search_index`, `search`, `columns` and
  `DataFileOperatorsMixin.__eq__` independently of public `Data.data` reads.
  `asarray()` uses the explicit annotated, read-only snapshot helper; columns
  resolve through the owner. Numerical search predicates read the selected column,
  and search assembles its result from one masked export. Rectangular column
  selection retains duplicate/repeated positions, shape, dtype, fill, raw hidden
  values and exclusions; the old `numpy.column_stack` path silently lost masks.
  Boolean lists now dispatch as positional selectors before the interval branch,
  and short Boolean selectors pad correctly. NumPy integer/floating scalar targets
  are supported, including entries of numeric target arrays. Characterisation on
  NumPy 2.0.2 versus 2.5.2 showed that numeric array pairs changed from discrete
  targets to sequence-pattern intervals. Explicitly restricting interval patterns
  to tuples/lists restores documented array matching across those versions while
  retaining legacy two-item numeric-list intervals (including fitting row bounds).
  Column iteration captures initial mask eligibility and yields current detached
  owner columns, as before. Equality checks shape without conversion and exports
  each operand once; its existing masked comparisons, independent NaN filtering,
  header/metadata checks and omission of role/ID comparison remain unchanged.
  `tests/Stoner/core/test_native_numerical_consumers.py` tests both current snapshots
  and a test-only native DataFrame getter; a separate poison getter rejects any
  public `.data` read by the selected consumers. Additional cases cover raw masks,
  duplicate/repeated output columns, short Boolean forms, numeric scalar/array
  targets, conversion counts, snapshot annotations/detachment and iterator lifetime.
  The public getter, numerical mutation/return conventions, version and dependency
  manifests remain unchanged. User search/upgrade guidance and ownership/contracts
  now document the selector corrections and preserved legacy rules. Existing
  uncommitted 8a/8b/8c.1/8c.2 work remains intact on `devel` at `721ee3db7`.
- **Stage 8c.3a validation (2026-10-16):** the final focused command was
  `./maintenance/run-baseline.ps1 -Check focused -Environment py314 -TestPaths tests/Stoner/core,tests/Stoner/test_Core.py,tests/Stoner/analysis/test_operation_selection.py,tests/Stoner/analysis/test_storage_consumers.py`:
  **380 passed**, 6 warnings, 59.35 s. The same paths with `-Environment py312`
  gave **380 passed**, 6 warnings, 87.24 s; with
  `-EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env` they
  gave **379 passed, 1 skipped**, 17 warnings, 83.59 s. The existing skip is the
  pandas-3-specific legacy metadata-loss test. Reports are under
  `maintenance/runs/20261016-231226-focused-780013db`,
  `20261016-231425-focused-1f24ca06` and `20261016-231423-focused-d8d12ec6`.
  Runtime versions were Python 3.14.7/NumPy 2.5.2/pandas 3.0.5/SciPy 1.18.0/
  xarray 2026.7.0 for py314; Python 3.12.14/NumPy 2.5.3/pandas 3.0.5/
  SciPy 1.18.1/xarray 2026.7.0 for py312; and Python 3.12.15/NumPy 2.0.2/
  pandas 2.2.3/SciPy 1.14.1/xarray 2024.10.0 for the isolated minimum prefix.
  All used pytest 9.1.1; no environment or dependency files changed in this batch.
  Full-suite command
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed:
  **871 passed**, 711 warnings, **80% coverage**, 509.55 s, using two workers.
  Evidence: `maintenance/runs/20261016-231419-parallel-6a0fab8e/run.log`,
  `pytest.xml` and `coverage.xml` in that report directory.
  Targeted Ruff `F821` checks and
  `git -c core.whitespace=cr-at-eol diff --check` passed. All 243 tracked
  `doc/plot_cache` files are unchanged. A clean snapshot of current tracked source
  built with `READTHEDOCS=True`, `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen`:
  **88 reviewed warnings, no unexpected warnings**, all five primary classes
  documented, no missing dynamic Data methods or removed API entries against the
  8c.2 output. Evidence: `maintenance/runs/stage8c3a-doc-build.log`,
  `stage8c3a-doc-warnings.log`, `stage8c3a-doc-audit.json` and
  `stage8c3a-doc-output`. Browser review over localhost verified the search examples,
  search API navigation, upgrade notes and developer guidance; the temporary tab
  and server were closed. These are source-checkout results; installed-package,
  hosted matrix and performance exit gates remain Stage 8f work.
- **Stage 8c.3b-8e implementation (2026-10-16):** `Data.data` now returns the
  owner's actual header-labelled DataFrame. DataFrame assignment uses the validated
  fixed-schema replacement route; numerical assignment retains the explicit
  positional replacement interface. Rolling windows read one current masked export
  per yield, preserving masks, raw values, dtype and fill without clones or
  unmasking concatenations. They preserve one-prefix/one-suffix wrapping on short
  inputs; centre-exclusion bounds now clip instead of selecting unrelated rows via
  negative endpoints. Selection/splitting, TDI input/output, records, transpose,
  display, HDF output, folder indexing and image-to-Data conversion use explicit
  numerical boundaries. XOR now produces deterministic Boolean comparisons for
  equal inputs instead of uninitialised `empty_like` output.
  `ImageFile.image`/`.data` and `ImageStack.stack` expose actual root Datasets;
  `ImageFile.imarray` and stack `imarray` retain read-only numerical snapshots.
  Native fixed-schema Dataset replacements copy their inputs and retain typed
  metadata/frame identities. Image/stack owners reject detectable coordinate,
  extent, variable, attribute and dtype drift after native access; native getters
  and replacements respect transaction locks. Baselines track geometry without
  duplicating intensity buffers. Unexposed internal states retain normal validated
  owner edit paths without repeatedly scanning native drift. Borrowed frame/crop
  getters resolve current views, with direct view write-back unsupported; owner
  pixel/mask transactions retain their shared/stable-ID semantics. Masked ImageFile
  assignment now explicitly exports exclusions rather than relying on raw NumPy
  coercion. Arithmetic, display/save, folder, Kerr, scan-frame and real-format
  callers use numerical adapters. No channel-axis redesign or wrapper expansion
  is included. `where` remains supported by curve_fit/image mean/stack average;
  neighbourhood filters and other fitting engines reject it explicitly before
  backend dispatch. Guides, examples, installed-package smoke checks and numerical
  regressions use the new public types or explicit exports as appropriate.
  `tests/Stoner/core/test_native_public_storage.py` adds actual native identity,
  no-export, replacement, drift, transaction, reference, ragged-padding,
  frame/crop, masked assignment, short-window and unsupported-selection checks.
  All pre-existing migration work is retained. The maintainer authorised committing
  and pushing the accumulated migration to `devel` after local validation in this
  session, so hosted final-interface and installed-distribution checks can run.
- **Stage 8f initial native-interface validation (2026-10-17, `a6235d922`):** runtime fingerprint
  `cae0153e76d7a1babf73f92e4d57680e9166566dacd3b70ab633aad1ac0cfdf0`.
  `./maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed
  **898 tests**, 708 warnings, **81% coverage**, 260.51 s with two workers.
  Evidence: `maintenance/runs/20261016-235733-parallel-42fd0791`.
  The final focused command uses
  `-Check focused -EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env -TestPaths tests/Stoner/core,tests/Stoner/Image,tests/Stoner/analysis/test_operation_selection.py,tests/Stoner/analysis/test_storage_consumers.py`:
  **583 passed, 1 existing pandas-3-specific skip**, 342 warnings, 88.27 s.
  Evidence: `maintenance/runs/20261016-235732-focused-79ec2298`. Runtime dependency
  versions remain those recorded in 8c.3a; no environment/dependency changes were
  made. The last targeted native drift/ownership/placeholder guard check passed
  **91 tests** (`maintenance/runs/stage8-native-guards-final.log`). The final
  installed-package smoke script's source preflight passes native type/identity,
  recoverable raw exclusions, shared frame writes, resource and TIFF checks;
  actual clean installed-import checks remain hosted evidence.
  Cached Sphinx build and API audit pass with **88 reviewed warnings, zero
  unexpected warnings**, all primary classes documented, no missing dynamic Data
  methods and no removed API objects against 8c.3a. Evidence:
  `maintenance/runs/stage8-exit-doc-build.log`, `stage8-exit-doc-warnings.log`,
  `stage8-exit-doc-audit.json` and `stage8-exit-doc-output`. Browser inspection
  over localhost verified native table/image/stack guidance and navigation; the
  temporary tab/server were closed. All 243 tracked plot-cache files are unchanged.
  Ruff `F821` over runtime/tests/changed maintenance tools and staged whitespace
  checks pass. New tests are indexed with canonical `tests/Stoner` capitalisation.
- **Stage 8f performance review:** the first complete integrated measurement
  passed 16/19 pairs, flagging small row appends and both frame edits. Profiling
  found repeated validation of trusted internal table replacements and repeated
  construction/validation of borrowed frame views. Owners now track whether the
  current backend object has escaped through a native getter: exposed native
  state still validates at every boundary, while already-validated unexposed
  state avoids redundant scans. Retired references are tracked by identity without
  retaining old measurement buffers. Shared handles validate root state and
  liveness without constructing an image merely to validate it. Row insertion
  retains one complete validation before adoption. All four flagged workload/size
  pairs then passed, with small appends 10.285 -> 12.618 ms, small frame edits
  0.492 -> 3.154 ms, large appends 53.259 -> 64.929 ms and large frame edits
  0.516 -> 3.445 ms. The complete final benchmark passed **19/19 pairs**, five
  repetitions with separate traced allocations, against the accepted legacy
  limits. Evidence: `maintenance/runs/stage8-exit-integrated.json` and `.log`;
  its production fingerprint matches the final runtime above. Large appends
  measured 46.102 -> 57.577 ms, stack insertion 234.237 -> 11.371 ms and frame
  editing 0.421 -> 3.344 ms. `maintenance/runs/stage8-versus-stage7.json` also
  compares all 19 production measurements with Stage 7: large appends
  61.440 -> 57.577 ms, large conversion 12.538 -> 4.248 ms, large frame editing
  3.222 -> 3.344 ms; small appends 9.840 -> 15.713 ms remain inside the agreed
  legacy limit. Cross-session Stage 7 timings are descriptive, not a claim that
  every difference is caused by this change. Development or interrupted
  source-fingerprint runs do not close the gate.
  Final native-read allocations pass against masked exports: table **2,401 vs
  3,604,549 bytes**, image **7,234 vs 789,339**, stack **133,305 vs 789,799**.
  Regional image selection peaks at 1,313,963 bytes and selected mean at 1,379,622
  bytes. Evidence: `maintenance/runs/stage8-exit-native-allocations.json`, including
  resolved environment and the same final runtime fingerprint. These are allocation
  measurements, not a claim that native computations automatically apply masks.
- **Stage 8f initial hosted evidence (`a6235d922`):** all four full matrix cells
  passed **898 tests** each: Linux Python 3.12/3.13/3.14 and macOS Python 3.14;
  final result publishing also passed. [Matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38979578424).
  The full lower-dependency suite passed **897 tests, 1 existing skip** on NumPy
  2.0.2/pandas 2.2.3/SciPy 1.14.1/xarray 2024.10.0.
  [Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38979578508).
  Fresh wheel/sdist installs, archive assets and native-interface/resource/TIFF
  smoke checks passed on both Python 3.12 and 3.14.
  [Installed distributions](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38979578436).
  Hosted documentation passed with **61 reviewed Linux warnings**, no unexpected
  warnings after normalising GitHub's escaped terminal colours against the unchanged
  Linux warning baseline.
  [Documentation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38979611113).
  Local log copies: `maintenance/runs/stage8-hosted-matrix-initial.log`,
  `stage8-hosted-minimum.log`, `stage8-hosted-packages.log`, `stage8-hosted-docs.log`
  and `stage8-hosted-doc-warning-audit.json`.
- **Stage 8e final keyword audit:** legacy numerical methods accepting arbitrary
  keyword dictionaries could silently ignore `where`. `_reject_where` now rejects
  it before work/mutation in `apply`, `decompose`, `integrate`, `bin`,
  `outlier_detection`, `peaks`, `find_peaks`, `section`, `swap_column` and the
  `hysteresis_correct` utility. Their existing numerical and callback behaviour
  remains unchanged without this unsupported top-level keyword. `Data.select`
  keeps its column-query namespace, including columns named `where`; `Data.apply`
  retains explicit callback arguments through `_extra={"where": value}`. The guide
  and interchange contract document those distinctions. New tests check rejection
  before callback/backend work, unchanged raw values/masks/schema/typed metadata,
  and preserved query/callback namespaces. The reusable native benchmark now
  explicitly selects the source checkout before importing Stoner.
- **Stage 8f final keyword-follow-up local evidence:** runtime fingerprint
  `6e0b72f062eca9d4287cb5800ecfd44064698cb04436ce6466950e3d20b3d281`.
  The full py314 command above passed **909 tests**, 708 warnings, **81% coverage**,
  225.17 s (`maintenance/runs/20261017-002136-parallel-18c560cc`). The same focused
  minimum-dependency paths passed **594 tests, 1 existing skip**, 342 warnings,
  77.00 s (`maintenance/runs/20261017-002136-focused-016a7e85`). The final namespace
  and selection regression rerun passed **50 tests**, including explicitly
  forwarded callback `where` (`maintenance/runs/stage8-keyword-final.log`). Cached
  Sphinx rebuilt with **88 reviewed warnings**, zero unexpected warnings, no missing
  dynamic methods or removed API entries (`stage8-complete-doc-build.log`,
  `stage8-complete-doc-warnings.log`, `stage8-complete-doc-audit.json` and
  `stage8-complete-doc-output` under `maintenance/runs`). The revised callback/query
  guidance was visually checked over localhost; its temporary tab/server were closed.
  Final integrated benchmarks passed **19/19 pairs**, five repetitions with separate
  traced allocations (`maintenance/runs/stage8-complete-integrated.json` and `.log`).
  Its runtime fingerprint matches the native allocation report. All 19 workloads
  are also compared with Stage 7 in `stage8-complete-versus-stage7.json`: large
  appends 61.440 -> 56.713 ms, conversion 12.538 -> 4.039 ms and frame edits
  3.222 -> 3.291 ms. Final native-read peaks are table 2,401 vs 3,604,549 bytes,
  image 7,569 vs 789,686 and stack 133,313 vs 789,754; selected-image input/mean
  peaks are 1,314,163/1,379,814 bytes (`stage8-complete-native-allocations.json`).
  Static/whitespace checks pass and all 243 tracked plot-cache files remain unchanged.
- **Stage 8f final hosted evidence (2026-10-17, `449c2eb251adf1477e124ed8769b306772101614`):**
  all four full matrix cells passed **909 tests** each: Linux Python 3.12
  (180.86 s), 3.13 (177.29 s), 3.14 (180.39 s), and macOS Python 3.14
  (498.54 s). Final result publishing also passed.
  [Full matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38981834098).
  The supported lower-dependency full suite passed **908 tests, 1 existing skip**,
  6,942 warnings, 172.45 s, with Python 3.12/NumPy 2.0.2/pandas 2.2.3/
  SciPy 1.14.1/xarray 2024.10.0.
  [Lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38981834059).
  Fresh wheel and sdist installs on Python 3.12 and 3.14 passed archive-content,
  out-of-checkout import provenance, native-interface, resource and TIFF checks.
  [Installed distributions](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38981834151).
  Hosted documentation passed with **61 reviewed Linux warnings, zero unexpected**
  against the unchanged warning baseline.
  [Documentation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/38981842460).
  Local evidence under `maintenance/runs`: `stage8-complete-hosted-matrix.log`,
  `stage8-complete-hosted-minimum.log`, `stage8-complete-hosted-packages.log`,
  `stage8-complete-hosted-docs.log` and
  `stage8-complete-hosted-doc-warning-audit.json`.
  These hosted gates and the matching local runtime fingerprint above close Stage 8.
  The subsequent completion record changes only the two Markdown plans; validation
  remains attached to this exact runtime revision, with no source, test or dependency
  changes in the completion record.
- **Stage 8 completion handover:** Stage 9 follows the validated native ownership
  interface; the first batch and current next step are recorded below. Retain
  scientific fixtures and all 243 cached plots. Publication remains a separately
  authorised release action.
  The maintainer subsequently authorised committing this migration and version
  `0.12.0a1` to `devel` and pushing to origin. Publication remains a separate action.

- **Stage 9a implementation (2026-10-17):** added `head`/`tail` in
  `Stoner/core/methods.py`, sharing a private row-preview adapter. Pandas determines
  positions; existing clone and owner selection preserve scientific and ownership
  contracts. The inventory and complete wrapper policy are recorded above.
  `tests/Stoner/core/test_row_previews.py` covers negative/default/zero/oversize
  counts, empty axes, duplicate headers, roles/IDs, recoverable exclusions,
  dtype/fill (including large uint64 and complex values), independent values and
  nested typed metadata/cycles, subclass preservation, native drift, editing guards
  and rejection of unsupported counts/keywords. The guide adds usage examples;
  the `Data` class overview now correctly describes `.data` as a DataFrame.
  Existing sorting, predicate selection, duplicate reduction and image operations
  remain unchanged. Previews use the established full-clone path before trimming;
  reducing that allocation requires a separate measured copying-boundary change.
- **Stage 9a validation:** verified Miniforge `py314` (Python 3.14.7, NumPy 2.5.2,
  pandas 3.0.5, SciPy 1.18.0, xarray 2026.7.0, pytest 9.1.1) and the existing
  pure-Conda lower-dependency prefix `maintenance/runs/stage8c2-minimum-env`
  (Python 3.12.15, NumPy 2.0.2, pandas 2.2.3, SciPy 1.14.1, xarray 2024.10.0,
  pytest 9.1.1); no dependencies or environment specifications changed.
  The new preview tests passed **52 tests** on py314. The lower-dependency command
  `maintenance/run-baseline.ps1 -Check focused -EnvironmentPrefix
  D:\PythonCode\maintenance\runs\stage8c2-minimum-env -TestPaths
  tests/Stoner/core/test_row_previews.py,tests/Stoner/core/test_storage_structural.py,tests/Stoner/core/test_native_public_storage.py`
  passed **106 tests**, 13 warnings, 35.64 s
  (`maintenance/runs/20261017-180023-focused-a2bdb8e5`). The first full py314 run
  passed 959 tests and exposed two fixed API-directory counts. Updated
  `tests/Stoner/test_Core.py` and `tests/Stoner/folders/test_each.py` to explicitly
  require `head`/`tail` while retaining their existing remainder-count assertions.
  The focused command with `-TestPaths tests/Stoner/test_Core.py::test_dir,
  tests/Stoner/folders/test_each.py::test_each_call,tests/Stoner/core/test_row_previews.py`
  passed **54 tests** on py314 (1 warning, 21.52 s;
  `maintenance/runs/20261017-180614-focused-5e4e7b17`) and the lower prefix
  (13 warnings, 9.58 s; `maintenance/runs/20261017-180703-focused-99f2f39d`).
  Final `maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed
  **961 tests**, 708 warnings, **81% coverage**, 298.88 s with two workers
  (`maintenance/runs/20261017-180702-parallel-58c9cabf`). This final run includes
  the new previews and both explicit API-inventory updates; Stage 9a is complete.
  Cached Sphinx HTML built from a clean tracked-source snapshot with
  `READTHEDOCS=True`, **88 reviewed Windows warnings and zero unexpected warnings**
  against the unchanged baseline. `maintenance/audit-docs.py` compared it with
  Stage 8: all primary classes and dynamic methods are documented, with no removed
  API objects. Evidence: `maintenance/runs/stage9a-doc-build.log`,
  `stage9a-doc-warnings.log`, `stage9a-doc-audit.json` and `stage9a-doc-output`.
  Localhost visual review checked the new guide section and both API links;
  temporary browser tab and server were closed. Ruff F821 and whitespace checks
  pass; all 243 tracked plot-cache files remain unchanged. This batch has local
  validation only; the hosted Stage 8 runs above are not Stage 9 evidence.
- **Stage 9b implementation (2026-10-17):** `Stoner/core/storage_operations.py`
  adds a private numeric-key permutation adapter and checks writability before
  comparison/backend work. Compatible real keys use a positional pandas key frame;
  unsupported dtypes and custom comparators retain the numerical path described
  above. No storage-owner or image interfaces changed. The `sort` docstring and
  guide now correctly describe its in-place return, raw exclusions, role defaults,
  duplicate-name handling, stable ascending ties, whole-permutation reversal and
  native-reference lifetime; the old guide incorrectly claimed ascending-only
  support. `tests/Stoner/core/test_numeric_sorting.py` characterises the legacy
  contract and verifies delegated/fallback paths, ownership, atomic selector and
  comparator failures, dtype precision, empty axes and early editing/drift guards.
- **Stage 9b validation:** the initial pre-change characterisation passed **64
  tests** on py314 (`maintenance/runs/20261017-185738-focused-7ed905cb`). After
  implementation, numeric sorting plus structural tests passed **88 tests** on
  py314 (`maintenance/runs/20261017-185927-focused-eeb6f347`, 7.23 s). The final
  lower-dependency command `maintenance/run-baseline.ps1 -Check focused
  -EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env
  -TestPaths tests/Stoner/core/test_numeric_sorting.py,tests/Stoner/core/test_storage_structural.py`
  passed **88 tests**, 13 warnings, 6.89 s
  (`maintenance/runs/20261017-190110-focused-d30d7d12`). Float16 fixtures now import an explicit
  valid fill value, avoiding unrelated constructor-default overflow warnings.
  Final `maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed
  **1,033 tests**, 708 warnings, **81% coverage**, 214.82 s with two workers
  (`maintenance/runs/20261017-190207-parallel-588f2519`), including both the retained
  Stage 9a previews and all 72 new sorting cases. Stage 9b is complete.
  Verified environments remain
  py314 Python 3.14.7/NumPy 2.5.2/pandas 3.0.5/SciPy 1.18.0/xarray 2026.7.0 and
  minimum Python 3.12.15/NumPy 2.0.2/pandas 2.2.3/SciPy 1.14.1/xarray 2024.10.0;
  both use pytest 9.1.1. No dependency/environment specifications changed.
  Exploratory pandas comparisons covered **900 permutations per environment**:
  all 600 compatible real-dtype cases matched; float16 and complex differences
  justify the retained paths (`maintenance/runs/stage9b-sort-review-current.json`
  and `stage9b-sort-review-minimum.json`). Complete-sort benchmarks compare the
  exact pre-batch function from `6f0ca8045` against this adapter, using five timed
  repetitions plus separate traced allocations on 1,000/100,000-row, eight-column
  float64/uint64 tables with one/three keys. All **8/8 pairs** meet the recorded
  time limit (larger of 2x legacy or legacy + 5 ms); all allocation peaks are below
  legacy, also satisfying its 2x ceiling. Large float64 sorts measured 24.930 ->
  23.722 ms (one key), 37.375 -> 27.371 ms (three); uint64 22.180 -> 21.953 ms and
  32.566 -> 23.384 ms. Large traced peaks fell from about 22.54 MB to 16.14 MB.
  Evidence: `maintenance/runs/stage9b-sort-benchmark.json` and `.log`, runtime
  fingerprint `ed4c6982a5fd5decfdb23a446215d7e2b68c7edcc5eb8fd82c50e98778ad1ab8`.
  These are focused warm-operation measurements, not total process RSS or a new
  package-wide performance claim. Cached Sphinx HTML built from a clean tracked
  source snapshot (`READTHEDOCS=True`) with **88 reviewed Windows warnings, zero
  unexpected warnings**, all primary classes/dynamic methods documented and no
  removed API entries versus 9a. Evidence: `maintenance/runs/stage9b-doc-build.log`,
  `stage9b-doc-warnings.log`, `stage9b-doc-audit.json` and `stage9b-doc-output`.
  Localhost visual review checked the sorting guide, API page and guide-to-API
  link; the temporary browser tab/server were closed. Ruff F821 and whitespace
  checks pass; all 243 tracked plot-cache files remain unchanged. This batch has
  local validation only; hosted Stage 8 results are not Stage 9b evidence.
- **Stage 9c review (2026-10-17):** retained `Data.unique`'s numerical implementation
  unchanged after the documented backend comparison. Updated its docstring and
  `doc/UserGuide/datafile.rst` to specify optional tuple order, dtype/fill, masks,
  flattening, first-occurrence indices, inverse shape, numerical ownership and
  committed-state reads. `tests/Stoner/core/test_unique_contract.py` adds 54
  boundary cases covering all flag combinations, finite/mixed/all-excluded input,
  duplicate-header and multi-column selectors, empty selections, full-width uint64,
  Boolean/float/complex dtypes, NaN/complex-NaN equivalence, result independence,
  schema/metadata identity, active drafts, native drift and unsupported `where`.
  No runtime algorithm, storage owner, image interface or public method name changed.
- **Stage 9c validation:** py314 characterisation passed **54 tests**, 1 existing
  SciPy warning, 7.06 s (`maintenance/runs/20261017-215219-focused-84008e72`).
  Exploratory reports compare seven representative input families and all four
  index/inverse combinations on current NumPy 2.5.2/pandas 3.0.5 and minimum NumPy
  2.0.2/pandas 2.2.3: `maintenance/runs/stage9c-unique-review-current.json` and
  `stage9c-unique-review-minimum.json`. Both demonstrate encounter-order and
  mask/complex-NaN differences in raw pandas uniqueness; they also record raw
  `factorize(sort=True)` codes/categories and `drop_duplicates` results.
  The lower-dependency command `maintenance/run-baseline.ps1 -Check focused
  -EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env
  -TestPaths tests/Stoner/core/test_unique_contract.py,tests/Stoner/core/test_storage_structural.py`
  passed **70 tests**, 13 warnings, 6.29 s
  (`maintenance/runs/20261017-215336-focused-08a6bfb9`). Verified environments remain
  py314 Python 3.14.7/NumPy 2.5.2/pandas 3.0.5/SciPy 1.18.0/xarray 2026.7.0 and
  minimum Python 3.12.15/NumPy 2.0.2/pandas 2.2.3/SciPy 1.14.1/xarray 2024.10.0,
  both with pytest 9.1.1; no dependency/environment files changed.
  Final `maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed
  **1,087 tests**, 708 warnings, **81% coverage**, 268.45 s with two workers
  (`maintenance/runs/20261017-215337-parallel-ef11f450`), including retained Stage
  9a/9b behaviour and all 54 uniqueness cases. Stage 9c is complete.
  Cached Sphinx HTML built from a clean
  tracked-source snapshot with `READTHEDOCS=True`: **88 reviewed Windows warnings,
  zero unexpected warnings**, all primary classes and dynamic methods documented,
  and no removed API entries versus 9b. Evidence: `maintenance/runs/stage9c-doc-build.log`,
  `stage9c-doc-warnings.log`, `stage9c-doc-audit.json` and `stage9c-doc-output`.
  Localhost visual review checked the guide, API contract and guide-to-API link;
  temporary tab/server were closed. Ruff F821 and whitespace checks pass, with
  all 243 tracked plot-cache files unchanged. No new benchmark claim is made:
  this batch leaves the executable uniqueness path unchanged. Validation is local;
  previous hosted Stage 8 runs are not evidence for Stage 9c.
- **Stage 9d implementation (2026-10-17):** replaced preview clone-then-trim on the
  standard path with selected-row frame/mask copies and a pre-seeded deepcopy memo.
  Changed `Stoner/core/methods.py`, `storage_bridge.py` and the two numerical-leaf
  copies in `_TableState.__deepcopy__` in `storage_owner.py`. Custom hooks retain the
  old wrapper path. Existing preview/scientific contracts pass; twelve new tests in
  `tests/Stoner/core/test_row_previews.py` protect shared attribute/typed metadata
  cycles, source-independent frame/mask aliases, inherited subclass attributes,
  absence of full-frame copying/NumPy exports/owner trimming, custom clone/deepcopy
  transformations, failure atomicity and the canonical result-frame policy. Full raw
  frame/mask backups in metadata/attributes remain full independent snapshots, with
  their shared aliases intact. The guide/docstrings explain selected-row
  copying and the retained ancillary/custom-hook costs. No new public API added.
- **Stage 9d validation:** final current focused tests passed **99 tests**,
  1 existing warning, 9.50 s (`maintenance/runs/20261017-221651-focused-20c34ce3`).
  The final lower-dependency command `maintenance/run-baseline.ps1 -Check focused
  -EnvironmentPrefix D:\PythonCode\maintenance\runs\stage8c2-minimum-env
  -TestPaths tests/Stoner/core/test_row_previews.py,tests/Stoner/core/test_storage_structural.py,tests/Stoner/core/test_storage_bridge.py,tests/Stoner/core/test_storage_owner.py`
  passed **99 tests**, 13 warnings, 7.87 s
  (`maintenance/runs/20261017-221742-focused-adbabcd9`). Environments/dependencies
  remain as verified in 9c; no environment or packaging specifications changed.
  Final `maintenance/run-baseline.ps1 -Check parallel -Environment py314` passed
  **1,099 tests**, 708 warnings, **81% coverage**, 282.27 s with two workers
  (`maintenance/runs/20261017-221836-parallel-dbc30922`). This exact final source
  includes canonical backend-frame policy and full ancillary raw-backup preservation;
  its fingerprint matches the final benchmark below. Stage 9d is complete.
  Cached Sphinx HTML built from a clean
  tracked-source snapshot with `READTHEDOCS=True`: **88 reviewed Windows warnings,
  zero unexpected warnings**, all primary classes/dynamic methods documented and
  no removed API entries versus 9c. Final-source evidence:
  `maintenance/runs/stage9d-verified-doc-build.log`, `stage9d-verified-doc-warnings.log`,
  `stage9d-verified-doc-audit.json` and `stage9d-verified-doc-output`.
  Localhost visual review checked the guide, both preview API contracts and the
  guide-to-tail link; temporary browser tab/server were closed. All 243 tracked
  plot-cache files remain unchanged. This batch's validation is local; earlier
  hosted Stage 8 runs are not Stage 9d evidence.
  Measured the unmodified 9a wrapper first (`maintenance/runs/stage9d-preview-baseline.json`,
  runtime fingerprint `7e3de2d60d72b6811e36eee5c7bb26b14964b887e7389a21c1cdc14480e0e72b`)
  and retained its exact function in the ignored benchmark snapshot. Final paired
  complete-preview benchmarks use 1,000/100,000-row, eight-column float64 inputs,
  both ends, counts 0/5/-5/oversize, five timed repetitions and separate traced
  allocations. All **16/16 pairs** meet the accepted time limit (larger of 2x
  legacy or legacy + 5 ms) and the 2x allocation ceiling; every peak is below legacy.
  For 100,000 rows, head(5) measured 12.302 -> 6.992 ms and 14,412,430 -> 30,514
  peak bytes; tail(5) 12.444 -> 6.865 ms and 14,414,063 -> 27,861 bytes. Full-size
  previews fell from about 23.34 MB to 7.23 MB peak. These are warm complete-operation
  measurements and traced allocations, not process RSS. Evidence:
  `maintenance/runs/stage9d-preview-final.json` and `.log`, runtime fingerprint
  `a958a5809ae60494188e04dda3779f4f64624f6f18fa80ec3aa389a042b5be94`.
  The paired comparison isolates the old/new wrappers on the same copy foundation;
  the pre-change capture is a separate contextual baseline. Ruff F821 and whitespace
  checks pass.
- **Stage 9e cumulative review (2026-10-17):** reviewed the complete 9a-9d runtime,
  regression and documentation changes. Confirmed positional independent previews,
  compatible pandas sorting with numerical fallbacks, retained NumPy uniqueness,
  and selective preview copying with custom-hook fallback. No further runtime
  changes were required. The deepcopy graph, raw backup snapshots, exclusions,
  dtype/fill, schema and metadata contracts remain covered by regression tests.
- **Stage 9e installed-distribution coverage:** extended
  `maintenance/check-installed-package.py` to exercise head/tail raw values and
  masks, independent edits, schema/roles and typed recursive metadata, reverse
  sorting with ties/NaNs and exclusion movement, and masked uniqueness with both
  optional index outputs. Hosted wheel/sdist checks run outside the checkout on
  Python 3.12 and 3.14. Source preflights pass on py314 and the supported lower
  Python 3.12 environment; these alone do not establish installed-package success.
- **Stage 9e local checks:** F821 static checking and whitespace checks pass;
  all 243 tracked cached plots remain unchanged. Runtime source fingerprint is
  `a958a5809ae60494188e04dda3779f4f64624f6f18fa80ec3aa389a042b5be94`,
  matching the final 9d full-suite and documentation evidence above. Reuse that
  unchanged-source 1,099-test local pass. Fresh integrated and native-access
  benchmarks and exact-revision hosted gates follow before closing 9e.
- **Stage 9e measured exit gates:** fresh paired integrated measurements pass
  **19/19** time/allocation limits using five timing repeats and separate traced
  peaks, against the isolated pre-migration checkout. Evidence is ignored local
  `maintenance/runs/stage9-exit-integrated.json`; both import provenance and source
  fingerprints were checked during measurement. Production fingerprint matches
  the unchanged 9d runtime above. Native getter allocation checks pass **3/3**
  (`stage9-exit-native.json`): table 2,401 versus 3,604,549 bytes for masked export,
  image 7,675 versus 789,637, stack 133,350 versus 789,848. These are traced operation
  allocations, not process RSS; no new uniqueness speed claim is made. The 9b
  sorting and 9d preview-specific paired results remain applicable to this source.
- **Stage 9e hosted revision:** committed and pushed accumulated Stage 9 changes
  as `4525bf7d593788ef4d4e0d2d5afcba3f6ca85ca9` on `devel` using the existing
  migration authorisation. All final hosted gates below refer to that immutable
  revision, rather than the earlier Stage 8 runs.
  - [Supported test matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39071419390):
    **1,099 passed in every cell**: Linux Python 3.12, 3.13 and 3.14, plus macOS
    Python 3.14. Downloaded JUnit files confirm zero failures/errors/skips in each
    cell. Final test-result publishing succeeded, as did Ubuntu 3.14's Coveralls
    and Codacy coverage uploads; this does not claim separate service reanalysis.
  - [Supported lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39071419386):
    **1,098 passed, 1 existing skip**, including the new wrapper regressions. The
    skip is the pandas 3 accessor-lifecycle observation on the pandas 2 environment.
  - [Installed distributions](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39071419383):
    both Python 3.12/3.14 jobs passed archive content checks and independent wheel
    and sdist smoke checks outside the checkout, including reviewed wrappers.
  - [Cached documentation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39071419166):
    succeeded with **61 reviewed Linux warnings, zero unexpected**. The local
    warning audit is `stage9-exit-hosted-doc-warning-audit.json`. The unchanged
    source's Windows documentation/API inventory and visual review are recorded
    under 9d above; all 243 tracked cache files remain unchanged.
- **Stage 9 exit decision:** 9a-9e are complete. The cumulative review, final-source
  local tests, supported/lower hosted suites, installed archives, cached docs/API
  audit and measured performance satisfy the wrapper exit gate. Record this
  closure in a documentation-only commit; hosted runtime evidence remains pinned
  to `4525bf7d593788ef4d4e0d2d5afcba3f6ca85ca9` and its fingerprint above.
- **Next step:** follow the separate release-preparation checklist in
  `MAINTENANCE_PLAN.md` when release work is authorised. No release, tag, publication
  or Conda deployment is part of this batch. Retain scientific fixtures and all
  243 cached plots. Additional wrapper families, labelled-row APIs and lazy or
  broader coordinate-aware interfaces require independent bounded proposals;
  they are not outstanding Stage 9 requirements.

### Post-migration scan frame access (2026-10-18)

This bounded maintenance batch removes the numerical export from
`ScanMetadataMixin._instantiate`, shared by Attocube and MAXIMUS stacks. The
existing image handle already carries its presentation attributes; replacing its
metadata owner needs no detached pixels or exclusion mask. Selection still uses
the existing positional, name/channel and stable-frame-ID rules. No cache or
loader interpretation changes are introduced.

Saved handles continue to resolve their stable frame after reordering and raise
`ReferenceError` after deletion. Common metadata remains a live default beneath
typed local overrides. Pixel/mask writes and edit locks retain their owner path;
raw values, ragged extents, fill values, coordinate values/units and Dataset
attributes remain covered by the specialised and general image regressions.

The source-fingerprinted native-interface benchmark now includes metadata-only
Attocube frame access. On Windows py314, five warm timing repeats and a separate
traced-allocation measurement gave:

| Frame dimensions | Before median ms | After median ms | Before peak bytes | After peak bytes |
| ---------------- | ---------------- | --------------- | ----------------- | ---------------- |
| 64 x 64          | 1.927            | 0.839           | 375,274           | 209,900          |
| 1024 x 1024      | 16.068           | 0.820           | 20,234,538        | 195,855          |

These are frame-access measurements, not whole-loader timings or process RSS.
Local evidence is under ignored `maintenance/runs/attocube-frame-access/`:
`before.json`, `after.json` and the test/environment logs. The post-change runtime
source fingerprint is
`faf2b995fa8d64b784472d072f093558b39146e47a55cdb8235cfcebc7d4b1a5`.

Local validation on verified Conda environments:

- Current py314 full suite: **1,331 passed**, including runnable documentation
  samples and real Attocube HDF5/TIFF and MAXIMUS fixtures (`full.xml`).
- Supported lower Python 3.12 dependencies: **261 passed** across
  `tests/Stoner/Image`, `tests/Stoner/test_FileFormats.py` and
  `tests/Stoner/test_HDF5.py` (`minimum.xml`). Both environments satisfy all 38
  `tests/test-env.yml` specifications; dependency snapshots are retained locally.
- The four new scan regressions cover numerical-export avoidance and preserved
  scientific/live-handle state for both Attocube and MAXIMUS. The export guards
  fail on the original implementation and pass after the owner-only rebind.
- Ruff F821 and whitespace checks pass. The benchmark's three existing native
  getter allocation checks also pass. Runtime fingerprint verification confirms
  that the tested source still matches the post-change measurement.
- Cached Sphinx built from freshly staged sources passes the warning/API audit:
  **62 reviewed warnings, zero unexpected and no removed API entries**. Existing
  ignored generated API pages remain untouched; all 276 tracked plot-cache files
  remain unchanged. Release notes describe the frame-access improvement.

Hosted validation of pushed revision
`9b42bfd744c4ed47f41c0cd5644a94f1ecdb0471` (2026-10-18):

- [Supported test matrix](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39159346842):
  **1,331 passed in each job**, Linux Python 3.12, 3.13 and 3.14 plus macOS Python
  3.14. Downloaded JUnit reports confirm zero failures/errors/skips and all four
  new scan regressions passing in every cell. Final test-result publishing and
  Ubuntu 3.14's Coveralls/Codacy coverage uploads succeeded; this does not claim
  separate downstream service reanalysis.
- The macOS `test_attocube_scan` passed in **45.787 seconds**, compared with
  **201.434 seconds** in the retained JUnit report for `33b36ec81`. These are
  observed timings on hosted runners using the same workflow, not a controlled
  performance comparison. The metadata-only benchmark above remains the bounded
  allocation/time evidence. Linux Attocube timings were 16.114/16.579/8.201 seconds
  for Python 3.12/3.13/3.14 respectively.
- [Supported lower dependencies](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39159346843):
  **1,330 passed, one expected skip**. The skip is
  `test_legacy_pandas_metadata_loss`, a pandas 3 accessor-lifecycle observation
  that does not apply to pandas 2. All four scan regressions passed; the real
  Attocube test passed in 13.545 seconds.
- [Installed distributions](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39159346813):
  Python 3.12 and 3.14 jobs passed archive-content and strict metadata checks,
  followed by fresh wheel and sdist installations exercised outside the checkout.
- [Cached documentation](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39159360513):
  succeeded with **36 warnings**. Hosted-log review found the existing canonical
  `Stoner.Image.folders.ImageFolder.flatten` duplicate, already covered by the
  Windows warning baseline but absent from the Linux ceiling. Its diagnostic
  origin also varies between a docstring and `<unknown>`. This existing warning
  was reviewed explicitly; no baseline was raised. The remaining diagnostics
  match the reviewed Linux duplicate/footnote warnings. The local inventory audit
  above remains the evidence for no API removals.

Hosted snapshots, JUnit artifacts, full matrix/package/documentation logs and
`validation-summary.json` are retained under the ignored `hosted/` subdirectory
of this batch's evidence folder. All four workflows reference the pushed revision
above and completed successfully, including their reporting steps. The scan
optimisation is complete; later documentation-only notes do not change its
runtime fingerprint. No dependency, API, loader-identification or release change
is part of this batch.

### Accepted batch 3 performance decision

Accepted per-workload median-time limit: the larger of **2 x legacy median** or
**legacy median + 5 ms**. This avoids treating tiny absolute differences as
meaningful regressions while bounding larger workload slowdowns.

Accepted traced-peak-allocation limit: the larger of **2 x legacy peak** or
**3 x data-and-mask payload + 1 MiB**. Payload includes appended rows or the inserted
frame for growing workloads. This permits independent values/mask buffers and
bounded temporary conversions; it does not excuse unbounded accumulated state.
The assessment script computes these rules. The maintainer accepted progression
to stage 4 on 2026-09-13 after the results and thresholds were presented.
Recheck these thresholds on later integrated workloads.

| Large workload       | Legacy ms | Prototype ms | Legacy peak MiB | Prototype peak MiB |
| -------------------- | --------- | ------------ | --------------- | ------------------ |
| Construct table      |    10.516 |        2.474 |           2.312 |              6.878 |
| Select column        |     0.945 |        0.463 |           2.491 |              0.861 |
| Weighted fit         |    22.311 |       12.705 |          17.656 |             15.360 |
| Append 25 rows       |    49.196 |       60.397 |          15.297 |             13.762 |
| Convert table        |     9.238 |        1.209 |           9.161 |              6.868 |
| Insert stack frame   |   235.265 |       21.348 |          28.766 |             41.830 |
| Reduce stack         |     8.169 |       10.733 |           4.067 |             16.065 |
| Convert stack        |     3.471 |        2.106 |          16.008 |             12.002 |
| Edit frame pixel     |     0.604 |        0.671 |           0.008 |              1.507 |

Table input is 100,000 x 8 float64; stack input is 16 x 512 x 512 uint16.
Peaks are traced allocations during the operation, not total process RSS.
Construction is a narrower prototype than full legacy Data initialisation.
Loading the real CoreTest fixture plus the conversion bridge took 34.502 ms
versus 35.282 ms legacy; it is not a general loader-performance claim.
