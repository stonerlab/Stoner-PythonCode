# Data storage migration plan

Agreed direction: 2026-09-13. Target branch: `devel`.
Status: Stage 6 in progress; public ImageFile and ImageStack now own validated xarray storage.
Target release: undecided (`0.12` or `1.0`).
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
- [ ] **7. Validate migration and prepare compatibility release.** Run appropriate
  full-suite, platform, dependency, documentation and installed-package checks;
  repeat representative benchmarks. Reconcile dependency specifications, including
  xarray and tested pandas bounds. Set Python 3.12 as the minimum throughout
  packaging, environments, CI and documentation, removing Python 3.11 from the new
  version's support matrix. Validate Python 3.15 when publicly released and supported
  by the required dependencies; record readiness separately if it remains pending.
  Document changed APIs and the replacement of
  custom array classes. The maintainer selected `0.12.0a1` as the migration's
  initial alpha version; the canonical package version has been updated accordingly.
  Gate: all retained contracts pass or have reviewed, documented changes; hosted
  validation is distinguished from local evidence. Publishing is a separate action.
  **Prerequisite:** Stage 6e legacy-class removal is complete, including for a test
  or prerelease build. A passing suite with the old adapters present is not enough.
- [ ] **8. Simplify and extend Stoner wrappers.** Only after storage validation,
  audit existing methods for backend delegation and select new backend methods
  worth exposing. Deliver small, independently tested wrapper batches as below.
  Gate: a predictable documented API with equivalent retained scientific behaviour.
  Include labelled row access using the underlying pandas index and `.loc`, as
  requested by the maintainer after Stage 6e. Define its public spelling, label
  versus position selection, duplicate-label behaviour, result type and guarded
  mutation semantics. Distinguish selecting rows by index label from looking up
  named columns within a row; do not recreate either through a custom array class.
  This is an API enhancement batch, not additional Stage 6 removal work.
- [ ] **9. Expose native storage through the public attributes.** The maintainer's
  target is a pandas DataFrame at `Data.data` and native xarray objects at
  `ImageFile.image` and `ImageStack.stack`, exposing the actual backend storage
  without first rendering it as a NumPy snapshot. Coordinate this design with
  Stage 8 so wrappers work directly with the intended public backend interface.
  - [ ] Specify the xarray DataArray/Dataset interface for images and stacks,
    including multichannel grids, exclusions, ragged extents and calibration.
    Expose standard backend objects rather than new custom array subclasses.
    Reassess masking before choosing this interface: distinguish missing values
    from excluded measurements, preserve recoverable raw values, and define how
    native operations propagate exclusions and how users remove them.
  - [ ] Define live access, assignment and retained-reference behaviour. Resolve
    how direct pandas/xarray edits interact with owner validation, column roles,
    masks, metadata, stable frame identities and shared crops. Native exposure
    must not silently desynchronise these structures; document any operations
    that require an explicit owner edit or replacement.
  - [ ] Migrate consumers and examples to native indexing and backend operations.
    Retain explicit `to_numpy()` for numerical consumers that need NumPy, without
    requiring conversion where a routine already accepts the native object.
    Document changes from the Stage 6 snapshot attributes and their assignment
    contracts, including the relationship between `.stack` and `.imarray`.
  - [ ] Verify public types, label/coordinate access, mutation and reference
    semantics, scientific results and performance/allocation costs. Run the full
    suite after the interface switch. Gate: native access avoids an implicit
    NumPy export and the documented ownership and scientific contracts pass.

### Follow-on wrapper policy (batch 8)

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
- **Documentation validation boundary:** public source documentation was updated,
  but the attempted cached Sphinx build could not write an autosummary source
  stub under the execution filesystem restrictions. No successful complete build
  is claimed. Repeat in a suitable isolated documentation checkout during Stage 7;
  generated API stubs and the retained plot cache were not refreshed here.
- **Next:** Stage 7 platform, dependency, installed-package, documentation and
  integrated performance validation. Wrapper expansion, including pandas-backed
  labelled row access, remains Stage 8; native `.data`/`.image`/`.stack` exposure
  is recorded as Stage 9, with interface design coordinated during Stage 8.
  Current Stage 6 snapshot contracts remain in effect until that implementation.
  Retain scientific fixtures and all 243
  cached plots. Local Stage 6 completion does not establish hosted CI or release readiness.
  The maintainer subsequently authorised committing this migration and version
  `0.12.0a1` to `devel` and pushing to origin. Publication remains a separate action.

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
