# Data storage migration plan

Agreed direction: 2026-09-13. Target branch: `devel`.
Status: architecture agreed; implementation has not started.
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

- [ ] **1. Characterise current contracts.** Add a focused contract inventory and
  meaningful regressions for the behaviours above using the existing backend.
  Reuse current tests and real scientific fixtures. Record the verified environment,
  baseline commands and outcomes. Gate: preserved behaviours and proposed changes
  are distinguishable; no production storage changes.
- [ ] **2. Specify interchange and ownership.** Settle values/mask/schema ownership,
  duplicate-column identity, row identity, dtype policy, metadata conflict rules,
  pandas/xarray import/export and writable-view transition. Define conversion and
  editing signatures. Gate: concrete round-trip examples and an explicit API-change
  list; unresolved choices and their implications are recorded.
- [ ] **3. Prototype and measure.** Exercise an isolated composition prototype with
  duplicate columns, multi-y fitting/errors, masked integers, unequal frame sizes
  and stack-item edits. Compare accessor ergonomics. Measure time, peak memory and
  conversion costs against the current backend for loading, column operations,
  fitting, row appends, stack insertion and image reductions. Gate: correctness
  evidence and agreed performance tolerances before production replacement.
- [ ] **4. Migrate Data storage primitives.** Implement construction, schema/resolver,
  `setas`, masks, selection, direct assignment, copying and structural edits using
  the approved ownership model. Establish shared conversion/result-wrapping helpers.
  Gate: primitive contracts and interchange tests pass; any temporary compatibility
  layer has a stated removal condition.
- [ ] **5. Integrate Data consumers.** Migrate numerical analysis, fitting, plotting,
  loaders/savers, folders and dynamically attached methods in focused sub-batches.
  Preserve load identification, priorities and established Stoner load exceptions.
  Gate: retained Data APIs and real-fixture results pass, with documented intentional
  differences and updated examples.
- [ ] **6. Migrate image storage and consumers.** Start with ImageFile construction,
  masks and numerical adapters, then stack storage/extents and item ownership, then
  Kerr classes, drawing, transforms, folders and format subclasses. Gate: image and
  stack contracts, scientific results and lossless interchange pass.
- [ ] **7. Validate migration and prepare compatibility release.** Run appropriate
  full-suite, platform, dependency, documentation and installed-package checks;
  repeat representative benchmarks. Reconcile dependency specifications, including
  xarray and tested pandas bounds. Set Python 3.12 as the minimum throughout
  packaging, environments, CI and documentation, removing Python 3.11 from the new
  version's support matrix. Validate Python 3.15 when publicly released and supported
  by the required dependencies; record readiness separately if it remains pending.
  Document changed APIs and the replacement of
  custom array classes. Decide `0.12` versus `1.0` from actual compatibility impact.
  Gate: all retained contracts pass or have reviewed, documented changes; hosted
  validation is distinguished from local evidence. Publishing is a separate action.
- [ ] **8. Simplify and extend Stoner wrappers.** Only after storage validation,
  audit existing methods for backend delegation and select new backend methods
  worth exposing. Deliver small, independently tested wrapper batches as below.
  Gate: a predictable documented API with equivalent retained scientific behaviour.

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
- [Core tests](tests/Stoner/core), [image tests](tests/Stoner/image),
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

- **Completed:** architecture investigation and agreement, including the separate
  follow-on wrapper phase; repository plan created. No storage implementation or
  prototype has been undertaken.
- **Next:** batch 1, beginning with Data column resolution, `setas`, indexing and
  writable-view contracts. Inspect existing core tests, identify coverage gaps and
  record proposed API changes separately from preserved behaviour. Defer image
  characterisation to the next sub-batch if needed.
- **Validation so far:** source/test inspection and upstream documentation review
  only; no migration runtime or performance evidence yet.
- **Open decisions:** concrete mask/schema representation and interchange format;
  dtype support; writable-view transition; external backend edit reconciliation;
  alignment and metadata conflict policies; performance tolerances; release number.
- **Constraints:** retain scientific fixtures and documentation plot cache; keep
  unrelated release preparation separate; do not introduce blanket delegation.
  The new version targets Python 3.12+, with Python 3.15 support gated on public
  release, dependency readiness and project validation. Python 3.11 is out of scope.
