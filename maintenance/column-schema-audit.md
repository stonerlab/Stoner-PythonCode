# Column selection and schema audit

CS1-CS5, updated 2026-10-28, `devel`. This is the active caller inventory for
[the column schema plan](../STORAGE_MIGRATION_PLAN.md#column-schema-and-setas-consolidation-2026-10-27).
The tables retain the CS1 source snapshot and line numbers used to migrate CS2;
their descriptions of previous defects are historical starting points. Matching
and caller cardinality are implemented; the CS2 closure below records those
changes. CS3's composed schema and role facade were the baseline for the isolated
CS4 Index comparison; CS5 retains that composed design. Keep the contracts for
CS6 documentation and regression review, then retire implementation-only
findings into Git history and ignored run logs.

## Contract vocabulary

| Code | Contract |
| --- | --- |
| M | Resolve all matches, flatten ordered selector collections and retain repeats. Consumers may require a non-empty result. |
| O | Exactly one column per scientific input or structural reference. Explicit ambiguous selection raises `ValueError` listing the selector, positions and headers. |
| F | Deliberately take the first resolved column. Make that policy visible with `first=True` when resolving selectors. |
| I | One insertion/replacement anchor, separate from ordinary selection. `None` and the one-past-end integer mean append; named references must be unambiguous. |
| R | Resolver/facade dispatch: forward the same matching, `first` and shape policy without adding a new policy. |
| G | Derive role groups/defaults; distinguish a default first channel from an explicit ambiguous selector. |

O does not mean taking `[0]` after resolving. Validate before materialising values,
mutating data, selecting rows or creating plot axes. F is reserved for an existing
documented/conscious first policy; do not automatically add it to scalar callers.
Absent explicit selectors, keep existing first assigned/default role choices where
an operation handles one channel. Multiple assigned x groups remain meaningful;
`startx` selects the group, not all independent variables simultaneously.

Public return policy: integer selector -> integer; string -> integer for one match
or flat list for several; compiled regex/slice/collection -> flat list, including
singleton and empty collections. `force_list=True` always returns a list.
`first=True` returns the first resolved position, or `[position]` with force_list.
It validates the complete selector before truncation and raises `KeyError` on an
empty result. Missing names remain `KeyError`, invalid integer positions
`IndexError`, invalid regex fallback `re.error`, unsupported/Boolean selectors
`TypeError`. Do not coerce numeric strings into positions.

Matching: exact all -> whole-header case-sensitive glob if glob syntax exists,
otherwise literal substring all -> regex search all on failure. Compiled patterns
bypass the string stages. Each stage stops on success. Use complete glob bracket
expressions plus `*` and `?` for detection; unmatched brackets are literal in the
substring stage. `fnmatchcase` semantics avoid Windows case normalisation.

## Source inventory

Inventory all production Python source, not just the Data class body: its
`class_modifier` attaches methods from core, columns, fitting, functions, features,
filtering and plotting. `Data._col_args` delegates to
`core.numerical.column_arguments`. Core structural methods delegate to
`core.storage_operations`. Operators call those owner operations through utils.

The tables cover every enclosing function containing a `find_col`, shared/plot
`resolve_columns`, `_col_args` or `_get_cols` call at audit time, including recursive
resolver calls and nested folder helpers. Lines identify call sites, rather than
function definition lines; paths below are relative to `Stoner/`.

AST inventory totals: 75 `find_col`, 28 `_col_args`, 16 `_get_cols` and 11
`resolve_columns` calls: 130 call sites across 87 distinct enclosing functions.
These counts include resolver implementations and plot resolver dispatch, not
130 independent public methods. A separate source search checked schema access,
role projections and dynamic delegation.

### Core and owner boundaries

| File | Functions and call lines | CS2/CS3 contract and required work |
| --- | --- | --- |
| `core/methods.py` | `find_col` 173 | R: add keyword-only `first`, preserve positional `force_list`, correct stale compiled-pattern docstring. |
| `core/data.py` | `__getattr__` 530; `_setattr_col` 866 | M: dynamic column attributes/role writes may expose several columns; preserve ordinary attribute/metadata precedence and row shape. |
| `core/methods.py` | `search_index` 49-50 | O explicit x; F assigned default. Current `[0]` hides ambiguous explicit selection. A row predicate has one value and one row; no implicit any/all across x. |
| `core/methods.py` | `closest` 121 | O explicit x; F default. Flattened `argmin` over several columns is not a row index. |
| `core/methods.py` | `search` 390 | M output columns; O predicate x through `search_index`. Preserve existing 1D output for one selected output column. |
| `core/methods.py` | `find_duplicates` 197; `remove_duplicates` 250 | M x keys and per-y/error channels. `find_duplicates` currently scalarises explicit key lists despite its multi-key algorithm; remove that truncation. |
| `core/methods.py` | `section` 439 | G/F: use first assigned y/z within one selected x group; no explicit header selector on these coordinate parameters. |
| `core/methods.py` | `split` 717, 727 | O selected grouping column; F assigned default. Callback grouping remains one key per row; further positional arguments remain recursive grouping levels. |
| `core/methods.py` | `rename` 1038 | M: rename every matched column to the supplied new string, preserving IDs/roles. Existing list indexing of the header proxy fails. |
| `core/interfaces.py` | `count` 198 | M explicit columns, F default y. Preserve axis-dependent reduction shapes; value counting must count values, not coordinate-array elements (`argwhere(...).size` overcounts 2D matches). |
| `core/numerical.py` | `column_arguments` 70, 86 | R/G: preserve `scalar=True` as deliberate first conversion, not an exactly-one guard. O consumers must validate explicit inputs before this helper truncates them. Keep `scalar=False` predictor collections. |
| `core/storage_owner.py` | `_index` 404; `_selection` 486 | M: rectangular numerical selection/write and structural selection. Preserve row-versus-column indexing and fresh IDs for repeated structural selection. |
| `core/storage_bridge.py` | both `find_col` methods 52, 247 | R: bound and cloned `_ResolvedSetas` share resolution. |
| `core/storage_bridge.py` | `_storage_slice` 348; `_BoundSetas.__setitem__` 111 | M: numerical annotations and per-match role assignment; retain transaction guards. |
| `core/storage_bridge.py` | both `_get_cols` methods 61, 244 | G: common absolute uncertainty positions; remove multi-x adjustment only with equivalent shared implementation. |
| `core/storage_operations.py` | `add_column` 35 | I: named anchor O, explicit width/None append; preserve negative existing-column references. Never pass a multi-match list into `insert_columns`. |
| `core/storage_operations.py` | `del_column` 45, 60 | M normal deletion. For `duplicates=True`, resolve selected header names as a set and retain the first occurrence of each selected name; no selector targets all duplicated names. |
| `core/storage_operations.py` | `del_rows` 84 | O predicate column when val is supplied. Boolean row masks/callbacks remain separate modes. |
| `core/storage_operations.py` | `swap_column` 122 | O each endpoint of every swap; do not pair/broadcast ambiguous endpoint matches. |
| `core/storage_operations.py` | `sort` 158 | M ordered sort keys, lexicographic priority, retaining repeated keys; native numeric positional labels avoid duplicate alignment. |
| `core/storage_operations.py` | `append` 187 | M mapping-key selection, scalar value broadcast to all matching columns. Process mapping insertion order, later overlapping keys win. Keep missing columns masked and late failures atomic. Data-to-Data append retains occurrence-based header alignment, not fuzzy matching. |
| `core/storage.py` | `resolve_columns` 88, 98 | R: one implementation of matching, full validation and collection flattening; no scalar truncation until requested. |

### Setas and numerical results

| File | Functions and call lines | CS2/CS3 contract and required work |
| --- | --- | --- |
| `core/setas.py` | `find_col` 651, 655, 657 | R: replace independent legacy resolution. Currently compiled duplicate names collapse to the first occurrence repeatedly; numeric fallback and nested collection behaviour also differ from native storage. |
| `core/setas.py` | `__call__` 333, 341, 359; `_add_core_` 516, 521 | M/G: role-to-selector and selector-to-role mapping directions must both assign all matches. Header-key direction currently indexes a list with resolved lists. |
| `core/setas.py` | `__getitem__` 438 | M: header/regex selection returns one role or a list of roles under resolver shapes. Role keys (`x`, `#x`, etc.) keep their separate established meaning; do not feed them through fuzzy column lookup. |
| `core/setas.py` | `__setitem__` 481, 484 | M: scalar role broadcast or an equal-length list of roles; reject length mismatches rather than silently truncating. |
| `core/setas.py` | `_sub_core_` 541, 580 | M: unset every selected column; preserve role-letter and encoded-role operations with documented precedence over fallback parsing. |
| `core/setas.py` | `cols` 179 | G: derive without writing inferred roles or retaining a competing mutable authority. |
| `core/utils.py` | `_assemble_x_data` 66, 72, 74 | M predictor columns; preserve matrix shape and external complete observation vectors. |
| `core/utils.py` | `_assemble_col_data` 83; `_assemble_sigma` 107, 109 | O per y/z/component or per-channel sigma after outer expansion. Shared sigma may be broadcast across channels; an ambiguous selector is not an observation vector. |
| `core/utils.py` | `assemnle_data` 157, 160 | M predictors/channels; retain external arrays. Its bounds/search path currently uses first x; document that legacy predicate convention explicitly instead of discarding multivariate predictors. |

Assignment contract: resolve before mutation; one role broadcasts to all matches;
an explicit role sequence must match the expanded selection length. Preserve
ordered repeated positions and apply assignments in input order (later writes
win), including mapping overlaps. Invalid later selectors/roles roll back the
whole bound edit. Allow several x assignments because multiple x groups are part
of the scientific API. Exactly-one requirements belong to consuming operations.
CS3 implements the same atomic validation for standalone Setas while preserving
bound/detached ownership and existing callable return conventions.

### Analysis and fitting

| File | Functions and call lines | CS2/CS3 contract and required work |
| --- | --- | --- |
| `analysis/columns.py` | `_do_error_calc` 15; `__get_math_val` 72 | F simple arithmetic operand is already explicitly documented as first match; use `first=True`. Recognise explicit `(value, error)` tuples before resolving; two header matches must never accidentally become a value/error pair. Each tuple member selects one operand column by that policy. |
| `analysis/columns.py` | `add` 130; `diffsum` 183; `divide` 232; `multiply` 419; `subtract` 557, 562 | I result anchor; preserve explicit uncertainty tuples and independent result/error columns. `subtract`'s extra resolution at 562 is unused and can be removed in CS2. |
| `analysis/columns.py` | `max` 265, 267; `min` 362, 364 | O explicit column, F assigned y; keep one scalar value and one source-row index, never flattened multi-column argmin/argmax. |
| `analysis/columns.py` | `mean` 309; `std` 485 | O explicit data/sigma selector, F default y/error; external per-row sigma remains valid. Scalar output contracts do not gain multi-channel output implicitly. |
| `analysis/filtering.py` | `filter` 110; `del_nan` 147 | M flat column selections. Current list comprehensions can create nested matches; flatten once. Row callback gets an ordered selected vector; NaN deletion reduces across selected columns. |
| `analysis/filtering.py` | `SG_Filter` 211 | O x derivative coordinate, M y signals. Existing code accepts x lists but differentiation assumes one final x derivative row; validate explicit x. |
| `analysis/filtering.py` | `bin` 310, 319-322 | O x, M y with aligned errors. Validate errors as zero, shared-one or one per y; preserve count/error output columns. |
| `analysis/filtering.py` | `deduplicate` 379 | M composite keys, already supported. |
| `analysis/filtering.py` | `extrapolate` 442 | O x, M y/error channels; result insertion anchors remain I downstream. |
| `analysis/filtering.py` | `interpolate` 557 | O explicit x or F assigned default; `xcol=False` remains row-index interpolation. All numerical output columns are interpolated together. |
| `analysis/filtering.py` | `outlier_detection` 770 | O selected channel, F default y; mask/delete action and callbacks operate on one channel at a time. |
| `analysis/filtering.py` | `scale` 861, 876-877, 884-885 | O x/y in each source; a 2D affine transform is not a channel broadcast. |
| `analysis/filtering.py` | `smooth` 983 | O x, M y: existing loop supports channels but scalar helper/header handling truncates before it. Preserve per-channel headers and insertion ordering. |
| `analysis/filtering.py` | `spline` 1075 | O x/y/sigma, F defaults; one UnivariateSpline result, not a multivariate or multi-output spline. |
| `analysis/features.py` | `peaks` 71; `find_peaks` 218 | O explicit x/y, F default roles. Existing explicit y list is silently reduced to its first item. |
| `analysis/functions.py` | `apply` 68 | I selected destination; scalar row-function result replaces/inserts one column, array results use the existing multi-output branch. Do not interpret several matching destination names as output shape. |
| `analysis/functions.py` | `clip` 124, 126 | O explicit predicate column, F default y. |
| `analysis/functions.py` | `decompose` 163, 166-167 | O x, M y; symmetric/antisymmetric output anchors I. |
| `analysis/functions.py` | `integrate` 263, 267-268, 287 | O x, M y (existing channel loop); fix scalarisation and channel orientation. Result destination I; metadata result naming must not index headers with several matches. |
| `analysis/functions.py` | `normalise` 344, 348, 351 | M targets (documented), O base, F default target. Current `scalar=True` drops additional explicit targets. |
| `analysis/functions.py` | `stitch` 446 | O x/y from both datasets, F defaults; single pair of curves. |
| `analysis/functions.py` | `threshold` 586, 613, 632 | O signal and explicit coordinate, F defaults. Multiple thresholds are values, not selected channels. `xcol=False` keeps full-row return mode. |
| `analysis/utils.py` | `split_up_down` 467; `hysteresis_correct` 591 | O explicit scan/signal axes, F assigned defaults; branch detection and loop correction operate on one trace. |
| `analysis/fitting/functions.py` | `_record_curve_fit_result` 321 | I result/residual destination; arithmetic `+1` requires one anchor before mutating the owner. |
| `analysis/fitting/functions.py` | `polyfit` 1075, 1085, 1090 | O x, M y independently fitted; resolve each y to an integer before building labels/results. |
| `analysis/fitting/selection.py` | `curve_fit_inputs` 25; `_observations` 19 | M predictors and y channels; O per-channel observation/sigma after expansion. Keep complete external arrays, positive scalar sigma and shared/per-channel error policy. `bounds` currently selects rows using first x; retain that explicit predicate convention while `where` can inspect the full frame. |

Do not turn multiple independent variables in general `curve_fit`/legacy fitting
assembly into exactly-one x just because line plots and `polyfit` require it.
Do not infer uncertainty tuple syntax from the number of resolved name matches.

### Plotting, folders and image conversion

| File | Functions and call lines | CS2/CS3 contract and required work |
| --- | --- | --- |
| `plot/selection.py` | `_positions` 40; `resolve_columns` 73 | M y, O x/z/u/v/w. Existing guard checks cardinality before indexing x `[0]`; this is already correct, not a hidden first-x defect. G/defaults and absolute group uncertainty positions stay shared. |
| `plot/selection.py` | `_uncertainty` 111 | O per-row uncertainty selector; symmetric/asymmetric arrays and scalar uncertainties retain existing modes. |
| `plot/statistics.py` | `_row_values` 35; `prepare_series` 63, 71 | O weights/reduced values, M sample channels with independent masks; G y fallback may use all numeric columns. |
| `plot/fields.py` | `draw_matrix` 351 | O x column; y integer is a coordinate row, not a column selector. Preserve external coordinate vectors. |
| `plot/functions.py` | `griddata` 65 | O coordinate axes, M scalar fields (z in 2D, u in 3D); F assigned defaults. Do not reduce supported multiple dependent fields to one. |
| `plot/accessor.py` | `_projection_for` 275; `__call__` 378, 393 | R: use plot resolver and keep validation before axes allocation; explicit matrix/stats modes retain their own contracts. |
| `folders/functions.py` | `gather._gatherer` 123, 145 | O common x and xerr, M dependent/error channels per file. Initial `cols['xcol'][0]` incorrectly assumes list-shaped default x; resolve exactly-one consistently on each member. |
| `Image/core.py` | `_init_from_datafile` 545 | O x/y/z intensity for a 2D ImageFile; F assigned defaults. Conversion should not silently pick an explicit ambiguous intensity. |

No format module directly calls these resolver functions. Loader role assignment
nevertheless flows through Setas: `formats/data/instruments.py` uses exact
`H_vsm (T)`/`m (emu)` headers and `formats/data/rigs.py` uses `T (C)`/`R (Ohm)`.
Positional/encoded roles in other loaders remain unchanged. The instrument saver
at `instruments.py:85` explicitly takes its first assigned y; keep that F policy.
Run real loader fixtures when CS2 changes parsing, not synthetic headers alone.

Additional indirect consumers: `folders/each.py` forwards Setas selection/edits;
`folders/functions.py::concatenate` uses x-role sorting; `core/data.py` role access,
`core/property.py` axes, `core/operators.py::__invert__` role-derived column order,
`core/numerical.py::numerical_result` first-role convenience annotations and
`tools/widgets.py` y-role defaults must continue using the common derived groups.
These do not introduce another fuzzy resolver. Public `column`, `//`, tuple
selection and assignment reach the owner resolver through `_storage_slice`/`_index`.

## Schema ownership inventory

All production table `.schema`/`._schema` accesses found by source search are in
`core/storage.py`, `storage_owner.py`, `storage_bridge.py` and
`storage_operations.py`. ImageStorage has its separate frame schema and is outside
this consolidation. The CS3 state-publication audit additionally found that
`analysis/fitting/functions.py::curve_fit` installed an exported package directly
as owner state. It now uses the canonical owner publication boundary; a search for
schema fields alone did not identify that escape in CS1.

- Storage validation, copy and pandas/package exports use ordered records and
  displayed-header/UUID projections. Keep version-one package labels stable.
- Owner adopt/export/deepcopy, `_SchemaField`, IDs, replace, selection, row/column
  insertion and numerical edit commits retain one authority and atomic mask/value
  updates. Repeated selection creates fresh IDs for additional occurrences.
- Bridge preview copying and detached numerical annotations derive from the
  owner. Bound Setas reads/writes no longer construct snapshots; the only retained
  `_ResolvedSetas` name is an alias for reading earlier pickles.
- Operations append aligns duplicate headers by occurrence; column concatenation
  remaps colliding IDs. These structural rules must survive ColumnSchema extraction.

## Tests and validation

`tests/Stoner/core/test_column_cardinality.py` adds executable contracts for all
six resolver entry points (shared schema, Data, bound Setas, cloned Setas,
annotated numerical results and standalone Setas),
existing positional shapes, regex flags, precedence examples, explicit first
selection, plot cardinality, multivariate fit preparation with independent masks,
statistical/structural selection, role assignment rollback and insertion anchors.

All 50 CS1 expected-failure cases are active CS2 regressions. They cover all-match
strings, new precedence, `first` and full-selector validation, standalone duplicate
regex identity, reverse mapping role assignment, ambiguous search predicates,
duplicate rename/mapping append and composite duplicate keys. Existing tests for
search shapes, duplicate sorting, unique inverse shapes, column assignment and
plotting were updated for all-match names. One CS1 fixture expectation was corrected:
`Sig`/`Sig*` selects Sigma headers as well as Signal headers.

`tests/Stoner/core/test_column_matching.py` adds 36 cases for cheap-path dispatch,
case sensitivity, literal brackets, exactly-one guards before mutation, role writes
with duplicate names, mapping overlap/rollback, mask-preserving multi-target
normalisation, multi-channel smoothing/integration and a real instrument fixture.
Existing `test_storage_contracts.py`, `test_storage_structural.py`, `test_setas.py`,
analysis/fitting and plot tests remain the broader scientific baseline.

Reproduce the inventory from the repository root with `rg` on `Stoner`, including
definitions/doc references, and AST call extraction to identify enclosing functions:

```powershell
rg -n '\bfind_col\b|\bresolve_columns\b|\b_col_args\b|\b_get_cols\b' Stoner --glob '*.py'
rg -n '\._state\.schema|\._schema\b|\.schema\b' Stoner/core Stoner/analysis Stoner/plot Stoner/folders Stoner/formats --glob '*.py'
```

Validation: verified Miniforge py314 (Python 3.14.7, pandas 3.0.5, NumPy 2.5.2,
pytest 9.1.1), satisfying all 39 `tests/test-env.yml` Conda specifications.
Full local run: **1,573 passed**, with no failures, errors, skips or expected
failures. JUnit: ignored `maintenance/runs/cs2-full/pytest.xml`. After private-helper
import cleanup, the final affected suite passed **1,101 tests** with no failures,
errors, skips or expected failures (`maintenance/runs/cs2-final/pytest.xml`). No hosted CI
or lower-pandas-bound pass is claimed. The available py312 environment has pandas
3.0.5 and lacks pytest-timeout, so it is not used as minimum-dependency evidence.

Fresh cached documentation passes the unchanged warning baseline: 62 reviewed
warnings, no unexpected warnings and no missing dynamic Data methods. The older
HTML's `Data.title` entry corresponds to title control now exposed through
`data.plot.title(...)`. The old attribute was already absent at pre-CS2 HEAD,
verified using an isolated HEAD-package import; the ignored API comparison records
that earlier plotting API migration. Evidence:
`maintenance/runs/cs2-docs-clean/verified-audit.json`.
Tracked plot caches are unchanged. Ruff F821 and the CRLF-aware whitespace check
pass. CS3's final validation is recorded below; the Index comparison is next, CS4.

Reproduce the final affected run from the repository root:

```powershell
$env:MPLBACKEND = 'Agg'
$env:QT_QPA_PLATFORM = 'offscreen'
New-Item -ItemType Directory -Force maintenance/runs/cs2-final | Out-Null
C:/ProgramData/miniforge3/Scripts/conda.exe run --no-capture-output -n py314 python -m pytest tests/Stoner/core tests/Stoner/analysis tests/Stoner/plot tests/Stoner/folders -q -o addopts= --basetemp=maintenance/runs/cs2-final/tmp -o cache_dir=maintenance/runs/cs2-final/cache --junitxml=maintenance/runs/cs2-final/pytest.xml
```

For the full run omit the path selection and add `-n 2 --dist loadfile`, using
fresh basetemp/cache/JUnit paths beneath an existing ignored run directory.
Final documentation evidence and the next bounded step are in the migration
plan's CS2 handover.

## CS2 closure and CS3 boundary

- The single resolver in `core/storage.py` owns exact/glob/substring/regex
  precedence, all-match results, full selector validation and explicit `first`.
  Bound, cloned, numerical and standalone Setas use that implementation.
- Private helpers in `core/column_selection.py` validate exactly-one arguments
  and insertion anchors. `_col_args(single=(...))` guards explicit selectors
  before scalar conversion while preserving default-role behaviour. General
  fitting keeps multi-column predictors; bounds deliberately uses the first
  predictor without truncating model inputs.
- Multi-column paths now flatten collections, retain duplicate-name occurrences,
  preserve per-channel identity/masks and handle mapping/role broadcasts. Role
  data writes use positions rather than resolving repeated role names again.
  Normalisation snapshots source channels before insertion and preserves masks;
  smoothing/integration operate on every selected dependent channel.
- Encoded Setas subtraction and role letters retain their own grammar before
  column matching, so `.` placeholders do not become all-column regex matches.
  Arithmetic retains its deliberate first-operand convention, with explicit
  uncertainty tuples parsed before selection.
- CS3 replaces `_BoundSetas` snapshots with direct schema access and centralises
  grouping. The ordinary pandas header projection remains the ownership question
  for CS4/CS5. Native pandas lookup/alignment and version-one interchange stay intact.

## CS3 composed baseline for the Index experiment

- `core/column_schema.py::ColumnSchema` is the authoritative immutable ordered
  collection of Column records. It owns field replacement, IDs, repeated selection,
  insertion/collision remapping and resizing, and delegates matching to the tested
  CS2 resolver. Public package schemas remain editable lists.
- Bound and detached Setas share pure assignment, update, overlay, subtraction
  and group derivation. Every mutator prepares its complete result before commit.
  Saved views resolve the current owner; explicit clones detach descriptions and
  preserve callable return-to-owner binding without copying Data.
- Fresh group results replace `_cols` and `_col_defaults` instance caches. All
  x-group uncertainty positions are absolute; bridge and plot corrections are
  removed. Inferred defaults never become explicit roles.
- `DataOwner._publish_storage` converts independently prepared packages to native
  headers and ColumnSchema after caller transaction checks. Fitting now goes
  through it, preventing UUID-labelled package frames/list schemas from escaping
  into native owner state. Existing field edits, selection, insertion, replacement
  and numerical edit transactions retain their value/mask/metadata boundaries.
- Forty new tests cover immutable schema operations, saved views, zero Setas
  construction/numerical export on schema reads/edits, independent clones, atomic
  errors, transaction guards, group defaults, multiple x groups, legacy class/state
  upgrades and fitting publication. Existing TypeHintedDict pickle coercion is not
  changed or claimed lossless; DataStorage remains the lossless interchange API.

Final CS3 full suite: **1,613 passed**, zero failures/errors/skips/xfails. Fresh
cached documentation: **62 reviewed warnings**, zero unexpected warnings and no
API removals compared with the clean CS2 inventory. Evidence is under ignored
`maintenance/runs/cs3-verified/` and `maintenance/runs/cs3-docs/`. All 39 test-env
specifications are satisfied in py314. Ruff F821/F401 on the extracted modules and
CRLF-aware whitespace checks pass. No lower-dependency or hosted-CI pass is claimed.


## CS4 isolated Index comparison

The [runnable experiment](column_index_prototype/README.md) owns ColumnSchema on
a real flat pandas Index and reuses the production owner algorithms. CS2 matching
and cardinality remain shared. No production import/backend change is made.

One owner field disappears; two representations of header text, external masks,
typed metadata and atomic structural publication remain. Native copy/selection
needs private propagation hooks. Rename/insertion/horizontal concat/reindex and
alignment can lose records; Series/plain export is deliberately lossy. Native
row concat can preserve the wrong left schema when roles conflict. Explicit
transformation proof is required on re-entry, and missing records are rejected
without matching duplicate labels to invent identities. Saved bound Setas views,
metadata-only buffer sharing, transaction commit/rollback and explicit combination
are protected by the same fixture checks for both ownership models.

102 prototype cases pass on pandas 2.2.3 and 3.0.5. Current environment: 848 passes
including the full core suite; lower pandas line: 276 passes including CS1-CS3
column suites. Native matrices and paired warm timings/allocations are recorded
in ignored `maintenance/runs/cs4-release-{current,lower}/`. The plan's CS4 handover
records environment overlays, source fingerprints, private hooks, performance
and compatibility boundaries. CS5 now records the decision below; production
Index use remains disabled.


## CS5 decision (2026-10-28)

Retain CS3 composition and reject custom pandas Index adoption for this
consolidation, as accepted by the maintainer. The migration plan's
[CS5 decision](../STORAGE_MIGRATION_PLAN.md#cs5-decision-2026-10-28) records each
adoption criterion against the CS4 ownership, propagation, compatibility and
performance evidence. Moving header consistency into an Index and adding private
hooks/provenance adapters does not simplify the complete owner lifecycle.

ColumnSchema remains authoritative; ordinary pandas labels are its projection.
CS2 matching/cardinality and CS3 durable views, shared role operations, external
mask/metadata publication and version-one interchange contracts remain accepted.
The prototype stays isolated as executable evidence; its native propagation and
combination policies are not production API commitments. CS6 must reconcile the
existing docs/examples, review these contracts and record focused/full local,
Sphinx and final-revision hosted validation. CS5 verified the existing CS4 report
counts only; no new runtime test or hosted pass is claimed.


### CS6 local handover (2026-10-28)

The retained composed ownership and CS2 cardinality rules are documented in the
guides/API/release notes and the runnable `doc/samples/column_schema.py` example.
It executes through the existing sample suite and covers duplicates, literal/glob/
regex selection, explicit first, scalar rejection, saved views, masked repetition,
stable/new IDs, interchange and second-group uncertainty rendering. No resolver,
role-parser, grouping or owner runtime behaviour changed in CS6.

Current full suite: 1,614 passes; the final lower-line full result, environments
and source hashes are recorded in the migration plan's CS6 handover. Sphinx keeps
62 reviewed warnings, zero unexpected warnings and no API removals. Prototype
isolation, cache retention, lint and CI/content policy checks pass. Retain this
inventory until final-revision hosted matrix/lower-line/docs/package checks are
recorded; CS6 and the parent consolidation item remain open for those gates.
