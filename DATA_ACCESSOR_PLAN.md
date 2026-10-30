# Data accessor roadmap

Date: 2026-10-29. Target checkout: `devel`.
Status: fitting accessor implemented; other groups remain future-work candidates.

## Purpose and agreed direction

Group specialised operations behind properties bound to a `Data` owner, following
the approach used by `Data.plot` and `DataPlotter`. Improve discovery without
changing numerical behaviour, masks, metadata, column roles or storage ownership.
Keep the ordinary table interface directly on `Data`.

The selected first group is fitting. Its implementation, registry, compatibility,
documentation and test plan is in [FIT_ACCESSOR_PLAN.md](FIT_ACCESSOR_PLAN.md).
The other groups below are candidates, not commitments to implement or remove APIs.

Use [STRING_ENUM_PLAN.md](STRING_ENUM_PLAN.md) for string enums covering closed
public choices such as fitting outputs and text formats. Coordinate enum adoption
with each accessor migration while preserving accepted string values.

## Source-backed candidate inventory

The review inspected the live class, its dynamically attached functions and
inherited methods. At review time it exposed 91 public callable names, including
inherited container methods, and 18 properties. These counts are a snapshot,
not a test requirement. The working tree also contains the preceding move of
`annotate_fit` from `Data` to `Data.plot`; preserve that work.

| Proposed accessor | Existing methods | Assessment and boundary |
| --- | --- | --- |
| `fit` | `curve_fit`, `lmfit`, `odr`, `differential_evolution`, `polyfit` | Selected next. Shared model handling, fit metadata and numerical selection. `polyfit` already has a deprecation note; preserve its current behaviour without making it a default. |
| `signal` | `SG_Filter`, `smooth`, `bin`, `make_bins`, `interpolate`, `extrapolate`, `spline`, `peaks`, `find_peaks`, `threshold` | Strong future candidate for sampled-signal processing, resampling and feature extraction. A later design may separate resampling from feature finding if that improves the user workflow. |
| `stats` | `mean`, `std`, `min`, `max`, `span` | Coherent small group of scientific reductions. Retain role defaults, bounds, uncertainties and current return types, including value/index or value/error tuples. |
| `calc` | `add`, `subtract`, `multiply`, `divide`, `diffsum` | Coherent column arithmetic with uncertainty propagation. Keep distinct from table operators and ordinary structural editing. |
| `clean` | `del_nan`, `find_duplicates`, `remove_duplicates`, `deduplicate`, `outlier_detection` | Possible later group. First document differences in approximate versus exact duplicates, averaging, masking, mutation, cloning and returned arrays/dictionaries. These operations are not interchangeable. |
| `transform` or an alignment group | `integrate`, `normalise`, `decompose`, `apply`, `scale`, `stitch` | Exploratory only. Calculus, normalisation and inter-dataset alignment may be too broad for one accessor; leave directly on `Data` until a coherent boundary is agreed. |

Relevant implementation sources:

- `Stoner/core/data.py`: `class_modifier` sources and the `plot` property.
- `Stoner/analysis/fitting/functions.py`: the five fitting methods.
- `Stoner/analysis/columns.py`: statistics and column arithmetic.
- `Stoner/analysis/filtering.py`: a mixture of signal processing, cleaning and alignment.
- `Stoner/analysis/features.py`: `peaks` and `find_peaks`.
- `Stoner/analysis/functions.py`: transformations, `threshold` and `stitch`.
- `Stoner/core/methods.py`: duplicates, selection and structural table operations.
- `Stoner/plot/functions.py`: independently callable numerical `griddata`.

## Boundaries that must not follow source files blindly

- `filter()` sets row exclusions; it is not a signal-smoothing algorithm. Consider
  it alongside selection/cleaning rather than moving all of `filtering.py` together.
- `scale()` aligns datasets and `stitch()` joins them; neither is a fitting
  algorithm merely because its implementation can use optimisation.
- `griddata()` interpolates numerical fields without drawing. It could join a
  future resampling group, but does not belong on `plot`.
- `count()` implements the container-facing observation count. Keep its direct
  interface even if a statistics accessor eventually offers an additional entry.
- Metadata formatting and plot annotation have different ownership:
  `Data.format()` formats recorded values; `Data.plot.annotate_fit()` draws or
  returns fit-annotation text. The fit accessor must not absorb rendering.

## Interfaces to retain directly on Data

Keep column/row access, selection, iteration, insertion/deletion, renaming,
sorting, splitting, `head`, `tail`, cloning and operators on `Data`. Retain
`load`, `save`, interchange/conversion methods and explicit editing contexts.
`metadata` and `setas` already provide related object interfaces; avoid redundant
accessors for them. Do not introduce an unrestricted pandas attribute proxy.

## Rules for every future accessor

1. Accessor lookup is cheap and side-effect free. Read the current owner on each
   operation; do not snapshot measurements or create a second storage owner.
2. Preserve column selection, masks, uncertainties, mutation rules and conditional
   return types. Specify signature changes explicitly: fitting will use `x`/`y`
   coordinate selectors consistently with plotting, translating to the existing
   implementations' `xcol`/`ycol` parameters.
3. Methods returning the same `Data` continue doing so. Chaining across groups
   therefore repeats the accessor, for example `d.signal.smooth(...).signal.bin(...)`.
4. Prefer explicit declared methods with useful signatures. Dynamic lookup is
   appropriate for a defined registry, not arbitrary backend attributes.
5. Review accessor names against existing attributes and attribute-based column
   lookup: a property such as `fit` takes precedence over a column with that name.
   Explicit column selection must continue working for those headers.
6. Include internal consumers, subclass overrides, unbound method calls, folder
   bulk operations, documentation and scripts in each migration audit.
7. Specify completion, help, copy/pickle and folder behaviour alongside ordinary
   calls. An accessor is part of the public API, not just a namespace shortcut.
8. Decide compatibility aliases and removal timing separately for each group.
   Do not infer the plotting migration's compatibility policy for all other APIs.

## Future work checklist

- [x] Migrate fitting consumers in the package, folders, documentation, samples
  and scripts; validation evidence is recorded in `FIT_ACCESSOR_PLAN.md`.
- [ ] Agree fitting compatibility warning timing and removal policy before adding
  deprecation wrappers or removing the direct methods.
- [ ] Review `signal` ownership and naming after fitting has passed its exit gates.
- [ ] Decide whether `stats` and `calc` justify separate migrations.
- [ ] Compare cleaning methods' actual semantics before designing `clean`.
- [ ] Revisit transformation/alignment and numerical gridding only with a clear
  workflow and migration proposal.

Do not combine these later candidates into the fit implementation. Maintain the
active plans as handovers; record completed validation in commits/PRs and link
evidence rather than accumulating separate completed phase reports.
