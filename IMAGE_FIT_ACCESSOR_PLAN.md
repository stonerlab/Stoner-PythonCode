# ImageFile.fit implementation and testing plan

Date: 2026-10-30. Target checkout: `devel`.
Status: design and implementation plan only; no image fitting accessor is implemented.

Implementation sequence: [image plotting](IMAGE_PLOT_ACCESSOR_PLAN.md), then
[image ROI management](IMAGE_ROI_ACCESSOR_PLAN.md), then image fitting. The
[release plan](RELEASE_0_12_PLAN.md) gives each accessor an alpha, followed by
beta API lock and RC full freeze. Follow
[AGENTS.md](AGENTS.md), [DOCSTRING_STYLE.md](DOCSTRING_STYLE.md), the existing
[Data fitting plan](FIT_ACCESSOR_PLAN.md) and [string-enum plan](STRING_ENUM_PLAN.md).
This is a separate extension after the published 0.12.0a5 work; do not alter that tag.

## 1. Purpose and scope

The first scientific workflow is background correction: fit a plane or a full
quadratic surface using pixels believed to represent a flat background, evaluate
that fitted surface across the entire image, and subtract its spatial variation
or its complete value. Features outside the reference region must not bias the fit.

Agreed direction:

- Add an owner-bound `ImageFile.fit` property, with the same callable/explicit
  algorithm/model-shortcut structure as `Data.fit`.
- Start with planes and quadratic backgrounds, Boolean and rectangular selections,
  whole-image correction, fitting reports and reproducible coefficients.
- Preserve measured source pixels, exclusions, coordinates and typed metadata by
  default. Keep fitting/evaluation separate from applying a correction; provide
  `flatten` as a convenient combined operation.
- Keep rendering and graphical interaction on the plotting side.
- Consume the reusable `image.roi` state and resolver; do not implement a separate
  fitting-only ROI handler. Interactive selection is owned by the ROI accessor
  and uses the established plotting surface.

The detailed API names/defaults below are recommended implementation contracts,
not claims that these APIs already exist or that every earlier sketch was agreed.
Resolve the listed design checkpoints in the first batch, and update this plan
before dependent production work. No additional user approval is needed for
routine implementation choices within a subsequently authorised batch.

## 2. Source-backed starting point

- `Stoner/Image/core.py`: dynamically attached image methods; `image` exposes the
  native xarray Dataset, `imarray` is a detached read-only masked array.
- `Stoner/Image/storage_owner.py`, `storage.py`, `storage_bridge.py`: authoritative
  intensity/exclusion ownership, typed metadata, transactions, shared regions and
  stack-frame handles. Numerical snapshots are not a write-back interface.
- `Stoner/core/selection.py`: exact-shape Boolean selections, labelled alignment,
  persistent exclusions, finite-value filtering and stack valid extents.
- `Stoner/Image/imagefuncs.py::level_image`: an existing correction using
  directional averages and separate horizontal/vertical polynomials. It has a
  `box`, can clip/normalise output, and records `poly_sub`; it is not a joint
  six-coefficient surface fit. Its local crop coordinates and whole-image
  evaluation need auditing before considering any equivalence or replacement.
- `Stoner/Image/widgets.py`: `RegionSelect`, `ShapeSelect`, `LineSelect`;
  `attrs.py`: mask and pixel-drawing interfaces. These have existing GUI contracts.
- `Stoner/analysis/fitting/{accessor,registry,functions}.py`: Data fitting registry,
  shortcuts, result choices and numerical wrappers. The current lmfit wrapper
  passes only `model.independent_vars[0]`; do not assume it already supports two
  separately named independent coordinates.
- `Stoner/analysis/fitting/models/__init__.py`: `make_model` registers generated
  classes, but its decorator signature does not yet declare independent variables.

No new optimiser dependency is required. lmfit supports multiple independent
variables; see its [model documentation](https://lmfit.github.io/lmfit-py/model.html)
and [2D fitting example](https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html).
Later Gaussian/rotated-peak models are useful extensions, not first-batch scope.

## 3. Common look and feel with Data

| Concept | Data | Proposed ImageFile |
| --- | --- | --- |
| Default fitting algorithm | `d.fit(model, ...)` -> lmfit | `im.fit(model, ...)` -> lmfit |
| Explicit algorithm | `d.fit.lmfit(model, ...)` | `im.fit.lmfit(model, ...)` |
| Named model | `d.fit.linear(...)` | `im.fit.plane(...)`, `im.fit.quadratic_surface(...)` |
| Temporary observations | `where=...`, True includes | `where=...`, True includes |
| Current spatial selection | No new Data selection state proposed | `im.roi`; supporting image methods use it when where is omitted |
| Coordinates | `x`/`y` select table columns | `x`/`y` select existing spatial coordinates |
| Uncertainty | `sigma`, interpreted as standard deviation | `sigma`, same interpretation |
| Outputs | `FitOutput`, existing defaults | reuse `FitOutput`, same default FIT |
| Metadata namespace | `prefix` | `prefix` |
| Plotting | `.plot`, returns Axes | `.plot`, returns Axes; see companion plan |

Uniformity should cover names and semantics without pretending that an image has
table columns or that `y` remains the dependent variable. Image intensity is the
dependent variable; both `x` and `y` are independent coordinates. Do not expose
`xcol`/`ycol`, pseudo-columns, `setas`, or `z` as a competing intensity selector.

Lookup, `dir`, `help` and registry discovery must not read intensities, construct
models, fit, create a figure, or attach a bound accessor to the owner. Each call
uses current owner state. Clone/copy/pickle and stale shared-handle rules remain
those of the owner. No unrestricted forwarding to xarray or lmfit attributes.

## 4. Proposed first user workflow

The following is future API documentation, not an executable example yet:

```python
import numpy as np
from Stoner.enums import FitOutput

# True includes pixels in the reference/background region.
reference_region = np.zeros(image.shape, dtype=bool)
reference_region[:20, :] = True
reference_region[-20:, :] = True

report = image.fit.plane(where=reference_region, output=FitOutput.REPORT)
background = image.fit.evaluate(report)   # independent full-size ImageFile
corrected = image.fit.flatten(report)    # independent float image, zero background

# Same operation as a single fit-and-apply call; box uses pixel crop conventions.
corrected = image.fit.flatten(
    model="quadratic_surface", box=(10, 80, 20, 100),
)

# Alternatively reuse a current ROI; clearing it does not invalidate a saved fit.
image.roi.from_mask(reference_region)
report = image.fit.plane(output=FitOutput.REPORT)
image.roi.clear()
corrected = image.fit.flatten(report)
```

Recommended explicit method shape:

```text
image.fit.lmfit(model, x=None, y=None, p0=None, sigma=None, *,
               where=None, box=None, coordinates=CoordinateMode.NATIVE,
               output=FitOutput.FIT, prefix=None, **fit_options)
image.fit.evaluate(report)  # evaluate on the current matching owner grid
image.fit.flatten(model="plane", *, where=None, box=None, p0=None, sigma=None,
                  coordinates=CoordinateMode.NATIVE,
                  reference=BackgroundReference.ZERO, inplace=False, **fit_options)
```

`CoordinateMode` is implemented by the plotting milestone in `Stoner.enums`;
`BackgroundReference` remains a proposed string enum. Both should accept their
documented strings and reject wrong-domain enums. The proposed fitting defaults
must be tested before implementation. `flatten` accepts a previous REPORT in its `model`
position, as in `flatten(report)`; it must then reject fitting-only options rather
than silently ignoring or refitting them. `flatten` always returns an ImageFile;
`output` belongs to the fit methods, not this combined correction operation.

Start with lmfit; add `curve_fit` only after the common preparation and 2D calling
contract are proven. Do not advertise ODR, differential evolution or arbitrary
polynomial degree just because Data has those methods. Unsupported methods should
raise AttributeError, not silently choose another algorithm. Retain lmfit's own
`method` option; it is not an accessor algorithm switch.

## 5. Surfaces, registry and model construction

Use two explicit registered classes, provisionally in a new
`Stoner/analysis/fitting/models/surfaces.py`:

| Shortcut | Surface | Coefficient order |
| --- | --- | --- |
| `plane` | a + b*x + c*y | offset, slope_x, slope_y |
| `quadratic_surface` | a + b*x + c*y + d*x**2 + e*x*y + f*y**2 | offset, slope_x, slope_y, quadratic_x, cross_xy, quadratic_y |

The quadratic cross-term is required. Do not approximate it by two directional
fits, or overload the existing 1D `quadratic` registry key. Both built-ins are
linear in their coefficients even though one is quadratic in coordinates.
Provide initial guesses from a stable weighted linear solution; retain native
lmfit parameter constraints and reports for explicit fitting calls.

Extend the existing registry/decorators rather than create a second global model
catalogue. Proposed decorator extension:

```python
@make_model(independent_vars=("x", "y"))
def tilted_plane(x, y, offset, slope_x, slope_y):
    return offset + slope_x*x + slope_y*y
```

- Preserve bare/named `make_model`, guessers, hinters, replacement and metadata
  prefixes. The guesser must receive both selected coordinate arrays.
- Record optional independent-variable/capability metadata without constructing
  classes or calling factories during registration/discovery. Validate unknown
  model/factory dimensions when invoked. Do not infer a second coordinate is a
  fit parameter merely because it has a numeric default.
- Preserve the global read-only registry mapping. Decide and document contextual
  shortcut discovery/validation for Data versus images, without removing existing
  Data entries or changing their behaviour. Incompatible shortcuts must fail
  clearly; explicit multidimensional table curve fitting must remain supported.
- Audit names such as `evaluate` and `flatten` before reserving accessor members.
  Existing user registrations must not be silently replaced or removed.

## 6. Observation selection and coordinates

Use the ROI accessor's shared resolver and one fitting eligibility path; do not
create an independent region implementation. Do not obtain fit samples by
mutating/cropping the owner and attempting to restore it afterwards.

1. Validate the live owner and capture a consistent detached fitting snapshot.
2. Resolve the coordinate mode and spatial coordinates on that snapshot.
3. Resolve the current/explicit ROI through the common ImageROI selection adapter,
   then the existing Boolean selection/alignment machinery. `where=None` uses
   the current ROI (or all if inactive); explicit `where` overrides that state,
   and `where=ROI.all()` explicitly requests the whole domain.
4. Intersect it with an optional box, persistent exclusions, valid frame extents,
   finite intensities and the selected coordinates. True includes; image-mask
   True excludes. Store their distinction explicitly.
5. Validate uncertainties on participating pixels. Flatten coordinates, values
   and standard deviations through the same index map for the optimiser only.

First fitting release selection forms: reusable ROI descriptions, exact-shape
Boolean ndarray or exactly aligned xarray DataArray, plus the already implemented
`image.roi` current selection. `box=(xmin, xmax, ymin, ymax)` follows the existing
crop parser's edge conventions. Validate values and alignment before optimisation.
No broadcasting, silent reindexing, nearest-coordinate matching, numeric masks
or implicit GUI interaction. Combining `where` and `box` means intersection.
Empty/all-excluded selections fail before publishing metadata. The user can
combine several background patches with ROI composition or Boolean masks.
Interactive selection belongs to `image.roi`; fitting itself never opens a
selector. Callable region helpers require a shared ROI contract, not a permissive
fitting-only `where` API.

Coordinate rules shared with plotting:

- Native mode uses stored spatial coordinates; ordinary uncalibrated images have
  positional axes. Metadata such as `MicronsPerPixel` must not silently override
  conflicting explicit xarray coordinates. A legacy calibration adapter, if
  offered, must be explicit and shared by fitting and plotting.
- Pixel mode is an explicit alternative. `box` remains an index-space selection
  regardless of fitting units; use a Boolean coordinate comparison for a physical
  region. Never guess units from whether a number is an int or float.
- Image array order is `(y, x)`; model variables are `(x, y)`. Preserve non-square
  shapes, axis direction and offsets. A crop retains its existing coordinate
  labels; never re-zero native coordinates merely because the fit uses a box.
- First batch supports real finite separable coordinate vectors, including
  non-uniform/decreasing axes. Optional `x`/`y` names select existing compatible
  coordinate variables, not raw arrays. Reject unsupported 2D coordinate maps
  explicitly; retain such calibration unchanged in owner packages. Map support
  requires a later explicit fitting/plotting batch.
- An ImageFile that is itself a region/frame is the complete owner for this call:
  'whole image' means that handle's domain, not the unseen parent. To correct the
  parent, call its accessor with `where`/`box`. Never expand a region implicitly.

## 7. Numerical validity, weighting and reporting

- Require full-rank design matrices for the free surface coefficients: rank 3
  for an unconstrained plane and rank 6 for an unconstrained quadratic. Account
  explicitly for fixed parameters/constraints. A line of pixels cannot determine
  an unconstrained plane, and enough pixels alone do not prove geometric
  identifiability. Require positive residual degrees of freedom for
  reported estimated covariance; reject insufficient selections with clear errors.
- Accept `sigma=None`, a positive scalar, an exact-shape array or an exactly
  aligned DataArray. `None` means unweighted; supplied values are standard
  deviations, with weights 1/sigma. Invalid participating uncertainties fail;
  irrelevant values outside the selected domain do not remove good samples.
- Match Data's `absolute_sigma`/`scale_covar` relationship and validate conflicting
  options. State whether covariance uses absolute noise or residual scaling.
- Centre/scale coordinates internally where needed for stable polynomial fitting.
  Return coefficients/covariance in the documented original coordinate basis,
  applying the full covariance transformation including correlations. Retain
  preprocessing for evaluation and replay. Do not merely relabel normalised
  coefficients. A numerical spike must settle how this stays coherent with the
  native lmfit report before production implementation.
- Preserve parameter names, covariance ordering, units, fit success/status, sample
  count, rank/conditioning diagnostics and degrees of freedom. Surface height,
  gradients and curvature have corresponding intensity/coordinate units.
- No silent robust fitting, outlier rejection or clipping. Those change the
  scientific estimate and need later explicit controls/tests.

Reuse `FitOutput.FIT` as parameters/covariance, `REPORT` as native lmfit ModelResult,
`ROW` as interleaved parameters/errors plus the documented statistic, `DICT` as
named values/errors/statistics, `FULL` with a precisely documented tuple, and
`DATA` as the same image owner with fit metadata. No FIT/DATA call flattens pixels.
Enum/string parity and the exact matrix must be documented before exposing all
six modes. Do not invent a second default report return for model shortcuts.

Recommended REPORT extension: an image-specific immutable fitting context carried
with the native ModelResult, rather than a parallel report wrapper. It records
coordinate conventions, selection/eligibility, preprocessing, uncertainty policy
and grid signature, with no live owner/Matplotlib references. `params`, `eval` and
`fit_report()` retain lmfit behaviour. Settle the context attachment/public name
in batch 1; test report copying/pickling and the FULL/DATA representation.

Successful ordinary fits may record metadata as Data.fit does; failures publish
nothing. Evaluation/flattening must use the report's captured model state rather
than a newly registered replacement, a mutable owner's latest fit, or 'last fit'
magic. By default, reject a report from an incompatible coordinate grid. General
evaluation on a separately supplied grid is a later explicit API.

## 8. Whole-image correction and reference level

Let B(x,y) be the fitted surface and R the requested constant reference level:

`corrected(x,y) = measured(x,y) - B(x,y) + R`.

- `BackgroundReference.ZERO` (recommended default): R=0; remove the entire model.
- `BackgroundReference.REGION_MEAN`: R is the weighted mean of the fitted B on
  the eligible reference pixels, captured from that fit. Remove variation while
  preserving that defined background level. Use inverse-variance weights when
  uncertainties were supplied; otherwise equal weights.
- A finite numeric reference sets R explicitly in intensity units.

Do not conflate region-mean reference with the whole-image mean, the origin
intercept or an unweighted mean after masked values have been filled. Other
reference-point conventions can be added later as explicit choices.

Evaluate B on the entire matching grid, including pixels outside the reference
region. State that this extrapolates beyond sampled background patches; record
the sampled spatial span. Do not automatically suppress the correction outside
that span. Preserve foreground features by excluding them from parameter
estimation, not by leaving their background uncorrected.

`evaluate(report)` returns an independent floating-point background ImageFile
with source geometry/calibration and exclusions. Correction applies to all finite
measured pixels, including persistently excluded raw values, while retaining their
exclusion flags. Non-finite original values remain non-finite; no mask is cleared.
Correction must not replace measured pixels with fitted intensities.

`flatten` defaults to an independent float64 image, with typed metadata copied;
retain negative and above-display-range values. Do not call `asfloat`, divide
integer intensities by their dtype range, clip or normalise the scientific result.
Set dtype-dependent fill metadata deliberately. The original owner, its metadata
and any parent/shared frame must remain unchanged for the default operation.

An explicit `inplace=True` is a later gated option: initially accept only writable
floating owners with compatible dtype/geometry. Reject integer/bool/complex input
or unsupported shared owners before mutation; do not wrap/truncate values. Commit
pixels and correction provenance atomically through the owner, respecting existing
transaction rules. Do not promise stack-wide atomicity for a folder operation.
Existing numerical edit contexts do not make pixels and arbitrary metadata one
transaction. Prepare and validate all provenance before committing, and design
and test a failure-safe commit protocol before enabling in-place correction;
otherwise keep that option deferred. Do not claim existing owner APIs already
provide the combined guarantee.

Record model/key/version, ordered coefficients/covariance, coordinate names/units,
selection and eligibility, weights, reference level, evaluated grid and source
provenance. Use a versioned serialisable record; export a mask or an explicitly
documented compact representation, not only its hash. Preserve unrelated metadata
and provide stable prefixes for repeated fits/corrections. Reusing a saved report
after source edits is explicit behaviour: it applies the captured surface, not a
silent refit; verify the matching grid and retain that distinction in provenance.

## 9. Implementation structure and bounded batches

Recommended modules: `Stoner/Image/fitting.py` for ImageFitter; shared coordinate
preparation established by the earlier plot/ROI batches under
`Stoner/Image/coordinates.py`; fit preparation/context/result
helpers in a small image fitting support module; surface models in the existing
fitting-model package. Keep numerical work separate from the accessor and owner
commit logic. Reuse enum/registry/parameter-formatting services without copying
DataFitter or forcing images through temporary Data objects as their new owner.

| Batch | Work | Exit gate |
| --- | --- | --- |
| 1 | After plot/ROI alphas, finalise fit/reference/output contracts, native-report context and conditioning spike | Shared ROI/coordinate foundation available; no ambiguous defaults or 1D registry regressions |
| 2 | Shared coordinate/selection preparation, surface classes, registry/decorator multidimensional support | Synthetic plane/quadratic recovery, masks, rank and calibration tests; full existing registry/decorator tests |
| 3 | ImageFitter/property, lmfit callable and shortcuts, outputs and metadata | Explicit/shortcut parity, discovery/copy/pickle, failures without publication, Data.fit regressions |
| 4 | Full-grid evaluation and independent flattening; gate explicit in-place support separately | Features preserved, reference modes correct, exclusions/dtype/coordinates/atomicity verified |
| 5 | Documentation and sample workflows; optional folder integration | Runnable examples, cached Sphinx warning/inventory audit, appropriate full suites and visual verification |

`ImageFolder.each.fit` is not already image-aware: current FitProxy signatures
and dispatch assume DataFitter, and the worker's registration path calls its
shortcut machinery. Audit folder/ImageStack `each` paths separately. If included,
use capability-based owner dispatch, per-operation registration capture and the
existing serial/thread/process policy. Correction returns independent images by
default; explicitly define folder replacement/collection. Test ragged frame valid
extents, error tuples, ordering and worker cleanup. Defer this extension rather
than advertise partial support in the first release.

## 10. Implementation and testing checklist

- [ ] Exact/noisy plane recovery; full quadratic including a non-zero x*y term.
- [ ] Foreground island excluded from a fit: correct slopes/curvature, entire
  image background removed, feature shape/amplitude preserved outside the ROI.
- [ ] Offset rectangular ROI proves global evaluation does not re-zero coordinates;
  disjoint patches, holes, border-only reference, one-pixel axes and empty selection.
- [ ] Too few samples, rank deficiency, near-singular geometry, large coordinate
  offsets/small spacing; compare predictions and transformed covariance to an
  independent weighted design-matrix calculation with scale-aware tolerances.
- [ ] Non-square grids; unequal pixel sizes; non-uniform, decreasing and translated
  native axes; pixel/native differences; unsupported map/mismatched coordinate errors.
- [ ] Boolean ndarray and labelled xarray alignment, order/dims/shape/coordinate
  mismatch rejection, no broadcasting/reindexing; mask True vs where True polarity.
- [ ] Current ROI, explicit override, ROI.all and clear precedence; fitting snapshots
  the actual selection; evaluate/flatten still cover the whole grid after ROI changes.
- [ ] Excluded/non-finite values, no observations, padded frames, numeric/nullable
  selectors; prove source pixels, persistent mask and owner generation do not change
  for fitting-only and failed operations.
- [ ] Scalar/array/labelled sigma, invalid participating sigma, absolute/relative
  covariance scaling, parameter constraints, prefixes, fixed parameters and output
  enum/string parity. Reject unsupported/wrong-domain choices before optimisation.
- [ ] Known coefficients, uncertainty/covariance ordering and units, independent
  report replay after registry replacement, changed owner pixels and incompatible grids.
- [ ] Zero/region-mean/numeric reference modes; correction beyond ROI; corrected
  excluded raw values retain exclusions; NaNs/infinities remain non-finite.
- [ ] Signed/unsigned/bool/float input retains raw intensity units; float output
  contains negative values; no implicit clipping, normalisation or dtype overflow.
- [ ] Default corrections/backgrounds do not alias source/parent/frame buffers or
  nested typed metadata. Explicit in-place success/failure and transaction conflicts
  follow the specified owner rules, with no partial pixels/metadata publication.
- [ ] Accessor/signature/dir/help lookup is inert; factories are fresh; clone/copy/
  pickle do not capture bound accessors, live owners, locks, GUI objects or workers.
- [ ] Preserve all existing Data fitting/registry/output and legacy `level_image`
  tests. Do not assert equivalence between the two levelling algorithms.

## 11. Documentation, compatibility and completion

Add a beginner guide with a reproducible sloping background plus a foreground
feature, a selected reference region and before/background/corrected panels. Add a
quadratic example with a cross-term, calibrated axes and disjoint patches. Show
explicit `FitOutput.REPORT`, coefficients, uncertainty interpretation, reference
choices, extrapolation and the distinction between fitting and correction.
Document that API sketches in this plan are unavailable until implemented.

Keep `level_image` and its current options/returns during the first addition.
Do not replace it with the new joint solver under its old name, change clipping
defaults, or start deprecation warnings incidentally. Audit callers and select a
separate migration/removal policy if desired. Update image docs, samples, model
catalogue, API inventory and release notes; add cached figures deliberately and
retain them for `READTHEDOCS=True`. Use the plotting companion only once available;
early tests/examples may use existing `imshow` plus explicit Matplotlib axes.

Run focused image/selection/registry/fitting tests first, then the appropriate full
suite in a verified supported environment. Separate network/OCR/manual GUI evidence.
Build cached Sphinx against the reviewed platform warning/API baselines and serve
generated HTML alone over localhost for inspection. Require hosted exact-commit
matrix/lower-dependency/package/docs gates before a later release. Scientific
benchmarks/fixtures complement synthetic recovery; neither certifies all image
formats, facility data or physical instrument calibrations.

Next fitting step, after the plot and ROI milestones: batch 1 contract/numerical
spike, followed by surface-model and fitting eligibility preparation using the
shared ROI resolver. No production implementation or deprecation is part of this
planning change.
