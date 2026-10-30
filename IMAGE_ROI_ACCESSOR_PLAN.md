# ImageFile.roi implementation and testing plan

Date: 2026-10-30. Target checkout: `devel`.
Status: plan only; no ROI accessor is implemented by this change.

Implementation order: [ImageFile.plot](IMAGE_PLOT_ACCESSOR_PLAN.md), then
`ImageFile.roi`, then [ImageFile.fit](IMAGE_FIT_ACCESSOR_PLAN.md). This order is
the agreed image feature sequence for the next stable-release candidate scope;
see the release checkpoint in [DATA_ACCESSOR_PLAN.md](DATA_ACCESSOR_PLAN.md).

Follow [AGENTS.md](AGENTS.md), [DOCSTRING_STYLE.md](DOCSTRING_STYLE.md), the shared
[selection implementation](Stoner/core/selection.py) and
[string-enum plan](STRING_ENUM_PLAN.md). Detailed names/defaults below are
recommended contracts to settle in the first ROI batch, not implemented APIs.

## 1. Purpose and separation of concerns

Provide one reusable ROI facility for fitting, reductions, histogram analysis,
selection outlines, cropping and later scoped processing. Avoid independent box,
polygon and Boolean-mask interpretations in each consumer.

The public `image.roi` accessor manages the current region of interest for that
ImageFile handle. It uses immutable reusable ROI descriptions and a shared
geometry-to-inclusion resolver. Neither component owns image intensities.

Three concepts must stay distinct:

| Concept | Meaning | Mutated by setting/clearing ROI? |
| --- | --- | --- |
| Current ROI | Which spatial region a supporting operation considers | Yes |
| Persistent image exclusions | Pixels excluded from scientific use; mask True excludes | No |
| Eligible observations | ROI AND non-excluded AND valid values/coordinates/extents, with consumer-specific uncertainty rules | Calculated per operation |

ROI membership is independent of whether measured intensity is finite. This
allows the same ROI to be displayed, saved and applied to several measurements.
The consumer adds its scientific eligibility rules. Clearing the ROI is not
unmasking, restoring pixels, undoing a crop or undoing a background correction.

## 2. Source-backed reuse inventory

- `Stoner/Image/widgets.py::RegionSelect`: rectangular interactive selection,
  currently displays via legacy `imshow` and returns pixel bounds.
- `ShapeSelect`: interactive geometric selection, including existing finish,
  cancellation and inversion behaviour; audit the returned mask's polarity.
- `LineSelect`: profile endpoints, relevant to later line/strip ROIs but not a
  replacement for every area-selection tool.
- `Stoner/Image/attrs.py::MaskProxy.select`, `.draw`, `.threshold`, `.invert` and
  `DrawProxy`/`storage_bridge.py::OwnerDraw`: geometry/rasterisation and GUI code,
  often coupled to exclusion mutation. Reuse the geometry without exercising
  the old mask-changing side effects.
- `Stoner/Image/numerical.py::box`, `ImageFile._box`, crop/region ownership and
  image-folder mask selection: existing bounds and bulk selection consumers.
- `Stoner/core/selection.py::resolve_selection`, `selected_image_values`: exact
  Boolean shape/alignment checks and separate eligibility combination. Extend a
  deliberately typed boundary rather than add arbitrary callback/eval dispatch.

Inventory signatures, return types, coordinate conventions and GUI ownership
before extracting helpers. Do not replace existing selection code wholesale or
route ROI creation through a temporary mutation of `image.mask`.

## 3. Proposed public interface

Future API sketches, not executable against the current package:

```python
# Set current ROI by replacing the previous selection.
image.roi.box(10, 80, 20, 100)
image.roi.polygon([(10, 20), (80, 20), (45, 100)])
image.roi.circle(centre=(45, 60), radius=20)
image.roi.from_mask(included_pixels)  # True includes, unlike image.mask

# Interactive selection must be explicit.
image.roi.select_rectangle()
image.roi.select_polygon()

saved_region = image.roi.current     # immutable description, or None
membership = image.roi.resolve()     # detached labelled Boolean inclusion array
image.roi.clear()                   # no active ROI; existing exclusions unchanged
```

Recommended return conventions: ROI-setting methods, combinations and `clear()`
return the ImageFile owner, enabling `image.roi.box(...).plot()` or later
`image.roi.box(...).fit.plane(...)`. `current`, `active` and `resolve` inspect
state/membership. Do not return a mask from some setters and an owner from others.
Avoid `bool(image.roi)` as a test for membership: inactive and active-but-empty
are different states; use `active` and resolved sample counts.

For reusable construction and compound regions, introduce a public immutable
`ROI` description type, provisionally exported from `Stoner`:

```python
from Stoner import ROI
from Stoner.enums import CoordinateMode

left = ROI.box(0, 20, 0, 100, coordinates=CoordinateMode.PIXEL)
right = ROI.box(80, 100, 0, 100, coordinates=CoordinateMode.PIXEL)
background = left | right
image.roi.set(background)
image.roi.add(ROI.circle(centre=(50, 50), radius=5))
image.roi.intersect(usable_area)
image.roi.subtract(foreground_region)

report = image.fit.plane(output="report")  # future: use current ROI
whole_image_report = image.fit.plane(where=ROI.all(), output="report")
```

`ROI` constructors create descriptions without changing an image; accessor
constructors set the current selection. The distinction and return types must
be prominent in help/examples. Use explicit `add`/`intersect`/`subtract` methods
rather than another unchecked mode-string namespace. Direct expression operators
are union `|`, intersection `&`, difference `-` and complement `~`.

First batch: boxes, exact Boolean masks, immutable descriptions, current-state
management, clear/all and composition. Next batch: polygons/circles/ellipses and
adapters to existing interactive selectors. Lines/strips, intensity predicates,
connected components and labelled regions are later constructors; do not
advertise every skimage draw function as an ROI method.

## 4. Default operation selection and overrides

Once a method is explicitly made ROI-aware, use this shared policy:

1. `where=None`: use the current ROI; with no active ROI, include the whole domain.
2. Explicit `where=roi`, Boolean ndarray or aligned DataArray: use that temporary
   selection instead of the current ROI. It does not update current state.
3. `where=ROI.all()`: explicitly bypass the current ROI, while still respecting
   persistent exclusions and the consumer's validity checks.
4. An operation's additional `box` convenience intersects the selected `where`
   result. A box is not a separate source of hidden selection state.

Keep core `resolve_selection(None, target)` meaning all observations. Resolve
the owner-specific current ROI in an ImageFile adapter before calling the shared
Boolean/labelled validator. Do not make a low-level resolver inspect global or
unrelated owner state, or silently change Data.where behaviour.

Roll this policy out only to reviewed consumers, with compatibility tests and
release notes. A current ROI must not implicitly affect every existing image
method or change its output shape. Unsupported operations do not start slicing
their inputs behind the caller's back.

| Consumer | Intended current-ROI behaviour after its integration batch |
| --- | --- |
| ImageFile.fit methods | Parameter estimation uses current ROI by default |
| fit.evaluate / fit.flatten application | Evaluate/correct the entire owner grid; ROI limits estimation only |
| mean and other reviewed reductions | Use current ROI when where is omitted; explicit where overrides |
| plot.hist | Count selected eligible intensities by default |
| plot image/mesh/contour | Display whole image by default; show an optional ROI outline, not an implicit crop |
| plot/select preview | Display whole context so ROI creation is not trapped inside the old selection |
| crop | Existing explicit box behaviour unchanged; a dedicated ROI crop helper is separate |
| masks, filters, registration, saving | Existing semantics until deliberately integrated; no blanket ROI forwarding |

Drawing the current ROI belongs on `image.plot.roi(...)`, with explicit Axes and
Axes return. Interactive `image.roi.select_*` owns selection lifecycle and may
use that renderer. Setting a programmatic ROI does not create or show a figure.
Optional `image.roi.crop()` starts with a rectangular ROI and an independent
coordinate-preserving result. Non-rectangular extraction must define treatment
of holes/background before inclusion; do not silently crop only its bounding box.

## 5. Coordinate and membership contract

Share the coordinate resolver and proposed `CoordinateMode` string enum
introduced by ImagePlotter. Do not create an image-fitting-only ROI coordinate
system. Fitting native coordinates and display orientation do not change which
physical/pixel locations a region describes.

- Programmatic geometric constructors default to pixel coordinates, explicitly
  documented. Pixel `(x,y)` means `(column,row)`; Boolean arrays have `(y,x)` order.
  Use native mode explicitly for physical coordinates; do not infer units from
  Python int/float types.
- Define new pixel boxes with lower-inclusive/upper-exclusive bounds, so
  `ROI.box(10,80,20,100)` includes the corresponding integer-index crop. Bounds
  outside the image intersect its domain, never wrap negative indices. Validate
  invalid/reversed/NaN bounds separately from an empty intersection.
- Keep the old crop parser's fractional/negative/metadata-box options as explicit
  compatibility input conversion. Do not let new ROI constructors inherit those
  ambiguous conventions by accident. A fitting `box` convenience can call the
  old parser once and convert its resolved bounds into an explicit ROI.
- Specify membership at pixel centres. Circle/ellipse/polygon edge inclusion must
  be documented and tested against reused rasterisers; include polygon/circle
  boundaries deliberately. Do not assume every shape has the box's half-open rule.
- Native geometry retains coordinate names and units. When built against a
  reference, capture its calibration description without holding the owner.
  Reuse across images with compatible units is valid; reject incompatible or
  ambiguous units rather than silently convert. Settle the constructor/reference
  signature in the first batch, including unitless/native axes.
- Native boxes/circles on anisotropic/decreasing/non-uniform axes are evaluated
  using coordinate values. A physical circle generally is not a circle in index
  space. Preserve global coordinate offsets for cropped/region handles.
- Mask descriptions are bound to a captured grid/label signature. Reject shape,
  dimension, coordinate-order or label mismatches; no broadcasting/reindexing.
  Geometry descriptions can resolve on compatible different grids. Mixing
  coordinate spaces requires an explicit conversion using a reference grid.
- An all-region/complement resolves against the current valid geometry. Scientific
  exclusions, non-finite intensities and uncertainty validity remain additional
  consumer rules. Padded stack pixels are never eligible observations.

The first phase uses separable grids established by ImagePlotter. Unsupported
2D coordinate maps fail explicitly until a shared later geometry batch supports
them. Descriptions and resolved membership must not silently drop calibration.

## 6. State, lifetime and ownership

Keep current ROI as operational state on the ImageFile handle, outside persistent
exclusions and typed scientific metadata. An accessor is transient and reads the
current state; store no GUI object, bound accessor or writable membership view.
Setting/clearing current state is atomic; failed construction leaves it unchanged.

Recommended lifetime rules to finalise and test:

- No active ROI is distinct from an explicitly empty ROI or `ROI.all()`.
- Saved accessors follow current owner state; saved immutable descriptions retain
  their own geometry. `resolve` returns an independent Boolean DataArray.
- Value-only edits preserve ROI. Clone/deepcopy preserve a valid independent
  selection; no mutable state is shared. Pickle can retain serialisable
  descriptions, never renderer/callback state.
- ImageStorage v1 is not silently extended with operational ROI state. Ordinary
  save/load does not promise to restore it; offer explicit versioned ROI
  serialisation if persistence is needed. Fit/correction provenance captures
  the actual selection independently of later `clear()` or state changes.
- Audit crop, transpose, rotations, flips, resampling and owner replacement.
  Transform/restrict/rebase ROI explicitly when the coordinate mapping is known.
  Otherwise mark it stale and require clear/reselection; do not silently use a
  wrong selection or fall back to the whole image. Never invalidate for a mere
  intensity edit when geometry is unchanged.
- Parent, cropped handle and stack-frame handles have separate active selection
  state, even when pixels/masks are shared. No state leaks across siblings or
  frames. 'Whole image' means the current handle's domain, not its unseen parent.
- Resolve once per operation. Folder/worker support captures a serialisable
  selected description/membership in order, not a live proxy or a global ROI.

Interactive selection commits only on acceptance. Escape/cancel/closed-window
leaves the old ROI unchanged and cleans up owned callbacks/artists/temporary
figures. Legacy ShapeSelect cancellation/inversion semantics must be adapted
explicitly. Preserve caller-owned Axes/Figures and restore pyplot state. Existing
mask selectors retain their old mutation behaviour until separately migrated.

## 7. Implementation structure and sequence

Recommended split: pure descriptions/composition/resolution in a small shared ROI
module, provisionally `Stoner/core/roi.py`; owner-bound ImageROI management and
widget adapters in `Stoner/Image/roi.py`; shared coordinate preparation from the
plotting batch; a narrow extension/adaptation of `core/selection.py`.

Keep the description/resolution layer free of Matplotlib/Qt and image-storage
ownership imports. Use explicit target adapters for image references. This makes
the handler reusable across image consumers and later other spatial datasets,
without promising a Data.roi accessor or replacing existing table selectors.
Avoid generic getattr/eval-based dispatch or a new mutable image-storage layer.

| Batch | Work | Exit gate |
| --- | --- | --- |
| R1 | Audit existing selection code; freeze state/defaults/coordinates/returns | Reviewed contracts and API sketches; image plotting coordinate foundation already complete |
| R2 | Immutable boxes/masks/all/composition, ImageROI property/state/clear | Membership/alignment/state/lifetime tests; old masks and Data selection unchanged |
| R3 | Shape rasterisation and interactive adapters using existing code | Boundary/orientation tests; accept/cancel/cleanup; explicit manual GUI limits |
| R4 | Integrate reviewed reductions, plot.hist and ROI outlines | Explicit/current/all precedence, no pixel/exclusion mutation; consumer parity tests |
| R5 | Documentation, replay/serialisation and optional rectangular extraction | Reusable examples, Sphinx/API/visual checks; ImageFitter may now consume the common resolver |

ImageFolder/ImageStack-wide current ROI is a separate design, not automatic
propagation of one frame's state. Do not copy a shape-specific mask onto ragged or
calibrated members without validation. General processing-within-ROI needs its
own definition of output/edge effects and is deferred beyond selection support.

## 8. Testing and documentation checklist

- [ ] Lookup/current/active are inert; setters/clear return the owner; no stored
  accessors, numerical copies, figures or masks created merely by property lookup.
- [ ] Pixel/native boxes, centres, axes order, half-open bounds, shape boundaries,
  outside/empty regions, invalid arguments, no negative wrapping and exact crop
  compatibility conversion. Test non-square/offset/anisotropic/decreasing grids.
- [ ] Native unit/grid compatibility; mask alignment; dtype/nullable/numeric mask
  rejection; full/empty/complement and immutable compound operations.
- [ ] Reuse rectangle/polygon/circle descriptions across compatible images; combine
  disjoint patches and holes; mutable caller arrays cannot modify saved state.
- [ ] Default-current, explicit override, explicit all, box intersection and clear
  precedence; empty current ROI remains empty and does not become full-image.
- [ ] ROI membership remains distinct from exclusions/finite-value eligibility;
  selectors leave pixels, dtype, masks, typed metadata and calibration untouched.
- [ ] Clone/deepcopy/pickle, value edits, crop/flip/transpose/shape changes, stale
  geometry and parent/region/frame isolation follow the declared lifecycle table.
- [ ] Interactive accept/cancel/Escape/close, mask-polarity adapters, supplied Axes,
  callback teardown and repeated use. Event simulation plus explicit manual checks.
- [ ] Reductions/histograms/plot outlines agree on membership; masks still affect
  eligibility; displaying a region never implicitly crops or modifies it.
- [ ] Existing mask.draw/select, RegionSelect/ShapeSelect, crop, image mean/stack
  selection and Data.where tests remain valid except documented opt-in active-ROI
  integration behaviour. Existing calls without active ROI retain their results.

Document setters versus pure builders, True-includes polarity, clear versus
unmask, current versus explicit/all selection, multiple patches, units, geometry
changes and cancellation. Include fitting, histogram and reduction examples that
reuse one ROI, with a whole-image display showing its outline. Keep GUI examples
separate from unattended/headless examples. Add serialisable selection replay and
later ROI provenance to fit docs without duplicating region implementation.

Run focused selection/image/plot/ownership tests, followed by affected/full suites
in a verified supported environment. Build/audit cached Sphinx/API documentation
and inspect generated HTML over localhost. Record manual widget/backend limits
and exact-commit hosted release evidence; do not claim future APIs were tested by
structural validation of this plan.

Next bounded ROI step, after ImagePlotter: R1 inventory/contract review and pure
description/current-state implementation. ImageFitter must use this facility,
not introduce its own independent region selector.
