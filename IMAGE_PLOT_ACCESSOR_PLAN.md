# ImageFile.plot implementation and testing plan

Date: 2026-10-30. Target checkout: `devel`.
Status: standalone ImagePlotter published in 0.12.0a6; ROI/fit integration remains future work.

This is the first image milestone, followed by [ImageFile.roi](IMAGE_ROI_ACCESSOR_PLAN.md)
and then [ImageFile.fit](IMAGE_FIT_ACCESSOR_PLAN.md). Follow the shared API and
coordinate contracts and [release phase policy](RELEASE_0_12_PLAN.md). Follow [AGENTS.md](AGENTS.md),
[DOCSTRING_STYLE.md](DOCSTRING_STYLE.md), [DATA_ACCESSOR_PLAN.md](DATA_ACCESSOR_PLAN.md)
and [STRING_ENUM_PLAN.md](STRING_ENUM_PLAN.md). Do not alter published release tags.

## 1. Intent and source-backed scope

Give ImageFile and Data a common plotting interface: callable `.plot`, named plot
types, explicit Axes targets, the same style/layout contexts, callable property
forwarding and deliberate display/saving. Keep image-specific observation and
coordinate preparation separate from table column-role resolution.

Current sources to preserve/audit:

- `Stoner/plot/accessor.py::DataPlotter`, `properties.py`, `state.py`, `fields.py`
  and `statistics.py`: Axes-return drawing, contexts, live-target tracking and
  Axes-first/Figure-second getter/setter proxies.
- `Stoner/Image/core.py`, `imagefuncs.py::imshow` and `plot_histogram`: existing
  image rendering and Figure returns. `imshow` can create a new figure even when
  an Axes is supplied, overlays the mask, records axes/figure in metadata and
  installs a Qt clipboard callback when available. These are legacy behaviours to
  review, not requirements to copy into a new accessor.
- `Stoner/Image/imagefuncs.py::hist`, `profile_line`, `radial_profile`: numerical
  analysis returning Data objects. Do not move them merely because their results
  can be plotted. `draw` changes pixels and `mask` changes exclusions; both remain
  directly on ImageFile.
- `Stoner/Image/widgets.py`, `attrs.py`, `folders.py`, `stack.py` and `kerr.py`:
  selectors and display consumers that may expect a Figure or current pyplot state.
- Image owner/storage/coordinate modules: native Dataset and calibrated region/frame
  contracts. The numerical array alone does not describe physical geometry.

The first release adds display, contour/mesh, histogram, shared contexts and
property behaviour. It establishes coordinate services for the following ROI
milestone and does not depend on a complete ROI/fitting implementation. Interactive
selection belongs to ImageROI; plotting later supplies outlines/previews.
Surface rendering and collection conveniences can be later batches. It does not rename processing methods or add
unrestricted Matplotlib/xarray attribute forwarding to ImageFile itself.

## 2. Recommended public interface

The standalone calls below are implemented; later ROI/fit sketches remain future APIs:

```python
ax = image.plot()                      # same as image.plot.image()
ax = image.plot.image(cmap="viridis")  # canonical name also present on Data.plot
ax = image.plot.imshow(cmap="gray")    # documented image-display alias
ax = image.plot.contour(levels=12, ax=ax)
ax = image.plot.pcolormesh()
ax = image.plot.hist(bins=128)          # same name as Data.plot.hist

image.plot.xlabel = "Position / micrometres"
image.plot.title("Background-corrected image")
image.plot.show()

with image.plot.style("stoner"):
    figure = image.plot.new_figure(figsize=(9, 3))
    with image.plot.MultiPanel((1, 3), figure=figure) as axes:
        image.plot(ax=axes[0])
        background.plot(ax=axes[1])
        corrected.plot(ax=axes[2])
```

Use `.image` and `.hist` as canonical names, matching Data. Do not introduce
`.histogram` as the primary name merely because an earlier sketch used it. The
`imshow` alias gives existing image users an obvious transition.

Recommended image field signature:

```text
image.plot(x=None, y=None, *, ax=None, cmap="gray",
           coordinates=CoordinateMode.NATIVE, **image_options)
```

`x`/`y` select existing spatial coordinates, consistently with image fitting;
intensity is supplied by the owner. They are not Data column indices or raw
coordinate arrays. Each named field method documents its own valid options.
Do not introduce an ambiguous `z` selector, implicit figure option, or unrestricted
`kind` string dispatcher just to resemble unrelated legacy interfaces.

All drawing methods return their primary Matplotlib Axes. Multiple artists,
colour bars or overlaid masks do not change that return type. Users obtain
artists through native Axes methods/collections and save through `ax.figure` or
the shared `SavedFigure` context. Do not create a special image-only artist return
or `showfig` toggle. Rendering neither shows nor saves unless explicitly requested.

## 3. Shared accessor behaviour

| Operation | Contract for Data.plot and ImageFile.plot |
| --- | --- |
| Lookup | Owner-bound, cheap; no numerical read, figure creation, or display |
| Draw target | Explicit live `ax` wins; otherwise use current compatible Matplotlib target, creating only when drawing requires it |
| Return | Primary Axes |
| Target inspection | `.ax`, `.figure`, `.axes` inspect recent live targets; no creation or revival |
| Properties | Recent live Axes first, then Figure; assignment invokes native setter; reads give live callable value proxies |
| Native signatures | Preserve proxy `.get(...)`/`.set(...)` and callable native getter/setter arguments |
| Precedence | Real accessor members win; unknown/private names raise AttributeError |
| Style/layout | Same direct stonerplots exports and upstream signatures; same `.style(...)` behaviour |
| Display/save | Explicit `.show()`, native Figure saving or `SavedFigure` |
| Copy/pickle | Do not serialise render targets, callbacks, bound accessors or mutable artist state with scientific owners |

Reuse `PlotState`, `PlotProperty` and context/formatter exports through small
owner-neutral helpers, extracted only where needed. Do not subclass DataPlotter
and inherit table methods requiring `setas`, columns or table selections. Avoid a
large generic plotting framework or duplicated property/state implementations.
Use public interfaces or a deliberately reviewed shared helper, not cross-imports
of private table-only methods throughout image code.

Keep target state outside persistent scientific metadata, as on Data.plot.
An accessor obtained twice should resolve the same owner's latest live target,
without caching image snapshots. Closed figures are not revived. State lookup
must not change pyplot's current figure. Shared contexts must restore caller state
after normal exit and exceptions. Explicit-target draws must not create spare
figures or redirect to the last image drawn by another owner.

Implemented decision: ImagePlotter has no `.labels` list; Data's column labels do
not describe images. Coordinate metadata supplies image axis defaults, with
explicit `xlabel`/`ylabel`/`title` overrides and native Figure colour bars.

## 4. Coordinates, rendering and exclusions

Establish a shared coordinate resolver and `CoordinateMode` for the subsequent
ROI/fitting plans; never implement separate calibration precedence or
int-versus-float heuristics in each accessor.

- Native mode honours the stored x/y coordinates and units; uncalibrated axes are
  positional. Pixel mode explicitly requests indices. Any legacy metadata-only
  calibration conversion must be explicit and shared with fitting.
- Preserve array order `(y, x)`, spatial offsets, decreasing axes, crops and
  valid frame extents. Rendering a region does not imply plotting its entire parent.
- Recommend retaining the familiar image `origin="upper"` default. Treat this as
  an image orientation convention, not permission to relabel rows. Ensure image,
  mesh and contour methods place identical measured values at identical displayed
  coordinates; overlays must not flip vertically. Freeze exact origin/extent rules
  with corner-marker tests in the first batch.
- For a uniform separable grid, derive image extent from pixel-centre coordinates
  and edges; handle half-pixel offsets, unequal spacing, single-pixel axes and
  decreasing axes explicitly. A rendering `extent` override is display-only and
  must not modify stored calibration or change subsequent fitting units.
- Do not represent a non-uniform grid as an affine `imshow` extent. Use a mesh
  path for callable/default `.image` display, or fail with an actionable
  recommendation when an explicitly requested `imshow` cannot represent it.
  Document dispatch and test edge construction. Never resample scientific pixels
  silently for display. General 2D coordinate maps need an explicit later mesh
  batch; reject unsupported selected maps instead of substituting indices.
- Combine persistent exclusions and non-finite values for rendering without
  rewriting intensities or their mask. Make excluded pixels transparent by default;
  an optional coloured mask overlay has its own colour/alpha controls. Verify mask
  geometry uses the identical origin, extent and transform as the image.
- Masked/invalid/padded values must not set automatic colour limits. Handle
  constant/all-excluded images deliberately. Rendering may use Matplotlib's colour
  normalisation but must not normalise or clip the stored measurement.
- Respect supplied colormap/Norm objects and explicit limits. Sharing one Norm
  across raw/background/corrected panels should work without copying scientific
  values or changing caller-owned colour-map global state.

Histogram drawing uses eligible intensity samples and documented bin/density/range
options, with the same Axes return as Data.plot.hist. It must not accidentally take
a histogram of the counts returned by `image.hist()`. Preserve numerical
`image.hist()`, `profile_line()` and `radial_profile()`; their Data results can use
the existing Data.plot accessor. An eventual image profile convenience must have
an explicit name and extraction/coordinate contract.

The first plotting alpha uses explicit Boolean/labelled `where` inputs where
supported; omitted selection includes all eligible intensities. When ImageROI is
added, integrate its current/explicit/all selection policy into `.hist` and add
`.plot.roi(...)` outlines. Image/contour/mesh display remains full-domain by default;
a current ROI must not silently crop the display. Keep this later integration
explicit in release notes and tests; numerical Data.where semantics are unchanged.

## 5. Fitting workflow integration

The first ImageFitter workflow should be inspectable with standard plotting:

```python
# Proposed combined workflow; available only after both accessors are implemented.
from Stoner.enums import FitOutput

report = image.fit.quadratic_surface(where=reference_region, output=FitOutput.REPORT)
background = image.fit.evaluate(report)
corrected = image.fit.flatten(report)

with image.plot.MultiPanel((1, 3)) as axes:
    image.plot(ax=axes[0])
    background.plot(ax=axes[1])
    corrected.plot(ax=axes[2])
```

Start with displaying independent model/corrected ImageFiles on shared coordinate
axes. Contour overlays can come from `background.plot.contour(ax=ax)`; they do
not require a plotting helper to rerun a fit or discover a mutable 'last report'.
Correction extrapolates outside the reference region; show the selected patches
in the documentation. Residuals and corrected images must not be conflated when
the requested correction preserves a non-zero reference intensity.

An eventual `image.plot.annotate_fit(report, ...)` should reuse scientific parameter
formatting and annotation ownership conventions, support text-only formatting,
and use a recent live/explicit target without creating a figure. It is a later
batch: first define how report/provenance enters shared formatting rather than
copying Data's table-metadata assumptions. Selection outlines should similarly be
explicit plotting operations driven by shared ROI descriptions and never change
the persistent mask. Do not build an independent plotting-only ROI selector.

## 6. Compatibility and consumer migration

Keep legacy `image.imshow()` and its Figure return in the first accessor release.
Do not alias it directly to an Axes-return method. Retain its existing options,
metadata behaviour and callbacks until a deliberate compatibility migration.
If the implementation shares a new renderer, the adapter must preserve and test
all legacy behaviour; improving unwanted legacy side effects is a separate change.

Audit tracked sources before migration: package cross-calls, notebooks if tracked,
image documentation, samples, scripts, tests, widgets, Kerr helpers, image folders,
stacks, repr/display hooks, explicit `_` adaptation controls and string dispatch.
Generated/historical content and existing plot-cache files remain intact.

Classify each caller by what it needs. An old caller using the Figure should use
`ax.figure` after migration; a selector expecting a Figure/callback lifecycle may
remain on legacy `imshow` until its dedicated batch. Ordinary drawing callers can
move to `image.plot()`/`.image()`. Do not use a global text substitution across
these different return contracts.

Audit `plot_histogram` separately: it plots histogram plus cumulative distribution
and currently has its own return behaviour. It is not simply an alias for `.hist`.
If retained as a convenience, specify that extra curve and its Axes return on a
new named method; keep the legacy call working during the compatibility period.
Keep selectors, raster `draw` and numerical analysis directly on ImageFile.

Do not announce deprecation/removal timing without a separate decision. Document
both APIs during migration, distinguishing legacy Figure and new Axes returns.

## 7. ImageFolder, ImageStack and interactive use

Do not infer support from the presence of the existing `folder.each.plot` proxy.
Its dispatch and signatures import DataPlotter's supported kinds. Audit the actual
ImageFolder/ImageStack paths, ownership and materialisation behaviour first.

If added in a later batch, resolve each member's own plotting accessor without
loading members during lookup. Rendering remains serial, unlike fitting. Preserve
member order, explicit Axes targeting, exception collection and saved context
state. Do not pickle Figure/Axes/GUI callbacks into process workers. Retain frame
valid extents and per-frame calibration; do not plot ragged padding as data.

Existing interactive selectors depend on Figure events and pixel-coordinate
assumptions. Adapt them in the following ImageROI milestone, after the new
target/coordinate contract is proven.
No interactive selector is invoked by accessor lookup or an omitted fitting ROI.
Mouse/display/clipboard checks must be separated from headless numerical tests;
CI event simulation alone does not establish live GUI behaviour on every backend.

## 8. Bounded implementation batches

Recommended modules: `Stoner/Image/plotting.py` for ImagePlotter, the shared image
coordinate helper from the fitting plan, and small owner-neutral plotting helpers
under `Stoner/plot`. The ImageFile property is cheap and uses a local import where
necessary. Add explicit public exports without circular imports or class_modifier
overwriting the property. Source method inventories and collisions precede edits.

| Batch | Work | Exit gate |
| --- | --- | --- |
| 1 | Freeze naming, Axes/Figure compatibility, coordinate/origin/mask contracts and rendering inventory | Shared API comparison, caller classification and executable test specifications |
| 2 | Extract/reuse minimal target/property/context services; add ImagePlotter/property and uniform image rendering | Existing Data.plot suite unchanged; inert lookup, target state and Axes returns verified |
| 3 | Non-uniform mesh/contour and histogram; coordinate/exclusion handling | Spatial alignment and histogram counts independently checked; no source mutation |
| 4 | Standalone image examples and selected consumer migration; publish plot alpha | Figure-dependent callers preserved; docs/API examples and hosted release gates pass |
| 5 | During later ROI/fit milestones, add ROI outlines/current-selection histograms and fit panels/annotations as scoped | Shared resolver/lifecycle/provenance tests; no duplicate ROI implementation |

The agreed milestone order is plot, ROI, fit. Supporting design work may overlap,
but publish a useful standalone plotting alpha before the ROI and fitting alphas.
Do not require every DataPlotter method or fit-panel example for that first alpha,
and do not bundle Data signal/statistics accessors into this work.

## 9. Testing and documentation checklist

- [ ] Accessor lookup, dir/help/signatures, known and unknown property lookup, copy/
  clone/pickle are inert: no numerical read, figure creation or GUI interaction.
- [ ] Callable, `.image` and `.imshow` alias parity; every drawing return is Axes;
  explicit `ax` creates no spare Figure. Native artist arrays reflect source values.
- [ ] Latest live target, owner isolation, multiple figures, closed targets,
  Axes-first/Figure-second property forwarding, setters/getters and complex native
  `.get`/`.set` signatures match Data.plot. No copied/stale property values.
- [ ] Real accessor members win; missing/private names raise AttributeError without
  reviving targets. Do not reintroduce broad top-level ImageFile pseudo-properties.
- [ ] Shared styles/formatters/context exports, nested contexts, explicit Figure
  ownership and exception cleanup; pyplot state restored as documented.
- [ ] Uniform/non-square/unequal-spacing and decreasing axes; single-pixel axes;
  offset crop/frame coordinates; selected calibration, pixel/native modes and
  unsupported map handling. Verify corner-marker artist coordinates, not only images.
- [ ] Non-uniform mesh edges and contours align with image values and fitted surface;
  no interpolation concealed by `extent`. Explicit display extent does not affect fits.
- [ ] Masked/non-finite/all-excluded/constant images, valid frame extents, colour
  limits, shared Norm/cmap, and mask overlays at identical extents/orientation.
- [ ] Hist counts/bin edges match an independent eligible-pixel NumPy calculation.
  Source dtype, raw values, exclusions, coordinates and typed metadata are unchanged.
- [ ] Legacy `imshow` still returns Figure and honours tested options/callbacks;
  repr and selectors remain functional. Avoid accidental Figure leaks or unintended
  Qt dependencies on the new path. Test headless startup without clipboard support.
- [ ] Existing Data.plot and installed-distribution tests pass after any shared
  helper extraction. Add image plotting probes to installed-distribution coverage.
- [ ] If collection support is added: no eager loading at lookup, serial plotting,
  order/returns/errors, explicit targets, frame geometry and lifecycle cleanup.

Document a minimal image display, native Axes styling, histogram, calibrated
contours and the raw/background/corrected workflow. Put the Data/Image comparison
near the beginner guides. Show `image.profile_line(...).plot()` for extracted
Data, and label current versus proposed APIs until implementation is released.
Add migration examples that explicitly convert Figure consumers to `ax.figure`.

Run focused image/plot/selection/owner tests, then appropriate full suites in a
verified supported environment. Use headless Matplotlib/Qt for automation; record
manual backend/widget limits separately. Build cached Sphinx HTML against the
reviewed warning and public-inventory baselines. Deliberately add new cached
figures without replacing historical assets; `READTHEDOCS=True` consumes them.
Serve generated HTML alone over localhost to inspect axes, masks, colour bars,
panels, API links and navigation. Use a small set of tolerant image comparisons
where geometry assertions cannot detect a rendering defect; avoid broad brittle
pixel snapshots. Require exact-commit hosted release gates before publication.

## 10. Implementation handover

Standalone batches 1-4 are implemented locally. `Stoner/Image/plotting.py` supplies
ImagePlotter; ImageFile.plot and the ImagePlotter export are available.
`Stoner/plot/base.py` extracts the existing owner-neutral target/state/property/
style/context services, shared with DataPlotter without copying table methods.
`Stoner/Image/coordinates.py` supplies validated separable grids, edge/label
preparation and the public `CoordinateMode` enum in `Stoner.enums`.

Callable/image display, explicit uniform-only imshow, mesh, contour and histogram
methods return Axes. Native/pixel modes, decreasing/non-uniform grids, singleton
unit widths, temporary selections, exclusions, overlay geometry and raw intensity
units are documented/tested. Image rendering keeps upper origin by default.
Legacy Figure-return imshow, numerical hist/profile extraction, widgets and
folder/stack bulk rendering remain on their existing paths. Individual cropped
or stack-frame ImageFiles can plot. No current ROI or image fitting is implemented.

Local validation used Windows Conda py314, Python 3.14.7, verified against
`tests/test-env.yml`. All 41 new image tests and existing plotting tests passed
(288 focused tests). The full suite passed 1,801 tests excluding network/OCR;
the network example passed separately. The full suite exercised the 84 offline
documentation examples, including the corrected 2x2 stonerplots example. New-code
Ruff and Python parsing checks pass. The new installed-package smoke probe was
exercised from source, not a locally built distribution.

The cached Sphinx build passed with 62 warnings and no unexpected warnings,
missing dynamic Data methods or removed inventory objects against the reviewed
Windows baseline. New ImagePlotter/CoordinateMode targets and inherited Data.plot
targets are present; the rendered guide and its API link were inspected over
localhost. Three new `image_plot_accessor` plot-cache assets were deliberately
added; existing cache files were unchanged. Logs are under ignored
`maintenance/runs/image-plot/`. Stale generated ImageFile pages were preserved
there and regenerated from the maintained template.

The authorised plotting alpha is now published as
[0.12.0a6](https://github.com/stonerlab/Stoner-PythonCode/releases/tag/v0.12.0a6)
at `3d72b7047994af54307d2234c5e77e79e837fb23`. All exact-source matrix cells passed
1,804 tests; lower dependencies passed 1,803 with one expected skip. Fresh
wheel/sdist and documentation gates passed. PyPI/Conda publication, the public
wheel's installed image plotting probes, Conda import/version and citation/DOI
were verified; evidence and the remaining Read the Docs activation follow-up are
in [MAINTENANCE_PLAN.md](MAINTENANCE_PLAN.md). This hosted evidence supersedes the
local-only validation boundary above for automated tests and installed archives.

Next bounded step: ImageROI R1 and its shared selection state/resolver. The
plotting/ROI/fitting alpha order and beta/RC phase policy remain unchanged.
Manual live GUI/clipboard behaviour and physical instrument compatibility remain
unverified. No compatibility wrappers were removed or new deprecation warnings added.
