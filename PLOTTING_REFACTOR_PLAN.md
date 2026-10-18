# Plotting redesign

Status: stages 0 and 1 completed on 18 October 2026.
This is a separate workstream from the completed native storage
migration in [STORAGE_MIGRATION_PLAN.md](STORAGE_MIGRATION_PLAN.md).

## Objectives and agreed scope

- Keep `from Stoner import Data` sufficient for normal plotting and publication
  layouts. Keep `d.plot()` as an intelligent choice of plot and arguments.
- Make `Data.plot` a property returning a callable plotting accessor with explicit
  methods for particular plot types.
- Use stonerplots for styles, tick formatting, stacked panels, multiple panels,
  independent double axes, insets and figure saving.
- Delegate to pandas plotting where it removes useful amounts of code without
  weakening Stoner's scientific semantics. Direct Matplotlib implementation is
  equally acceptable and preferable where it is clearer.
- API breakage is acceptable for this substantially revised version. Update
  examples and documentation alongside each change. Do not build a large legacy
  compatibility layer or preserve accidental dispatch/return behaviour.
- Preserve numerical values, exclusions, metadata, column roles, column selection
  rules and storage ownership. Changing plotting must not change measurements.

The main scope is Data/table plotting and its folder consumers. Audit image APIs
for shared utilities and routing, but do not redesign image display/widgets as an
incidental part of this work.

The contracts below were accepted for stages 0 and 1. Later-stage APIs are still
planned: the stage 1 accessor exposes ordinary plotting, while field/layout
helpers remain temporarily on their legacy entry points.

## What the code currently does

Inspected checkout: `devel` at
`9db1aea5d33b39593ea52988dd751f315908efaa`, remote
`git@github.com:stonerlab/Stoner-PythonCode.git`. The working tree was clean before
this plan. Windows Git has `core.fileMode=false` and `core.autocrlf=false`.

| Source | Responsibilities that need separating |
| --- | --- |
| `Stoner/plot/core.py` | `PlotMixin`, figure state, integer axes selection, labels, return policy, template state, and broad pyplot/Axes/Figure attribute forwarding. |
| `Stoner/plot/functions.py` | Plot dispatch, column/error defaults, masked data extraction, gridding, renderers, layouts, insets, double axes, saving and display. |
| `Stoner/plot/formats.py` | Stylesheet discovery, global rcParams changes, figure construction, engineering tick formatters, labels, legends and custom template subclasses. |
| `Stoner/core/setas.py` and `core/numerical.py` | Scientific role/column inference shared with non-plotting code; reuse this policy rather than replacing it with pandas defaults. |
| `Stoner/core/data.py` | Dynamic method composition and deepcopy. The accessor must not be overwritten by `class_modifier`. |
| `Stoner/tools/classes.py` | `copy_into` currently assigns `dest.fig` and copies public plotting attributes. Update this when removing the mixin. |
| `Stoner/folders/mixins.py` and `folders/each.py` | Page layouts, explicit subplot routing, bulk calls, callable introspection and result handling. |
| `doc/UserGuide/plotfile.rst`, `doc/samples`, `tests/Stoner/plot/test_plot.py` | The existing public contract and executable examples. |

Important findings:

1. The intelligent guess is distributed between `plot`, `_fix_cols`,
   `Data._col_args` and `Setas._get_cols`; it is not just the short dispatch table.
   Unassigned two-, three- and four-column tables mean **XY, XYE and XDYE**,
   respectively. Three unassigned columns do not mean a surface.
2. `plot_xy` also infers uncertainties, labels, titles and multiple Y traces.
   `_Plot` removes excluded X/Y points and shortens uncertainties consistently.
   Metadata interpolation and custom labels need an explicit place in the new
   design.
3. `DefaultPlotStyle.new_figure` resets the style and reapplies a template even
   when using an existing axes. This can override an enclosing style context.
4. The current return policy mixes Figure, Data and None through `showfig` and
   `Options.no_figs`; image/vector helpers also return artists. Automatic plotting
   calls `plt.show()` in addition to some helpers' display calls.
5. Keyword-only explicit columns without `setas` can leave the dispatch axes count
   at zero. Three positional arguments can reach an uninitialised local variable.
   These are defects to remove, not compatibility requirements.
6. A retained Data figure takes precedence over an externally selected axes.
   Merely attaching stonerplots classes to the old mixin will not integrate them.
7. `Data.data` is now the live pandas DataFrame of **raw values**. Exclusions are
   separate. Blindly forwarding `d.data.plot` would plot excluded measurements.

## Proposed public API

`Data.plot` returns a small `DataPlotter` object bound to that Data instance.
`DataPlotter.__call__` resolves the plot type, then uses the same implementation
as its explicit methods. The getter itself creates no figure, changes no style
and takes no snapshot of the data.

```python
from Stoner import Data

d = Data("sample.txt", setas="xye")

ax = d.plot()                         # infer X, Y and uncertainty columns
ax = d.plot(x=0, y=1)                 # explicit selectors override roles
ax = d.plot.line(x=0, y=1)            # explicitly request a plain line
ax = d.plot.errorbar()                # infer uncertainties
ax = d.plot.scatter(x=0, y=1)
ax = d.plot(kind="scatter", x=0, y=1) # same implementation as .scatter

ax.set(xlabel="Field", ylabel="Signal")
ax.figure.savefig("signal.pdf")
```

Use `x`, `y`, `z`, `u`, `v`, `w`, `xerr` and `yerr` rather than perpetuating the
`xcol`/`ycol` family in the new interface. Keep Stoner's existing selector rules,
including positional indices and pattern selection, above pandas labels.

Allow at most X and Y as positional arguments to the automatic call; make `kind`,
further coordinates, format and other controls keyword arguments. This removes
the ambiguity between a third column and a Matplotlib format string. Named
surface/vector methods can have explicit coordinate signatures.

The proposed initial method families are:

| New method | Purpose and initial implementation choice |
| --- | --- |
| `plot()` | Scientific automatic dispatch. |
| `line`, `scatter` | Explicit XY primitives; direct Axes methods are the starting recommendation because selection/mask handling is already needed. |
| `errorbar`, `errorfill` | Stoner uncertainty resolution with Matplotlib or the existing shaded-error helper. |
| `surface`, `contour`, `pcolormesh`, `image` | Existing XYZ/gridding capabilities with explicit rendering choices. |
| `matrix` | Existing matrix-oriented surface capability; retain its distinct input interpretation. |
| `vector`, `vector3d`, `quiver`, `voxels` | Preserve colour-vector, glyph, 3D vector and scalar-volume capabilities. |
| `hist`, `box`, `kde`, `hexbin` | First candidates for small pandas-backed implementations, admitted after their selection and missing-data contracts are checked. |

Explicit named methods determine the rendering type. In particular, `line` is a
plain line; use the automatic call or `errorbar` for inferred uncertainties.
Unrecognised or incompatible scientific arguments must raise a useful error,
rather than disappearing in a keyword filter.

Do not add every pandas plotting method speculatively. Bar/area/pie plots can be
added when useful, with their own missing-value and input contracts. In particular,
do not accept a pandas operation that turns exclusions into zero-valued samples
without an intentional, documented policy.

### Return values, state and axes selection

- Each drawing method returns the **primary Matplotlib Axes** it drew into,
  including image, surface, vector and error-bar methods. Multiple Y traces share
  that axes. Layout contexts return their native axes/container handles.
- Obtain the Figure from `ax.figure`. `d.plot.ax` and `d.plot.figure` expose the
  most recently used axes/figure, or None before plotting. `d.plot.axes` exposes
  that figure's axes, including auxiliary axes; it is not an integer selector.
- Use `ax=` to target an axes explicitly. Otherwise follow Matplotlib's current
  axes convention, allowing unwrapped stonerplots contexts to work naturally.
  Create a new figure/axes if none exists. If an implicit current axes has the
  wrong projection, create a suitable new figure; reject an explicitly supplied
  incompatible axes rather than replacing it.
- Provide `d.plot.new_figure(**figure_kwargs)` for an independent figure without
  another import. Remove overloaded Boolean/integer `figure=` behaviour.
- Reading `d.plot.ax` must not call `gca()` or create axes. Recent-state references
  are for inspection, not an override of Matplotlib's current target. Validate
  references after a figure is closed or an inset has been replaced.
- Do not call `show()` or save as an incidental part of drawing. Notebook display
  remains Matplotlib's job. Offer an explicit `d.plot.show()` and use `SavedFigure`
  or `ax.figure.savefig` to save. Retire `showfig`; review other users of
  `Options.no_figs` before removing that option globally.

This deliberately changes per-Data figure ownership and conditional returns.
It removes the need for hidden context detection, a second layout stack, or a
custom PlotResult object. It also keeps independent figures explicit.

Use an ordinary property producing a cheap bound accessor initially, with
persistent labels/configuration and recent rendering state held once per Data.
An accessor retains its owner, so `Data(...).plot()` works. Avoid a weak-reference
only owner and avoid caching a pandas PlotAccessor or a stale frame.
Copy/clone/pickle copy configuration but detach live figure/axes state; update
both deepcopy and `copy_into` deliberately and test custom copy hooks.

### Context managers and formatting

Expose the public stonerplots objects under their existing names:

```python
with d.plot.SavedFigure("measurement.pdf", style=["stoner", "aps"]):
    with d.plot.StackVertical(2) as panels:
        d.plot(y=1, ax=panels[0])
        d.plot(y=2, ax=panels[1])

ax = d.plot.line(y=1)
with d.plot.DoubleYAxis(ax=ax) as right:
    d.plot.line(y=2, ax=right)

with d.plot.InsetPlot(ax=ax, loc="upper right") as inset:
    d.plot.scatter(x=0, y=1, ax=inset)
```

The full initial set is `SavedFigure`, `MultiPanel`, `StackVertical`, `DoubleYAxis`,
`InsetPlot`, `CentredAxes` and `PlotLabeller`. These should be direct, discoverable
exports, with the upstream signatures and yielded objects, rather than copied
implementations or another family of renamed wrappers. Explicit `ax=` in examples
makes their target clear; current-axes calls must also work.

Use stonerplots styles and formatter classes. A small `plot.style(...)` convenience
may expose Matplotlib's existing style context. No replacement template engine is
needed. Source-derived axis titles, legend labels and metadata interpolation stay
in Stoner; generic tick/layout formatting belongs in stonerplots.

Apply automatic labels/formatters only to the axes being plotted, not every axes
in a figure. Do not overwrite an explicit user label, locator or formatter on
each added series. Give every axis its own formatter/locator instance.
Drawing inherits current Matplotlib rcParams and the styling of existing axes.
Use `SavedFigure` or `plot.style(...)` for the `stoner` and journal styles. Do not
implicitly reapply a base style while creating a figure: this avoids overriding
an enclosing external style context and needs no hidden context detection.

Remove `DefaultPlotStyle` and custom-template hooks from the main design. Migrate
unique useful bundled styles upstream or show their replacement rc/style context;
`PRB` and `aps` are not visually identical aliases. Review the changed appearance
of template examples rather than claiming pixel equivalence. Remove old runtime
assets/package declarations only once their users have been migrated.

`multiple="common"` becomes ordinary multiple-Y plotting. Replace `multiple=
"panels"`, `multiple="subplots"`, and `multiple="y2"` with layout contexts and
explicit plotting calls. Do not also delegate `subplots`, `layout` or
`secondary_y` to pandas; stonerplots owns these layout operations. The pandas
`stacked` option describes stacking data values and is distinct from
`StackVertical` arranging axes.

## Intelligent dispatch contract

Resolve a request in this order: explicit method/`kind`, explicit selectors,
assigned `setas` roles, then existing column-count defaults. Fill omitted
arguments using roles without undoing explicit choices. Compute the plot type
from the resolved description, rather than retaining an axes-count value that
predates overrides.

| Resolved input | Automatic plot |
| --- | --- |
| X and one/more Y, no uncertainty | Lines on one axes. |
| X/Y with X or Y uncertainty | Error bars; align uncertainty with each selected trace. |
| X/Y/Z scalar field | Gridded 3D surface. |
| X/Y/U/V, optionally W | Existing 2D vector-colour image with optional arrow overlay. |
| X/Y/Z/U/V/W | Matplotlib 3D vector field. |
| X/Y/Z plus scalar U, without a complete vector | Voxels. |
| No roles, 2/3/4 columns | XY / XYE / XDYE. |
| No roles, 5/6 columns | Existing XYUVW / XYZUVW guesses. |
| Insufficient or contradictory information | Informative ValueError explaining the required selectors/kind. |

Do not silently choose Mayavi because it happens to be installed. The proposed
new API uses Matplotlib so axes, layouts and returns remain coherent. Any retained
Mayavi support needs an explicit separate interface and validation; it must not
change the automatic call according to installed optional packages.

Keep scientific gridding/interpolation, vector colour mapping and voxel filling
separate from figure layout. Reuse existing numerical helpers first. Their
defaults are not part of an unrelated gridding redesign.

Before changing missing-data rendering, distinguish removal of excluded points
from gaps in a line. The existing XY helper removes them. Preserve that initial
policy, including per-trace error alignment, or explicitly approve/document a
different rendering policy with scientific examples. Unselected columns must
not change the validity of a selected trace.

## Sensible pandas delegation

Pandas is a renderer option for selected operations, not a mandatory intermediate
for every plot. Start with a small declared dispatch table, not a general
`__getattr__` pass-through. A named method and `plot(kind=...)` share one path.

For every delegated operation:

1. Resolve Stoner selectors to column positions/identities first. Select only the
   needed live-frame columns and detach a plotting copy on each call. Reacquire
   after mutations; do not retain pandas' plotting accessor between calls.
2. Apply the matching Stoner exclusion mask to the copy. Never modify `Data.data`,
   store NaNs in the measurement table, or treat `Data.data.plot()` as mask-aware.
   Avoid using the general lossy `to_pandas()` export merely to prepare a plot:
   that export copies unrelated state and intentionally warns about omitted
   metadata/exclusions.
3. Handle duplicate headers using unique internal plotting keys and restore
   display labels afterwards. Keep Stoner's exact/pattern/positional resolution
   rather than asking pandas to resolve an ambiguous label a second time.
4. Define validity per operation: histogram/box/KDE samples independently per
   series; hexbin jointly across coordinates and any reduced value column.
   Check non-finite inputs, integer-to-NaN conversion, row order and missing-data
   behaviour. Do not apply an unrelated whole-frame `dropna()`.
5. Pass the resolved `ax` and `backend="matplotlib"` explicitly. Reject unsupported
   backend/layout controls with a useful direction to the appropriate context.
   Pandas' configurable backend must not silently change the return contract.
6. Compare with a direct Axes implementation. Keep whichever is smaller and easier
   to explain/test once selection, masks and labels are included. No delegation
   target is an architectural obligation.

The public raw DataFrame remains available for users deliberately choosing native
pandas semantics; its operations do not automatically apply Stoner exclusions.

## stonerplots work to consider

Reviewed upstream `main` at
[`c2500ee0e374657d426a63c3ef6893c95826ee18`](https://github.com/stonerlab/stonerplots/tree/c2500ee0e374657d426a63c3ef6893c95826ee18),
which declares 1.9.5. The installed 1.9.5 was also exercised locally.

| Candidate | Recommendation |
| --- | --- |
| Inset current-axes restoration | Fix and test upstream before depending on it. A local probe showed that leaving `InsetPlot(ax=parent)` did not restore the parent. `__enter__` does not store the current target before `__exit__` attempts restoration. Check replacement-inset handles as well. |
| `PlotLabeller` formatter/locator ownership | Add a regression with axes of very different ranges. Source inspection shows that stored formatter/locator instances are applied to several axes; verify and correct per-axis instantiation upstream. |
| `DoubleXAxis` | Strong case: it replaces Stoner's `x2`, complementing `DoubleYAxis` for independent scales, legend handling and restoration. Share implementation where useful. |
| Transform-related secondary axes | Useful separate candidate using Matplotlib `secondary_xaxis`/`secondary_yaxis`; distinguish a units transform from independent twin axes. Add for demonstrated examples, not as a prerequisite for all plotting. |
| Generic axes-selection context | Not needed for this design: explicit `ax=` and Matplotlib's current target already cover it. |
| Style-only or tick-formatting context | Already covered by Matplotlib style contexts, `SavedFigure(filename=False, ...)` and `PlotLabeller`; do not add duplicates. |

Filed reproducible upstream bug reports after checking existing issues:
[stonerplots #79](https://github.com/stonerlab/stonerplots/issues/79) tracks inset
restoration and [#80](https://github.com/stonerlab/stonerplots/issues/80) tracks
per-axis formatter/locator ownership. Optional strict-xfail probes in
`tests/Stoner/plot/test_upstream_contexts.py` record both 1.9.5 regressions. These
are prerequisites for the later layout integration, not fixes in this checkout.

Also test restoration on exceptions, nested panels/insets/double axes, saving an
existing figure (`SavedFigure(use=...)`), and repeated context instances. These
tests should establish what upstream actually guarantees. Fix generic context
issues in stonerplots rather than adding Stoner-specific copies.

PyPI reports 1.9.5. The phygbu channel contains
`noarch/stonerplots-1.9.5-pyh2b07374_0.conda` and
`noarch/bad_path-0.2.1-py_0.conda`; both conda-forge package endpoints returned 404
on inspection. There is no demonstrated need for a pip subsection. The minimum
version for the new Stoner dependency must include whichever upstream fixes the
new examples require; do not declare 1.9.5 sufficient merely because it imports.

## Internal organisation

Suggested small modules under `Stoner/plot`:

- `accessor.py`: explicit public methods, callable dispatch and docstrings.
- `selection.py`: resolved columns, per-trace masks/errors, labels and immutable
  request descriptions where useful; reuse core resolution rules.
- `renderers.py`: ordinary XY/uncertainty drawing and selected pandas delegates.
- `fields.py`: surface/image/vector/voxel drawing using existing numerical helpers.
- `state.py`: lightweight settings and recent axes/figure bookkeeping, if this is
  large enough to warrant its own module.

Do not add a renderer plugin framework, second storage abstraction or custom
Matplotlib/pandas subclasses for this refactor. Expose stonerplots objects from
the accessor/package. Replace the plotting mixin and its broad magic forwarding;
other analysis/fitting method composition can remain as it is.
Keep numerical utilities such as gridding callable independently of drawing and
preserve utility imports used by image code. Moving them out of the plotting
mixin must not remove them accidentally or make numerical work create a figure.

## Implementation batches and exit gates

Each batch updates this checklist/handover, its examples and its documentation.
Do not merge a stage that documents a different API from the code it contains.
Use a feature branch for the staged implementation and merge the complete redesign
after its integrated gates. Unported field helpers can remain internally during
the transition; they are not a commitment to permanent public compatibility.

- [x] **0. Fix the contracts and establish regression cases.** Use this proposal
  to record the accepted names, Axes returns, target selection and style policy.
  Characterise role/default resolution, current masks/error alignment, metadata
  labels and required 2D/3D examples. Add regressions for the reproduced dispatch
  defects and upstream context issues. Avoid asserting the old accidental API.
- [x] **1. Land the callable accessor and ordinary plotting.** Add `Data.plot`,
  direct XY/scatter/error methods, explicit `ax=`, consistent returns and copy
  semantics. Move/replace ordinary plotting attributes and forwarding; remove
  automatic display on the new path. Retain only the support still needed by
  unported field helpers until stage 5. Update core tests, fitting annotation
  calls, copying helpers and simple examples together. Gate: automatic/explicit plots resolve the same
  request, changing Data is visible on the next call, and drawing never mutates it.
- [ ] **2. Integrate stonerplots layouts and formatting.** Add the dependency and
  public contexts, replace templates and `multiple`, migrate inset/double-axis/
  panel/style examples. Verify/fix upstream restoration and formatting before
  choosing the dependency floor. Gate: the full nested-context examples work
  with one Stoner import; caller styles survive, with no extra axes or hidden
  figures. Update all dependency manifests in this batch.
- [ ] **3. Port fields and special scientific plots.** Move surface, matrix,
  image/contour/pseudocolour, vector and voxel entry points to the accessor. Reuse
  numerical algorithms, remove implicit backend selection and update the 3D/
  colour-field examples. Gate: coordinate orientation, masks, colour mapping,
  limits and selected artists match the intended scientific inputs.
- [ ] **4. Add bounded pandas conveniences.** Evaluate histogram/box/KDE/hexbin
  one family at a time. Implement directly where forwarding is more complicated.
  Gate: excluded values never reappear, duplicate headers resolve consistently,
  only required columns are materialised, and the backend/return contract holds.
- [ ] **5. Complete folders, documentation and removal.** Adapt folder pagination
  to stonerplots, pass `ax=` to members and update bulk callable introspection
  (`_worker` currently reads `func.__name__`). Decide supported nested `each`
  access explicitly; do not accidentally collect accessor objects as metadata.
  Remove remaining old plotting entry points/templates/proxies and update API
  inventories, the migration guide and examples in the same batch. Gate: old
  symbols have no unintended internal callers or published example references.
- [ ] **6. Validate the complete redesign.** Run the focused and full suites,
  lower dependencies, supported hosted matrix, installed wheel/sdist checks and
  documentation checks. Review representative rendered figures and the user
  guide over localhost. Record accepted visual/API changes in release notes.

Dependency changes must reconcile `pyproject.toml`, `requirements.txt`,
`recipe/meta.yaml`, `recipe/build-env.yml`, `tests/test-env.yml`,
`tests/minimum-env.yml`, `doc/docs-env.yml` and `doc/requirements.txt`, including
which build environments actually need the runtime dependency. Keep pure Conda
specifications and the existing Python/Matplotlib/pandas support policy unless
there is a separately justified change. Package builds remain a CI responsibility.

## Validation strategy

Prefer assertions about resolved columns, drawn values, masks, errors, axes,
labels and restored state over a large pixel-baseline suite. Add a few visual
reviews for stacked panels, double axes, insets, surface/vector colour claims and
replacement journal styles.

Essential cases include:

- Missing roles, partially specified roles, explicit overrides, multiple Y
  columns, multiple X groups/`startx`, regex selectors, duplicate headers, metadata
  interpolation and format strings passed explicitly.
- Per-trace exclusions, excluded uncertainties, scalar/array/asymmetric errors,
  non-finite values, empty input and all-excluded input. Validation errors occur
  before drawing and do not leave half-created layouts.
- Repeated plots, changed roles/data between calls, independent Data objects,
  explicit/current axes, new/closed figures, 2D/3D projection mismatches,
  copy/deepcopy/clone/pickle and temporary `Data(...).plot()` expressions.
- Mixed Data/pandas/Matplotlib calls inside the same public stonerplots contexts;
  nesting, exceptions, rcParams restoration, existing-figure saving, actual output
  files, and per-axis formatter independence.
- Folder plots across pages and `folder.each.plot(...)`, with serial Matplotlib
  rendering rather than new thread-safety promises.
- Pandas lower-bound/current behaviour for only the delegated methods, including
  a non-default global pandas plotting backend and no mutation of input storage.

Run executable documentation examples after changing them. Keep all tracked
`doc/plot_cache` files: cached `READTHEDOCS=True` builds must consume them. When
an API change intentionally changes a figure, refresh only the affected cache
entries in a deliberate example/cache batch and review them; do not refresh or
delete the whole cache routinely. Serve generated HTML alone over localhost for
visual checks, as described in `AGENTS.md`.

## Investigation evidence and handover

Verified the existing Miniforge `py314` environment against all 37 Conda package
specifications in `tests/test-env.yml`; none was absent and the checked lower
bounds were satisfied. Relevant versions: Python 3.14.7, Matplotlib 3.11.1, NumPy
2.5.2, pandas 3.0.5, SciPy 1.18.0, pytest 9.1.1 and stonerplots 1.9.5. Import
inspection confirmed this checkout's `Stoner` source and version 0.12.0a1.
Use `conda run`, including its DLL environment, rather than invoking the Python
executable alone.

Existing baseline: **12 passed, 66 deselected**, comprising all five plotting
tests and the seven selected common-Y, double-Y, inset, multiple-panel,
multiple-Y2, single-plot and subplot documentation examples. This is focused local
Windows evidence, not a full-suite or hosted result. The run emitted the existing
SciPy ODR deprecation and Agg display warnings.

Read-only probes additionally reproduced:

- Three unassigned columns select XY with Y uncertainty.
- `d.plot(xcol=0, ycol=1)` with no roles raises a NoneType-call TypeError.
- `d.plot(0, 1, "r-")` raises UnboundLocalError.
- With a Data bound to figure 1, a raw double-Y context on figure 2 leaves Stoner
  drawing on figure 1 and adds no lines to the requested secondary axes.
- stonerplots 1.9.5 does not restore the parent axes after the tested inset block.

### Stages 0 and 1 handover

Implemented on `codex/plotting-accessor-stages-0-1`, based on the inspected `devel`
revision. Changes are uncommitted. The original plan and maintenance-link changes
were carried onto this branch. There is no new runtime dependency in this batch.

- `Data.plot` is a side-effect-free getter returning a callable `DataPlotter`.
  Ordinary `line`, `scatter`, `errorbar` and `errorfill` methods use direct Axes
  primitives, return the primary Axes, respect the caller's style/current target,
  and do not show/save automatically. `new_figure`, `show`, recent live targets and
  column-label overrides are exposed explicitly.
- Selection resolves explicit columns over roles/defaults and recomputes dispatch.
  Role errors follow reordered Y selections; multi-X error positions are corrected
  at this plotting boundary. Per-trace exclusions, finite validity and symmetric/
  asymmetric errors are validated before drawing. Raw measurements are unchanged.
- Copy/deepcopy/clone, `Data(existing)`, `copy_into` and pickle detach live targets
  while preserving independent configuration. Fit annotations use the recent axes
  or an explicit target. Range selection uses its own axes and restores prior
  rendering state in a finally block; folder pagination passes explicit axes.
- The shaded-error helper uses public colour-cycle behaviour, handles asymmetric
  errors with two samples, and avoids division by zero for zero-width shading.
- Stage 2 layouts/templates and stage 3 fields remain on their existing helpers.
  Automatic field selection uses a small transitional bridge and retains those
  helpers' styling/backend limitations; no named field or stonerplots accessor
  attributes are claimed yet. The old mixin remains only for this transition.
  Ordinary plots have no broad Matplotlib attribute forwarding of their own.
- Updated the user guide, public API docs and affected runnable examples. Existing
  layout/template examples explicitly use legacy `plot_xy` until stage 2.
  Refreshed only accessor/single/common-Y/inset/bin guide illustrations: 12 existing
  cache files changed, three accessor cache files added; no original cache removed.

Validation on the verified Windows `py314` environment:

- Final full suite: **1,149 passed, 2 xfailed** in 234.31 seconds. The expected
  failures are the optional stonerplots 1.9.5 probes linked to #79 and #80, not
  failures of the new accessor. All runnable documentation examples were included.
  Reports: ignored `maintenance/runs/plotting-stages01/full-final.xml` and `.log`.
- Focused plotting/core/widget/example recheck: **145 passed, 2 xfailed**.
- Targeted Ruff checks pass for the new production modules and regression tests.
  Whitespace review uses `git -c core.whitespace=cr-at-eol diff --check` to honour
  the checkout's retained line endings; no repository-wide normalisation occurred.
- Fresh staged cached Sphinx HTML: **88 reviewed warnings, zero unexpected**.
  The audit against the earlier verified API inventory found no removed public
  entries and no missing dynamically attached methods; all five primary classes
  remain documented. Explicitly checked `Data.plot` and the new accessor methods.
  The staged build includes the referenced README/config/image assets and avoids
  stale generated stubs in the live documentation tree. All retained cache files
  were consumed without executing plotting examples during this cached build.
- Visually inspected the new four-method figure and the rendered guide/API links
  over localhost. The preview is served from `html-clean` on port 8789 while the
  user reviews it. This is local Windows proof; no hosted, lower-bound or installed
  distribution run is claimed, and no push/release was performed.

Next bounded batch: stage 2. Resolve upstream #79/#80, choose the corresponding
stonerplots dependency floor, then expose its contexts and replace the remaining
layout/template examples. Keep the field migration and pandas conveniences in
their separate planned stages.

## External sources inspected

- [stonerplots public exports and styles](https://github.com/stonerlab/stonerplots/blob/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/__init__.py)
- [stonerplots layout/context implementations](https://github.com/stonerlab/stonerplots/tree/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/context)
- [stonerplots tick formatters and PlotLabeller](https://github.com/stonerlab/stonerplots/blob/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/format.py)
- [pandas DataFrame.plot API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
- [pandas plotting and missing-data behaviour](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [PyPI metadata](https://pypi.org/pypi/stonerplots/json),
  [phygbu stonerplots metadata](https://api.anaconda.org/package/phygbu/stonerplots),
  [phygbu bad_path metadata](https://api.anaconda.org/package/phygbu/bad_path)
