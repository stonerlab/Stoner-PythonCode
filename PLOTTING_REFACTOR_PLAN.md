# Plotting redesign

Status: stages 0-6 completed on 18 October 2026.
Post-migration extension: OffsetPlots and the executable-example audit, 25 October 2026 (see below).
Further extension: DoubleXAxis/DoubleXYAxis with stonerplots 1.9.8 (see below).
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

The accessor now exposes ordinary, scientific field and statistical plotting,
together with the stonerplots layout/formatting contexts. Folder pagination and
serial bulk plotting use this interface; the transitional top-level plotting
methods, templates and broad formatting proxies have been removed. Integrated
hosted, lower-dependency and installed-distribution validation have passed; see
the stage 6 handover for results and the separate Attocube performance follow-up.

## Initial investigation before the redesign

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
| `hist`, `box`, `kde`, `hexbin` | Bounded statistical methods; stage 4 selected direct Matplotlib/SciPy after checking selection and missing-data contracts. |

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
which declares 1.9.5. That initial investigation exercised installed 1.9.5;
stage 2 subsequently verified the released 1.9.6 fixes as recorded below.

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
per-axis formatter/locator ownership. Both probes now pass as required regressions
against stonerplots 1.9.6 in `tests/Stoner/plot/test_upstream_contexts.py`. The
generic fixes remain upstream; Stoner does not copy or monkeypatch them.

Also test restoration on exceptions, nested panels/insets/double axes, saving an
existing figure (`SavedFigure(use=...)`), and repeated context instances. These
tests should establish what upstream actually guarantees. Fix generic context
issues in stonerplots rather than adding Stoner-specific copies.

At the initial investigation, PyPI reported 1.9.5 and the phygbu channel contained
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
- `statistics.py`: independent distribution samples and joint hexbin inputs,
  drawn directly with Matplotlib and SciPy's Gaussian KDE estimator.
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
- [x] **2. Integrate stonerplots layouts and formatting.** Add the dependency and
  public contexts, replace templates and `multiple`, migrate inset/double-axis/
  panel/style examples. Verify/fix upstream restoration and formatting before
  choosing the dependency floor. Gate: the full nested-context examples work
  with one Stoner import; caller styles survive, with no extra axes or hidden
  figures. Update all dependency manifests in this batch.
- [x] **3. Port fields and special scientific plots.** Move surface, matrix,
  image/contour/pseudocolour, vector and voxel entry points to the accessor. Reuse
  numerical algorithms, remove implicit backend selection and update the 3D/
  colour-field examples. Gate: coordinate orientation, masks, colour mapping,
  limits and selected artists match the intended scientific inputs.
- [x] **4. Add bounded pandas conveniences.** Evaluate histogram/box/KDE/hexbin
  one family at a time. Implement directly where forwarding is more complicated.
  Gate: excluded values never reappear, duplicate headers resolve consistently,
  only required columns are materialised, and the backend/return contract holds.
- [x] **5. Complete folders, documentation and removal.** Adapt folder pagination
  to stonerplots, pass `ax=` to members and update bulk callable introspection
  (`_worker` currently reads `func.__name__`). Decide supported nested `each`
  access explicitly; do not accidentally collect accessor objects as metadata.
  Remove remaining old plotting entry points/templates/proxies and update API
  inventories, the migration guide and examples in the same batch. Gate: old
  symbols have no unintended internal callers or published example references.
- [x] **6. Validate the complete redesign.** Run the focused and full suites,
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
revision. Committed as `374980c2e` before starting stage 2. The original plan and maintenance-link changes
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

### Stage 2 handover

Implemented on the same feature branch, following the stages 0/1 commit.
Committed as `ec0d62fa6` before starting stage 3; no push or release was performed.

- Added required `stonerplots>=1.9.6` to Python, Conda, testing and documentation
  manifests; the minimum environment pins 1.9.6. The recipe build environment
  intentionally contains packaging tools only. Verified PyPI metadata and the
  `phygbu` Conda build, then upgraded only stonerplots from 1.9.5 to 1.9.6 in
  `py314`. All 38 test-environment Conda specifications are satisfied.
- Exposed the seven planned context classes and both tick formatters directly on
  `DataPlotter` and `Stoner.plot`, preserving upstream identity/signatures/handles.
  `plot.style` is Matplotlib's native style context. No layout stack, context
  wrappers, template engine or implicit styling was introduced.
- Migrated inset, double-axis, stacked/grid-panel, colour-cycle, journal, sketch,
  Seaborn and fitting-residual examples. Legacy template API documentation now
  points to replacement examples and explains their changed appearance. Field
  helpers and folder pagination retain their required templates/assets until
  stages 3/5; their implementations were not prematurely removed.
- Nested layouts, inset handle identity, repeated contexts, normal/exception
  restoration, actual saving of new/existing figures, mixed pandas/Matplotlib
  draws, caller style restoration and independent ticker ownership are tested.
  `SavedFigure` and `PlotLabeller` scan figures at entry/exit and can change the
  current pyplot target. This upstream behaviour is documented; explicit `ax=`
  handles remain reliable. `DoubleXAxis` is still an upstream candidate, not a
  1.9.6 export; the guide uses native `Axes.twiny()` for that case.
- Corrected a sample uncertainty to use non-negative magnitudes, and an existing
  fit annotation to use axes-fraction coordinates. Refreshed only the affected
  guide/API cache entries: 23 example groups, including one new nested-context
  illustration. All existing cached files are retained.

Validation on Windows `py314` with stonerplots 1.9.6 and the previously recorded
Python/Matplotlib/NumPy/pandas/SciPy versions:

- Full suite: **1,175 passed**, including all runnable documentation examples and
  both former upstream xfails. Report: ignored
  `maintenance/runs/plotting-stage2/full.xml` and `.log`.
- Final focused plotting recheck: **79 passed**. The annotation-only example
  correction was rechecked separately after the full run.
- Targeted Ruff passes; only the existing capitalised `Stoner` package-name
  convention (`N999`) is excluded. Dependency consistency and whitespace pass.
- Fresh cached Sphinx HTML returns to **88 reviewed warnings, zero unexpected**,
  with no lost API entries or missing dynamic methods. The context aliases are
  documented without duplicating upstream class docstrings. Retained cache
  graphics are consumed by `READTHEDOCS=True` builds.
- Visually reviewed representative stacked/inset/double-axis, APS, centred-axes,
  custom-style and residual figures, plus the guide/API navigation over localhost.
  The stage 2 preview is served from `html-final` on port 8790. Validation is local;
  no hosted matrix, lower-dependency environment or installed-distribution result
  is claimed.

### Stage 3 handover

Implemented on the same feature branch after the stage 2 commit. Committed as
`1c1293512` before starting stage 4; no push or release was performed.

- Added explicit `surface`, `contour`, `pcolormesh`, `image`, `matrix`, `scatter3d`,
  `vector`, `quiver`, `vector3d`, `voxels` and `waterfall` methods. Named methods,
  `kind=` and automatic scientific dispatch share the native Matplotlib path.
  All return the primary Axes and inherit caller styling without showing/saving.
  A colour bar does not replace the recent primary target, including with an
  unmanaged Figure. The old automatic field bridge has been removed.
- Reused SciPy `griddata` interpolation and Stoner's vector HSL mapping, keeping
  independent numerical `Data.griddata` behaviour. Required columns alone define
  joint validity; measured exclusions and convex-hull gaps remain missing.
  Regular lattices preserve holes directly. Duplicate sites use the first valid
  sample, consistently for both vector pixels and their arrow overlay.
- Made coordinate interpretation explicit: grid counts are X/Y[/Z], 2D renderer
  arrays are Y/X, matrix inputs are rows X/columns Y, and voxel arrays are X/Y/Z.
  Image limits derive from half-cell edges, retaining scalar values for normalised
  colours. Matrix crops slice corresponding coordinates. Voxel visibility uses
  grid centres; corners and the colour bar share the documented volume mapping.
- Kept automatic three-column XYE guessing; an explicit field method supplies its
  own sensible column-order defaults on an unassigned table. Incompatible field
  uncertainties and obsolete layout/backend controls raise useful errors.
- Removed implicit Mayavi selection and its unused optional extra. Even the
  retained legacy 3D-vector entry point now forwards to Matplotlib. Other legacy
  public helpers/templates remain until stage 5. Corrected the HSL hue-zero sector
  and finite zero/constant-vector cases; explicit waterfall axes no longer create
  a hidden pyplot figure.
- Migrated the surface, matrix, scalar, vector, voxel, waterfall and 3D-fitting
  examples, adding a masked scalar comparison and rotating 3D vector example.
  Corrected the plane example's displayed equation to match its fitted function.
  Refreshed 13 affected cache groups, including two new groups, retaining every
  original cache file. The guide documents shapes, masks, colours and migration.

Validation on Windows `py314`, with all 38 test-environment Conda specifications
satisfied and the stage 2 versions including stonerplots 1.9.6:

- Final full suite: **1,218 passed** in 232.18 seconds, including all runnable
  documentation examples. Report: ignored
  `maintenance/runs/plotting-stage3/final-full.xml` and `.log`.
- Focused field suite: **41 passed**, covering non-square orientation, scalar
  normalisation, joint exclusions, interpolation, duplicate vectors, zero fields,
  pure Matplotlib 3D rendering, voxel centres/corners/visibility, matrix cropping,
  target state, source immutability and independent numerical gridding.
- Targeted Ruff checks pass for new/changed accessor code and tests, together
  with undefined-name checks on the modified legacy helpers. No required runtime
  dependencies changed; Mayavi was declared only in the removed Python extra.
- Fresh cached Sphinx HTML: **88 reviewed warnings, zero unexpected**, with no
  lost public inventory entries or missing dynamic Data methods. All **34**
  required accessor/context targets have canonical inventory entries, including
  the 11 new field methods. All 249 pre-existing cache files remain present;
  only affected groups were refreshed and six files added for the two new groups.
- Visually reviewed scalar/masked, surface, vector, voxel and fitted-sphere
  graphics, then the guide and new API links over localhost. Enlarged the new
  3D-vector example to retain its labels and re-rendered it after the full run;
  this final change affects figure size only. The stage 3 preview is served from
  `maintenance/runs/plotting-stage3/html` on port 8791 while the user reviews it.
  Whitespace checks pass with retained line endings. Validation is local Windows
  evidence; hosted, lower-dependency and installed-distribution gates remain for
  the complete redesign.

### Stage 4 handover

Implemented on the same feature branch after the stage 3 commit. Stage 4 was
committed as `4c948401e` before stage 5; no push or release was performed.

- Evaluated histogram/box/KDE/hexbin delegation against direct drawing. All four
  are smaller and clearer with native Matplotlib arrays after Stoner selection
  and mask handling; KDE uses the existing SciPy dependency's `gaussian_kde`.
  No pandas intermediate/export, backend forwarding, subclass or general plotting
  proxy was added. No dependency manifest changes are needed.
- Added explicit `hist`, `box`, `kde` and `hexbin` methods and matching `kind=`
  calls. Automatic scientific role/count dispatch is unchanged. Every method
  returns its primary Axes, inherits caller styles, honours explicit/current
  targets, and does not show/save. Hexbin colour bars retain the primary target,
  including on unmanaged object-oriented Figures.
- Distribution selectors use Y roles in the selected X group, otherwise all real
  numeric columns. Explicit selectors preserve Stoner lookup and ordering;
  `no_guess=True` disables the numeric fallback. There is no separate X validity
  requirement. Each series and optional source-row weights are filtered jointly,
  independently of other series or unselected columns. Every selected series must
  retain samples. Integer data is filtered without inserting NaNs into storage.
- Histograms share native bin edges and retain density/cumulative/stacked and
  signed-weight semantics. Box plots keep independent samples and categorical
  labels; a small signature-based choice uses native orientation/tick-label
  keyword names across the declared Matplotlib floor and current versions.
- KDEs require two distinct positive-weight samples, non-negative weights and
  valid bandwidth/evaluation controls. All curves are prepared before axes are
  selected. The default grid extends half the observed range beyond both ends;
  zero-weight outliers do not affect it. The guide states the Gaussian-tail
  interpretation rather than implying physical-boundary correction.
- Hexbin validity combines X/Y and optional C only, with native count/reduction,
  grid, logarithmic binning and colour-map controls. Raw source-row C arrays and
  selectors align with exclusions. Backend, grouping, legacy and layout controls
  are explicitly rejected; stonerplots contexts remain responsible for layouts.
- Added a runnable four-panel example and its three cache formats. No existing
  cache was refreshed or removed. The guide and accessor docstrings explain the
  selection, missing-data, weighting and return contracts.

Validation on Windows `py314`, freshly verified against all 38 test-environment
Conda specifications, with Python 3.14.7, stonerplots 1.9.6, Matplotlib 3.11.1,
NumPy 2.5.2, pandas 3.0.5 and SciPy 1.18.0:

- Full suite: **1,271 passed** in 255.94 seconds, including every runnable
  documentation example. Reports: ignored
  `maintenance/runs/plotting-stage4/full.xml` and `.log`.
- Focused statistical suite: **52 passed**. It checks numerical histogram/box/KDE
  results, hexbin counts/reductions against native Matplotlib, independent masks,
  weights and non-finite inputs, repeated calls after mutation, required-column
  reads, duplicate selectors, source ownership/immutability, caller styles, and a
  non-default pandas backend.
- Targeted Ruff passes for the accessor, new statistics module, tests and example.
  The core Data change is a one-line docstring update. Whitespace checks pass with
  retained line endings; the new regression path uses tracked `tests/Stoner` case.
- Fresh cached Sphinx HTML: **88 reviewed warnings, zero unexpected**, with no
  lost public inventory entries or missing dynamic Data methods. All **38**
  required accessor/context targets have canonical inventory entries. All 255
  pre-existing cached files remain present; the new example adds three formats.
- Visually reviewed the new example, rendered guide and new API navigation over
  localhost. The current preview serves only
  `maintenance/runs/plotting-stage4/html` on port 8792 while the user reviews it.
  These are local Windows results; the supported hosted/lower-dependency and
  installed-distribution gates remain for stage 6.

### Stage 5 handover

Implemented on `codex/plotting-accessor-stages-0-1` after committing stage 4.
Stage 5 was committed as `8d6b49b93` before stage 6 and pushed for hosted
validation. No release was performed.

- Replaced PlotFolder pagination with stonerplots MultiPanel and explicit member
  Axes. Pages support one/many members, portrait/landscape figures and mixed 2D/3D
  projections, with unused slots removed. Returns contain one primary Axes per
  member; colour bars are separate. `folder.figures` inspects recent live pages.
  Figure configuration is independent of member targets and detached on copying.
- Added explicit `folder.each.plot()` and named drawing methods, also available
  through dotted strings and `each.iter`. Plotting is serial even with global
  multiprocessing/threading enabled, including loading members without parallel
  prefetch. Calls preserve metadata by default; explicit `_return` retains the
  bulk result-storage contract. Accessor configuration/context attributes are not
  collected or exposed. Generic user callables can still request `_serial=True`.
  Callable introspection now supports partials and callable objects.
- Removed PlotMixin, top-level drawing/layout methods, template classes, their
  broad pyplot/Axes/Figure proxies and bundled template stylesheets. The required
  stonerplots dependency supplies styles. Retained independent numerical
  `Data.griddata` behaviour and image display/utilities. Updated copying, fitting
  labels and RangeSelect to use accessor state and explicit Axes/Figures.
- Migrated remaining runnable examples and API/upgrade guides together, including
  fitting annotations, smoothing layouts, image-profile plots and the full FMR
  folder example. Its uncertainty conversion uses an explicit column assignment
  to respect the storage API's read-only coordinate views. Added a synthetic
  pagination/overlay example. Refreshed only affected cache groups and retained
  every pre-existing cache file.
- Removed obsolete style assets from package-data checks while retaining OCR
  resources. Updated the installed-package style smoke check. Dependency floors
  remain unchanged. Reviewed 319 intentionally removed plotting inventory targets
  in `maintenance/docs/plotting-removed-api.json`; tightened both platform warning
  baselines by removing 26 obsolete allowances, without raising ceilings.

Validation on Windows `py314`, freshly verified against all 38 test-environment
Conda specifications, with Python 3.14.7, stonerplots 1.9.6, Matplotlib 3.11.1,
NumPy 2.5.2, pandas 3.0.5 and SciPy 1.18.0:

- Full suite: **1,324 passed** in 189.16 seconds, including runnable documentation
  examples and all 28 new folder plotting cases. Reports: ignored
  `maintenance/runs/plotting-stage5/final.xml` and `.log`. A focused core/bulk/folder
  recheck passed 45 tests. The complete main-guarded FMR example separately ran
  successfully and produced five figures.
- Targeted Ruff checks pass for the accessor, numerical gridding, bulk plotting
  and new/rewritten plotting tests. Undefined-name checks pass modified legacy
  consumers; unrelated existing lint findings remain outside this batch.
  Whitespace checks pass with retained line endings and canonical test path case.
- Source package-content checks pass for retained runtime assets. This is source
  evidence; no local distribution build or installed-distribution gate is claimed.
- Fresh cached Sphinx HTML has **62 reviewed warnings, zero unexpected**, down
  from 88 after removing obsolete allowances. API comparison finds no unintended
  removals; 38 accessor/context targets and four folder/gridding targets retain
  canonical inventory entries. All 258 pre-existing cached files remain; 123
  affected files were refreshed and 18 new files supply six rendered groups.
- Visually reviewed pagination, the overlay, migrated fitting output and the
  rendered guide/API navigation over localhost. The preview serves only
  `maintenance/runs/plotting-stage5/html` on port 8793 while the user reviews it.

The results above are local Windows evidence. The following stage 6 handover
records the integrated lower-dependency, hosted and installed-distribution gates.

### Stage 6 handover

- Hosted wheel/sdist checks passed for stage 5 on Python 3.12 and 3.14. Extended
  the maintained installed-package probe to exercise scientific inference with
  masks, stonerplots panels/insets/double axes, saving an existing figure, surface
  rendering, clone target detachment, pagination and named bulk drawing.
- A rejected all-excluded folder page left an open partial figure. Failed pages
  now close and restore member targets while preserving earlier completed pages.
  Regressions cover invalid data, callback errors and interruption.
- The hosted lower-dependency run exposed global external-LaTeX configuration in
  `scale_curves.py`, which broke a later example. The unnecessary setting is
  removed; example tests isolate rcParams and render every resulting figure.
  All 79 rendered examples pass locally, as do 37 focused folder/archive checks.
  Only the three affected scaling-cache files were refreshed.
- Added unreleased plotting API/visual release notes and aligned testing guidance
  with the current Python, pandas/xarray and stonerplots support boundaries.
  These changes were committed as `855f37010` and pushed to exercise the hosted
  matrix and extended installed-distribution probes.
- Five fitting examples now select Matplotlib's bundled DejaVu Sans for small
  mathematical annotations, avoiding the FreeType failures reproduced with the
  platform's preferred sans-serif font on macOS. Their five cached groups were
  refreshed. This typography change was committed as `33b36ec81`; caller styles
  and scientific implementations are unchanged. All five examples were rechecked
  with current/lower dependencies and passed in the final macOS run.

Validation (reports under ignored `maintenance/runs/plotting-stage6`):

- Current Windows environment, freshly verified against all 38 specifications:
  **1,327 passed** in 231.86 seconds with two workers. Separate rendered-example
  verification passed all 79 examples. The focused folder/archive run passed 37.
- Fresh local lower-dependency environment, verified against all 38 specifications:
  Python 3.12.15, NumPy 2.0.2, pandas 2.2.3, xarray 2024.10.0, SciPy 1.14.1,
  Matplotlib 3.8.4 and stonerplots 1.9.6. **1,326 passed, 1 skipped** in 264.70
  seconds. The skip characterises pandas 3's old accessor lifecycle and does not
  apply to pandas 2.2. Rendering exposes existing Matplotlib 3.8/Pyparsing alias
  deprecations; numerical/model warnings remain visible without new suppression.
- Fresh cached Sphinx HTML: **62 reviewed warnings, zero unexpected**, with no
  lost inventory entries or missing dynamic Data methods. All 38 required
  accessor/context names are canonical. All 276 cached files remain present.
  Source package-content, CI policy, repository-content and targeted Ruff checks
  pass. The changed scaling cache, context/panel/inset/twin-axis, scalar, vector,
  surface, statistical and replacement journal-style figures were reviewed.
- Hosted Python 3.12-3.14 Linux and Python 3.14 macOS each passed **1,327 tests**
  at `33b36ec81`. The Python 3.12 lower-dependency job passed **1,326 with 1
  expected skip**. Both Python 3.12/3.14 package jobs and the cached documentation
  build passed at that same commit. The package jobs built and strictly checked wheel/sdist metadata,
  then independently installed and exercised each archive outside the checkout,
  including the extended plotting probe. No distribution was built locally.
- Final matrix: [pytest run 39149915845](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39149915845).
  Lower lines: [39149915877](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39149915877).
  Installed distributions: [39149915872](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39149915872).
  Cached docs: [39149915436](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/39149915436).
- The final macOS run completed in 890.23 seconds; `test_attocube_scan` passed in
  201.434 seconds. Its two-minute faulthandler dump was diagnostic, not a fatal
  test timeout. A previous slow attempt was cancelled to retrieve partial logs;
  those logs showed continued progress, so only macOS was rerun on the same
  commit and allowed to finish. The final JUnit reports contain zero failures.
  The snapshot path in ScanMetadataMixin._instantiate eagerly exports image
  pixels/masks when rebinding frame metadata. Investigating that cost is a
  separate bounded storage-performance follow-up; no image implementation was
  changed during stage 6.

Current preview: `maintenance/runs/plotting-stage6/html` on port 8794, with
release notes and the updated plotting guide checked in the browser. No merge,
tag, release or publication has been performed.

## Post-migration extension: OffsetPlots (25 October 2026)

- Expose `OffsetPlots` directly from both `Data.plot` and `Stoner.plot`, with no
  wrapper or copied upstream implementation. Raise the stonerplots floor to
  1.9.7 across runtime, recipe, testing and documentation requirements. The
  packaging-only build environment needs no runtime dependency change.
- Document new/existing axes, linear/logarithmic offsets, pairing, complete error
  bars and unsupported artists. Add the executable two-panel example and its
  three cached render formats, retaining all 276 pre-existing cached files.
- Integration regressions cover export identity, per-trace exclusions, paired
  line/scatter artists, complete error bars (including `fmt="none"`), log factors,
  source-state preservation, existing axes and exception/target restoration.
  Extend the maintained installed-package plotting probe for future CI archives.
- Audit all 96 Python samples/scripts (274 plotting calls) and the guides/runtime
  docstrings for retired drawing calls. Retain deliberate migration comparisons
  and valid non-plotting `xcol`/`ycol` controls. Correct the image-guide FFT keyword
  and semilog plotting, the cookbook import/vector descriptions and FMR annotation
  coordinates/targets. The formerly skipped main-guarded FMR workflow is now part
  of the rendered-example collector; add image-derived table rendering coverage.
- Both current and lower test environments satisfy all 38 Conda specifications.
  Current versions: Python 3.14.7, Matplotlib 3.11.1, pandas 3.0.5; lower versions:
  Python 3.12.15, Matplotlib 3.8.4, pandas 2.2.3. Both use stonerplots 1.9.7.
  Plotting/example/script runs pass 315 tests with three expected issue #41
  failures in each environment before adding FMR to the collector. The final
  lower-dependency OffsetPlots/FMR/image recheck passes 14 tests.
  The final current full suite passes **1,372 tests with three expected issue #41
  failures**, including all 81 rendered documentation examples and the new FMR
  and image-profile regressions (221.20 seconds with two workers).
- Fresh cached Sphinx HTML has 62 reviewed warnings, zero new warnings and no
  removed inventory entries; all 39 required accessor/context names are canonical.
  Visually reviewed the new guide, illustration, API entry and linking over
  localhost. Source package-content, CI/repository policy and targeted lint checks
  pass. The installed-package probe was exercised from source; no installed
  wheel/sdist or hosted CI result is claimed for these changes.

Validation reports: ignored `maintenance/runs/offsetplots/`. Preview:
`http://127.0.0.1:8795/UserGuide/plotfile.html#plotting-offsets`.
Included in 0.12.0a3 release preparation; hosted/publication gates are tracked in
MAINTENANCE_PLAN.md.

## Post-migration extension: DoubleXAxis and DoubleXYAxis (25 October 2026)

- Directly expose both upstream classes through `Data.plot` and `Stoner.plot`,
  alongside `DoubleYAxis`. Reconcile the required stonerplots version to 1.9.8
  across runtime, recipe and test/docs environment manifests; the packaging-only
  build environment remains independent of runtime requirements.
- Add API/guide/upgrade documentation and a two-panel executable comparison of
  shared Y versus independent X/Y. Replace the obsolete `twiny()` recommendation
  in the migration guide. Add only the new example's three cached render files.
- Extend export identity and exception-restoration coverage to both contexts.
  Integration tests exercise all three double-axis variants with automatic and
  explicit targets, sharing/independent limits, colours, merged legends, source
  values and overlaid geometry after layout/resizing. All 44 context tests pass.
  The maintained installed-package probe also exercises the new contexts; its
  source-level preflight passes against the Conda release of stonerplots 1.9.8.
- Current Python 3.14/Matplotlib 3.11 and lower Python 3.12/Matplotlib 3.8
  environments each satisfy all 38 Conda test-environment specifications.
  Full current suite: **1,383 passed, three expected issue #41 failures**,
  including 82 rendered examples. Lower-dependency plotting/examples: **299
  passed**. The full run exposed an inherited constrained-layout warning in the
  new example; it now explicitly selects manual layout. Final double-axis/context
  example checks pass 14 tests in each environment, with layout warnings treated
  as errors. Separate rendering with inherited constrained layout also passes in
  both environments. The final sample-only correction was verified by these
  focused runs rather than another full-suite repetition.
  Cached Sphinx HTML has 62 reviewed warnings and no unexpected warnings or
  removed API entries; all 41 required accessor/context targets are canonical.
  Reviewed the guide, API entries, navigation and figure over localhost.
  Targeted lint and source package-content/CI/repository policy checks pass.
  These are local results, without hosted CI or installed-archive validation.

Reports: ignored `maintenance/runs/double-axes/`. Local preview:
`http://127.0.0.1:8796/UserGuide/plotfile.html#plotting-double-axes`.
Included with OffsetPlots in 0.12.0a3 release preparation; see MAINTENANCE_PLAN.md
for hosted validation and publication status.

## Post-migration extension: style names and overrides (25 October 2026)

- Replace the direct Matplotlib style alias with a small style-only context.
  Comma-separated strings follow SavedFigure's whitespace/empty-entry handling;
  lists retain order and `extra=` overrides all styles. None/omission uses
  `stonerplots.default.style`, while an empty list/string applies no stylesheet.
  Preserve native mappings, mixed lists, path-like files and the positional/
  keyword `after_reset` option. Do not select/create/save/close figures.
- Migrate eleven examples from mixed dictionaries/style lists to `extra=` or
  comma-separated names. Existing single-style calls remain valid. Updated guide,
  API docstrings and release notes document defaults and override precedence.
  Appearance is unchanged, so existing plot-cache files are retained.
- Focused current-environment checks pass 17 style tests; lower-dependency checks
  pass 31 style/example tests. Full rcParams snapshots agree with SavedFigure for
  names, ordered lists, defaults and overrides. Regressions cover nesting, errors,
  invalid overrides, reset behaviour, file inputs and unchanged current targets.
  Both environments satisfy all 38 Conda specifications; targeted lint passes.
  The full current suite passes **1,395 tests with three expected issue #41
  failures**, including all 82 rendered examples, in 242.82 seconds with two workers.
- Fresh cached Sphinx HTML has 62 reviewed warnings, zero unexpected warnings
  and no removed API entries. Reviewed the rendered guide and style method's
  signature/documentation over localhost. No cached images were refreshed.

Reports: ignored `maintenance/runs/style-wrapper/`. Preview:
`http://127.0.0.1:8797/UserGuide/plotfile.html#plotting-contexts`.
Included in 0.12.0a3 release preparation. The results above are local;
MAINTENANCE_PLAN.md tracks exact-revision hosted and installed-distribution gates.

## External sources inspected

- [stonerplots public exports and styles](https://github.com/stonerlab/stonerplots/blob/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/__init__.py)
- [stonerplots layout/context implementations](https://github.com/stonerlab/stonerplots/tree/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/context)
- [stonerplots tick formatters and PlotLabeller](https://github.com/stonerlab/stonerplots/blob/c2500ee0e374657d426a63c3ef6893c95826ee18/src/stonerplots/format.py)
- [pandas DataFrame.plot API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
- [pandas plotting and missing-data behaviour](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [SciPy scattered griddata interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html)
- [Matplotlib image origin and extent](https://matplotlib.org/stable/users/explain/artists/imshow_extent.html)
- [Matplotlib voxel corners and colours](https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.html)
- [Matplotlib histogram arrays, shared bins and weights](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)
- [Matplotlib box plot options](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html),
  [Matplotlib 3.8 box plot keyword names](https://matplotlib.org/3.8.4/api/_as_gen/matplotlib.axes.Axes.boxplot.html)
- [Matplotlib hexbin counts and reductions](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hexbin.html)
- [SciPy Gaussian KDE and bandwidth conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
- [PyPI metadata](https://pypi.org/pypi/stonerplots/json),
  [phygbu stonerplots metadata](https://api.anaconda.org/package/phygbu/stonerplots),
  [phygbu bad_path metadata](https://api.anaconda.org/package/phygbu/bad_path)
