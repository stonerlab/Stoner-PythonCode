# Data.fit implementation and migration plan

Date: 2026-10-29. Target checkout: `devel`.
Status: batches 1-3 implemented. The registry, decorators, `FitOutput`, `Data.fit`
and `folder.each.fit` are available, and internal package fitting calls are migrated.
Batch 4 documentation/sample/script migration is implemented; compatibility warning timing and removal policy remain undecided.
The earlier `annotate_fit` edits are preserved.

See [DATA_ACCESSOR_PLAN.md](DATA_ACCESSOR_PLAN.md) for other candidate groups and
[AGENTS.md](AGENTS.md), [DOCSTRING_STYLE.md](DOCSTRING_STYLE.md) and
[MAINTENANCE_PLAN.md](MAINTENANCE_PLAN.md) for repository requirements.

## 1. Agreed intent and scope

- Introduce an owner-bound `Data.fit` accessor for the existing fitting methods.
- Provide explicit algorithm methods: `curve_fit`, `lmfit`, `odr`,
  `differential_evolution` and the existing `polyfit`.
- Use `x` and `y` for coordinate-column selectors on the accessor, matching
  `Data.plot`. Translate them to the existing implementations' `xcol` and `ycol`.
- Resolve registered model names dynamically as fitting shortcuts. For example,
  `d.fit.linear(...)` delegates to the same machinery as `d.fit.lmfit(Linear, ...)`.
- Store registry entries under snake_case names mapped to model classes/factories.
  Retain existing model class names and direct imports.
- Augment the existing `@make_model` decorator to register the class it creates;
  preserve its existing use and its `.guesser` and `.hinter` decorators.
- Check physics names against sources. Conversation spelling is not an instruction
  to create a misspelt public API.
- Preserve scientific behaviour and existing fitting options. This is an API
  organisation change, not a rewrite of optimisation or result storage.
- Introduce `FitOutput`, a string enum for the existing output labels, while
  continuing to accept supported strings. Follow the package-wide conventions
  and staged audit in [STRING_ENUM_PLAN.md](STRING_ENUM_PLAN.md).
- Keep annotations on `Data.plot.annotate_fit`, including its text-only mode.

The detailed contracts below are implementation proposals where the discussion
has not yet fixed a choice. Resolve them before the corresponding implementation
batch; do not silently treat proposed migration policy as previously approved.

## 2. Proposed public interface

```python
from Stoner.analysis.fitting.models.generic import Linear

# Simple intent-first interface; setas supplies omitted coordinate selectors.
d.fit.linear()
d.fit.linear(x="Field", y="Signal", p0=initial_parameters)
d.fit.bloch_grueneisen(x="Temperature", y="Resistivity")

# Explicit algorithm interface for configured, custom or composite models.
report = d.fit.lmfit(Linear, x=0, y=1, output="report")
d.fit.curve_fit(Linear, x=0, y=1, p0=initial_parameters)
d.fit.odr(Linear, x=0, y=1)
d.fit.differential_evolution(Linear, x=0, y=1, p0=parameter_bounds)
d.fit.polyfit(x=0, y=1, polynomial_order=2)

# Read-only discovery of registered classes/factories.
model_class = d.fit.models["linear"]
available_names = list(d.fit.models)
```

These interface sketches omit measurement setup and initial values. Supply real
initial values/bounds in runnable documentation. In particular, do not imply
that different algorithms interpret `p0` identically.

### Algorithm and shortcut signatures

Keep each algorithm's existing parameter order and defaults, replacing its
implicit owner argument with the accessor's owner and renaming `xcol`/`ycol` to
`x`/`y`. A model shortcut has the accessor's `lmfit` signature with the model
argument already supplied:

```python
d.fit.linear(x, y, p0, sigma, **kwargs)
# Equivalent to:
d.fit.lmfit(Linear, x, y, p0, sigma, **kwargs)
```

Here `x` and `y` are existing Stoner column selectors, not new raw x/y arrays.
Omitting them retains the algorithm's existing column-role/default resolution.
Translate these names only at the accessor-to-implementation boundary; numerical
implementations and their direct function interfaces retain `xcol`/`ycol`.
Reject legacy `xcol`/`ycol` keywords on the new accessor with a clear `TypeError`
directing callers to `x`/`y`; do not let them leak through `**kwargs` to an
optimiser or create ambiguous double specifications. Direct legacy `Data` methods
retain `xcol`/`ycol` throughout their compatibility period. Forward
`bounds`, `where` where supported, `prefix`, `result`, `replace`, `header`,
`residuals`, `output` and optimiser options using existing algorithm contracts.
Accept `FitOutput.FIT`, `REPORT`, `ROW`, `FULL`, `DATA` and `DICT` alongside their
existing string values wherever that algorithm supports the option. Retain
`polyfit`'s existing interface rather than adding an unsupported output argument.
Do not consume `method` as an accessor algorithm switch: it may already select
an lmfit optimiser. Users select another algorithm through its explicit method.

Agreed and implemented: `d.fit(model, ...)` is equivalent to
`d.fit.lmfit(model, ...)`, giving the accessor a callable form like `DataPlotter`.

### Return values and side effects

Return exactly what the existing algorithm returns for the same options. Do not
introduce a universal result object or make all fits return `Data`. Existing
`output="data"` calls can chain through the owner; `output="report"` returns the
algorithm's report. Preserve fit metadata prefixes, parameter labels, uncertainties,
fit/residual columns, masks, column roles and multi-channel behaviour.
The enum standardises the selector, not the result object's type or contents.
Audit and document the supported-output matrix and use one shared definition
for validation/dispatch across the accessor and direct fitting entry points.

Obtaining `d.fit`, listing its names or resolving `d.fit.linear` must not fit,
guess parameters, instantiate a model, read numerical data or create a figure.
Each invocation uses the current owner and creates fresh model state.

## 3. Model registry and naming

Proposed registry contract:

- One process-local registry shared by `Data` instances, storing classes/factories
  and useful provenance (source module, qualified name and display name).
- Expose a live read-only mapping through `d.fit.models`; modifications go through
  registration functions. Do not serialise this global registry into each `Data`.
- Support `lmfit.Model` subclasses first. A factory must explicitly register under
  a name and produce a fresh `Model`; validate its result when invoked rather
  than calling it during registration. Raw functions enter through `@make_model`.
- Generate ASCII snake_case names from meaningful public names, handling existing
  underscores and acronyms. Validate explicit names as public Python identifiers
  in canonical snake_case, rejecting keywords, private names and invalid names.
- Reserve all public accessor members, including algorithm names and `models`.
  Reject collisions rather than allowing a model to replace an accessor method.
- Re-registering the same object under the same name is idempotent. A different
  object with that name raises a useful error naming both sources. Explicit
  replacement can be supported for interactive development; never replace silently.
- Unknown attribute names raise `AttributeError`, optionally suggesting close
  matches. Mapping lookup retains ordinary `KeyError`. Do not import arbitrary
  modules, evaluate strings or guess a different model when lookup fails.
- Include registered names in `dir(d.fit)` and give generated callables useful
  signatures, names and docstrings. Dynamic methods must remain discoverable.
- Implemented: saved shortcut callables retain the owner and registry key, resolving
  that key at invocation so explicit registry replacement is visible. Cache no
  mutable model instances. Define and test missing/replaced-entry behaviour.

### Canonical spelling example

The physical name is **BlochÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“GrÃƒÆ’Ã‚Â¼neisen**. Use `bloch_grueneisen` for the ASCII
registry key, mapped to the existing `BlochGrueneisen` class. The earlier
`bloch_gruneisen` conversation spelling was a typo, not an approved alias.
The existing callable `blochGrueneisen` remains unchanged.

Physics source: Vedeneev et al., *Physical Review B* **51**, 16380 (1995),
[doi:10.1103/PhysRevB.51.16380](https://doi.org/10.1103/PhysRevB.51.16380).

Other expected examples, subject to the complete catalogue audit:

| Registry key | Existing class |
| --- | --- |
| `linear` | `Linear` |
| `power_law` | `PowerLaw` |
| `bloch_law` | `BlochLaw` |
| `bloch_grueneisen` | `BlochGrueneisen` |
| `fuchs_sondheimer` | `FluchsSondheimer` (legacy Python spelling retained) |
| `rsj_noiseless` | `RSJ_Noiseless` |

Before registering the full catalogue, review scientific spellings, acronyms and
existing inconsistencies against model references. Use explicit registration
names for corrections without renaming legacy classes/functions as incidental
cleanup. Record source-backed mappings and decide any intentional aliases openly.

### Decorator integration

Extend `make_model` in `Stoner/analysis/fitting/models/__init__.py`. Its current
implementation returns a class internally named `_ModelDecorator`; derive the
registry key from the decorated function's public name, not that implementation
class name. Register the generated class by reference so later `.guesser` and
`.hinter` changes remain visible. Preserve current decorator return values.

Proposed forms:

```python
@make_model
def custom_response(x, amplitude, decay):
    return amplitude * np.exp(-x / decay)

# Optional extension when an explicit name is needed:
@make_model(name="custom_response")
def response_function(x, amplitude, decay):
    return amplitude * np.exp(-x / decay)

# For a handwritten Model subclass:
@register_model(name="custom_response")
class CustomResponse(Model):
    ...
```

These are independent examples: registering all three together should raise a
duplicate-name error. The bare `@make_model` spelling must remain supported.
Expose `register_model` from the fitting-model package; both decorators use the
same validation and registration functions.

Register built-in classes through that machinery with a reviewed explicit
catalogue or decorators on their definitions. Do not register every callable in
model modules: they contain raw functions, imported base classes and helpers.
Do not automatically import every model from third-party packages. Arbitrary
unregistered models remain usable through explicit algorithm methods.

## 4. Implementation structure and ownership

Suggested layout:

- `Stoner/analysis/fitting/registry.py`: registry, name normalisation, validation
  and registration decorator; independent of `Data` and the concrete model modules.
- `Stoner/analysis/fitting/accessor.py`: `DataFitter`, declared algorithm methods,
  generated shortcut callables, `models` view and completion support.
- `Stoner/core/data.py`: cheap `fit` property with a local accessor import,
  following the current `plot` pattern.
- `Stoner/analysis/fitting/models/__init__.py`: augment `make_model`, export public
  registration helpers and initialise the built-in catalogue without import cycles.

Keep numerical implementations in `fitting/functions.py` initially; explicit
accessor methods delegate to them. Avoid copying five implementations or routing
them back through compatibility aliases. Preserve subclass behaviour deliberately:
audit existing overrides and document the chosen extension point before removal
of old methods. Avoid import-time model construction, guesses or numerical work
beyond existing module initialisation.

Accessors hold the live owner; measurements, exclusions and metadata remain owned
by the existing storage layer. `Data` copy/clone/pickle must not capture a bound
accessor, generated closure or mutable model. Test those operations after accessor
lookup and use, including registration of a locally defined custom model.

## 5. Migration and consumers

Recommended staged compatibility policy, pending confirmation before coding:

1. Add the registry and accessor while direct `Data` fitting methods still work.
   Keep the first addition free of new deprecation warnings until consumers move.
2. Migrate package cross-calls, tests, documentation, samples and scripts. Keep
   numerical callables and model import paths stable. Use shortcuts in beginner
   examples and explicit algorithm calls where algorithm choice is the lesson.
   Rename coordinate keywords at migrated call sites: for example,
   `d.lmfit(Linear, xcol=0, ycol=1)` becomes `d.fit.lmfit(Linear, x=0, y=1)`.
   Keep the old coordinate names on direct function calls and compatibility
   wrappers; translate them explicitly when forwarding to the new accessor.
3. Add thin top-level forwarding methods with `DeprecationWarning` and useful
   `stacklevel` once their scope and removal release are agreed. Change dynamic
   attachment explicitly so `class_modifier` cannot overwrite these wrappers.
4. Remove the aliases only in an explicitly selected release, with release notes,
   API-inventory allowances and completed migration evidence. No removal version
   or deadline is set by this plan. Do not build a permanent duplicate API layer.

Audit more than direct `.lmfit(...)` calls: unbound `Data.lmfit(d, ...)`, imported
functions, string method names, subclass overrides and wrapper metadata can also
be consumers. Known internal cross-calls include extrapolation in `filtering.py`
and hysteresis helpers in `analysis/utils.py`. Update `make_model` documentation
and references to the old fitting method paths.

### Folder bulk operations

Implemented through `Stoner.folders.each.FitProxy`: callable lmfit default,
explicit algorithms, model shortcuts, completion and a read-only model view.
`Item.__call__` and `iter` validate fitting paths and dispatch them by dotted
name. Existing folder ordering, member updates, exception tuples and `_return`
metadata conventions are retained. Fitting uses the existing serial/thread/process
policy, independently of plotting's serial-only requirement.

Each bulk shortcut invocation captures its selected `ModelRegistration` and sends
it with the task. The worker resolves the member's accessor locally and uses that
definition without installing it in the global registry. The existing serializer
has been tested with local `make_model` classes, factories and replacements of
built-in names under fresh spawned processes. Factory construction is per member.
Explicit configured models are transferred through the existing argument path.

Process-pool fitting preflights the selected worker callable for serialisation;
failure gives a diagnostic rather than a serializer-triggered serial fallback.
The operation closes/joins its own pool on completion and terminates/joins it on
failure or explicit early iterator closure. Other pools are untouched.

New `fit` properties take precedence over attribute-based column lookup. Document
that a column named `fit` remains accessible by `d.column("fit")` and explicit
selectors. Do not reserve model names as top-level `Data` attributes.

## 6. Documentation deliverables

- [x] Document `Data.fit`, `DataFitter`, explicit algorithms, registry inspection,
  registration helpers, naming, collision rules and generated shortcut signatures.
- [x] Add a beginner example using `d.fit.linear()` with assigned roles and show
  `d.plot.annotate_fit(Linear)` separately. Explain when an import is still needed.
- [ ] Add a physics example using `d.fit.bloch_grueneisen(...)`, checked against
  the canonical model and realistic initial parameters/units.
- [ ] Explain explicit algorithm choice, configured/composite models, parameter
  bounds versus row-selection bounds, uncertainties and existing return options.
- [x] Document `Stoner.enums.FitOutput`, its string values and the output support
  matrix; demonstrate enum calls while explicitly retaining string compatibility.
- [x] Document `x`/`y` as column selectors consistent with plotting; show keyword
  migration from `xcol`/`ycol` and the unchanged positional order and role defaults.
- [ ] Add a runnable `@make_model` registration example with `.guesser`/`.hinter`,
  explicit name override and collision diagnostics.
- [x] Update the fitting learning path (`curve_fitting`, `fitting_simple`,
  `fitting_options`, `fitting_methods`), model/decorator docs and migration guide.
- [x] Audit all `doc/samples`, `scripts` and internal consumers using the
  tracked file list; retain direct-method tests as compatibility coverage. Preserve generated/historical content and `doc/plot_cache`.
- [x] Give dynamic built-in shortcuts stable generated documentation and inventory
  targets through the registry; do not rely only on runtime `__getattr__` for Sphinx.
- [x] Add folder examples, compatibility/removal notes and release notes.

Follow the Google-style/British English docstring standard. Retain cached figures
for `READTHEDOCS=True` builds; deliberately review any cache refresh from normal
builds. Generated `doc/classes` pages are not hand-maintained source. If stale
pages interfere, preserve them in an ignored run directory and regenerate.

## 7. Testing and validation plan

### Registry and decorators

- [x] Test snake_case conversion, acronyms, underscores, explicit overrides,
  physics-correct catalogue names and exclusion of raw/imported helper callables.
- [x] Test invalid/private/reserved names, duplicate errors, same-object
  re-registration, explicit replacement and registry isolation between tests.
- [x] Test bare `@make_model`, named decorator form, `.guesser`, `.hinter`,
  explicit subclass registration and factory result validation.
- [x] Test cold import and representative alternate import orders, without circular
  imports, duplicate built-in registration or model construction during lookup.

### Accessor and fitting parity

- [x] Test attribute lookup, `dir`, `help`/docstrings/signatures, mapping discovery,
  unknown-name errors and reserved-method precedence without numerical side effects.
- [x] Compare shortcuts against explicit `fit.lmfit` calls using deterministic
  datasets; verify independently created model state and current-owner reads.
- [x] Test positional and keyword `x`/`y` selectors, omitted role defaults and
  translation to each existing algorithm. Reject `xcol`/`ycol` on the accessor,
  including mixed old/new keywords; preserve them on legacy direct interfaces.
- [x] Exercise every explicit algorithm, including current `polyfit` behaviour.
- [ ] Verify initial guesses, parameter constraints, prefixes, model instances,
  composite models, callable models and all supported result/output modes.
- [ ] Verify enum/string output parity for every supporting algorithm and output
  mode, including multi-fit calls and folder dispatch. Cover invalid/wrong-domain
  values, defaults, direct function calls and configuration/serialisation paths.
- [ ] Preserve masks, uncertainties, duplicate-header/role selection, multiple
  predictors/dependent channels, fitted/residual column insertion and metadata.
- [x] Cover saved shortcuts, explicit registry replacement, clone/copy/pickle,
  subclass extension points and a column named `fit`.
- [ ] Test compatibility forwarding, warnings and unbound calls once introduced.

### Integration and documentation

- [x] Test serial and parallel folder calls, fresh-worker registration, errors,
  custom models and per-member fit metadata/results.
- [x] Run existing fitting tests, the new accessor/registry tests, relevant column
  selection tests and all migrated documentation examples; separate network tests.
- [x] Run the full suite for the shared API change in an environment verified
  against `tests/test-env.yml`, with headless Matplotlib/Qt where needed.
- [x] Build cached Sphinx HTML and audit against the reviewed platform warning
  baseline. Compare public inventories; allow removals only at the agreed stage.
- [x] Serve generated HTML alone over localhost; inspect beginner instructions,
  model catalogue, shortcut API pages and migration links.
- [ ] Before release/removal, verify hosted Python/platform, lower-dependency and
  installed-distribution checks on the exact pushed commit. Local results do not
  establish hosted results or physical instrument/data compatibility.

## 8. Bounded implementation batches and exit gates

| Batch | Deliverable | Exit gate |
| --- | --- | --- |
| 1 | Registry contract, complete model-name audit, `make_model` extension and built-in registration | Registry/decorator/import tests pass; canonical names and reserved names reviewed. |
| 2 | `DataFitter`, `Data.fit`, explicit methods, dynamic shortcuts and `FitOutput` | Signatures/discovery, enum/string parity and algorithm tests pass; owner/return contracts unchanged. |
| 3 | Folder proxy and internal/package consumer migration | Serial/parallel/custom-model tests and affected scientific regressions pass. |
| 4 | Samples, scripts, guides, API reference and compatibility wrappers | Example suite, full suite, Sphinx warning/inventory audit and visual checks pass. |
| 5 | Later removal of deprecated direct methods, if approved | Removal release explicitly chosen; consumers migrated; exact-commit hosted release gates pass. |

Batches 1-3 and the batch 4 documentation/sample/script migration are complete.
Compatibility warnings and removal policy remain deferred; direct fitting methods
remain available without new deprecation warnings.

No batch is complete solely because its code exists. Update the checklist and
handover with validation evidence, unresolved failures and the next bounded step.

## 9. Handover and decisions to settle

Updated: 2026-10-30.

Current state: registry/decorator foundation, `FitOutput`, `DataFitter`,
`Data.fit`, `folder.each.fit` and dynamic model shortcuts are implemented.
The registry holds 28 explicitly selected built-in
classes; custom classes and explicitly named factories share the same controlled
registration path. Named decorator forms and explicit `replace=True` are supported.
The callable is stored by reference, and provenance records the original function
for `make_model` classes without changing their legacy metadata prefix.

`FitOutput` is exported from `Stoner.enums`; direct fitting entry points unwrap
its members to legacy strings. Shared enum keys replace duplicated standard
output labels in dictionary dispatch. Existing string dispatch, including
`curve_fit` report-attribute selectors, remains available. Wrong-domain enums
are rejected before fitting. Polynomial fitting is unchanged.

The folder batch's 32 new tests passed on Windows Conda `py314` (Python 3.14.7),
with the environment verified against `tests/test-env.yml`. The existing
accessor/registry/output and scientific regressions also run in the full suite.
The full suite passed 1,759 tests with two workers, excluding network/OCR tests;
the network fitting example passed separately (one test). OCR and hosted CI were
not run. Logs and JUnit results are under ignored `maintenance/runs/folder-fit/`.
The final cached Sphinx build succeeded with 62 warnings, with no unexpected
warnings or missing dynamic `Data` methods against the reviewed Windows baseline.
The bulk-fitting guide and existing instrument-data folder example are updated.
No plot-cache files were regenerated in this batch. The earlier three new
`fit_accessor` assets are retained. Whitespace and plan-link checks pass.

The accessor delegates directly to numerical functions rather than legacy Data
method overrides. No package subclasses overriding those fitting methods were
found. Customise accessor behaviour through a DataFitter subclass returned by an
overridden Data.fit property; callable and shortcut forms honour its lmfit method.
Factory shortcuts call zero-argument factories. Constructor configuration remains
explicit through `model_registry.create(...)` and an explicit algorithm call.

The new named-coordinate parity test exposed a pre-existing differential-evolution
failure when its X selector resolved to a one-item list. Fit-statistics evaluation
now uses the shared single-column resolver and numerical column read. Direct and
accessor calls with named coordinates both pass.

Internal consumers in extrapolation and the two hysteresis fitting helpers now
use `Data.fit`. Source audit leaves NumPy/SciPy calls and accessor-to-function
delegation intact. New regression tests forbid legacy direct calls in those
helpers and verify the retained SQUID fixture and synthetic extrapolation.
The shared fitting preparation guard now distinguishes scalar integer Y-column
selectors from array containers, fixing an older scalar-selector failure.
The focused folder guide and sample are migrated; the subsequent broader
sample/script sweep is also complete, as recorded below.

Known pre-existing boundary: repeated-initial-guess lmfit DATA output expects
prefix/vary metadata normally supplied by INI setup. Parity tests supply those
fields. This batch does not repair or change that legacy path.

Documentation/sample/script migration is complete: the source audit found 72
fitting calls across tracked samples and scripts, all using `.fit` with accessor
coordinate keywords. Guides, fitting API cross-references, TeX fitting snippets,
release notes and migration instructions are updated. The numerical algorithm
reference now provides working links to the full options from each accessor
method and explains the owner and coordinate-keyword translation. The 83 offline documentation
examples and the network example passed. Interactive/file-dependent analysis
scripts were syntax-checked and audited, not run against laboratory data.
The full local suite passed 1,759 tests (network/OCR excluded), and the
network fitting example passed separately. No plot-cache assets were refreshed.
The rendered migration guide and its registration-guide link were checked over
localhost. The final clean cached Sphinx build passed with 62 warnings and no
unexpected warnings, missing dynamic Data methods or removed API objects; all
five accessor and numerical algorithm inventory targets are present. Whitespace
checks pass. OCR and hosted CI were not run. Current validation logs are under `maintenance/runs/fit-migration/`.

Next bounded batch: agree the compatibility warning timing and removal policy. The registry remains process-local; bulk fitting sends
only the chosen definition with each call. Choose warning timing and a removal
release before introducing deprecations. Preserve the pending `annotate_fit` edits.
Do not expand to other accessors or
redesign fitting results/algorithms without a separate agreed scope.
