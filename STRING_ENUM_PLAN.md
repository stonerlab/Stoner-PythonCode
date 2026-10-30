# String-valued API options and enums

Date: 2026-10-29. Target checkout: `devel`.
Status: source-backed audit and implementation plan. `Stoner.enums.FitOutput`
is implemented; other enum groups remain planned. The inventory is representative,
not exhaustive.

## Direction

Provide string enums for public arguments with a defined set of named choices.
Use Python's standard-library `enum.StrEnum` on the project's Python 3.12+ floor.
Keep existing supported string calls working. Improve discovery, annotations,
documentation and consistent naming without changing scientific results or
silently replacing existing return types.

Start with fitting outputs alongside [FIT_ACCESSOR_PLAN.md](FIT_ACCESSOR_PLAN.md).
Expand in bounded groups across the package after auditing each actual contract.
This complements [DATA_ACCESSOR_PLAN.md](DATA_ACCESSOR_PLAN.md); it does not depend
on introducing all the proposed accessors.

## Proposed public conventions

- Put reusable public enums in a lightweight `Stoner.enums` module, independent
  of `Data`, plotting, models and optional dependencies. Explicitly export this
  module from the package. Avoid a new third-party dependency.
- Use descriptive PascalCase enum classes and UPPER_SNAKE_CASE members. Prefer
  `FitOutput.REPORT` to an ambiguous package-wide `Output.REPORT`.
- Use explicit string values matching existing accepted tokens. Do not use
  automatic values where they would accidentally rename a public token.
- Annotate closed-choice arguments with their specific enum and accepted legacy
  string forms. Examples favour enum members; document ordinary strings too.
- Share one enum wherever the meaning and accepted choices really are the same.
  Do not merge unrelated domains because both have an argument named `mode`,
  `format`, `kind` or `output`.
- Initially retain existing default values and valid-string behaviour. Replacing
  a default with an enum also affects signatures/introspection and should be an
  explicit reviewed change, not an incidental part of adding enum support.
- Normalise at a clear API boundary and use one dispatch definition per domain.
  Public wrappers and lower-level entry points must agree, including direct
  function calls which bypass an accessor.
- Pass `.value` where a plain string is needed by a third-party API, exact-type
  check or serialisation boundary. Audit actual consumers instead of assuming
  that string-subclass behaviour handles every path.
- Do not coerce returned data into enum objects as part of this work. This plan
  concerns choice arguments; output-schema changes require separate review.

The first implemented enum:

```python
from enum import StrEnum

class FitOutput(StrEnum):
    FIT = "fit"
    REPORT = "report"
    ROW = "row"
    FULL = "full"
    DATA = "data"
    DICT = "dict"
```

Planned user calls:

```python
from Stoner.enums import FitOutput, TextFormat, NumberFormat

report = d.fit.linear(x="Field", y="Signal", output=FitOutput.REPORT)
report = d.fit.linear(x="Field", y="Signal", output="report")
text = d.format("Linear:slope", fmt=TextFormat.LATEX,
                mode=NumberFormat.ENGINEERING)
```

These calls illustrate the later accessor and formatting-enum interfaces.
`FitOutput` works with `Data.fit` and direct fitting methods such as `d.lmfit`;
`TextFormat` and `NumberFormat` remain unimplemented. Not every
algorithm supports every conceivable output.

## Initial source audit and candidate groups

| Proposed enum/domain | Observed choices and source | Scope and compatibility issues |
| --- | --- | --- |
| `FitOutput` | `fit`, `report`, `row`, `full`, `data`, `dict`; `Stoner/analysis/fitting/functions.py` and result properties in `classes.py` | Highest priority. Audit algorithm-by-algorithm and single/multiple fit returns. `polyfit` has no `output` selector: do not invent one. Common labels do not imply identical algorithm-native reports or full-result tuples. |
| `TextFormat` | `text`, `latex`, `html`; `Stoner/tools/formatting.py` and `Stoner/core/methods.py::format` | Shared text-rendering domain for `format_error`, `format_val` and `Data.format`. Preserve `latex` boolean precedence and existing fallback behaviour until explicitly reviewed. |
| `NumberFormat` | `float`, `eng`, `sci`; `Stoner/tools/formatting.py` | Proposed members `FLOAT`, `ENGINEERING`, `SCIENTIFIC`, retaining those exact string values. Reuse for `Data.format` and `Data.plot.annotate_fit`. |
| `PlotKind` | `auto`, declared XY/field/statistical kinds; `Stoner/plot/accessor.py` | Derive validation and documented choices from one definition. Internal inferred kinds such as `xy` are not necessarily accepted public `kind` values. Preserve named plotting methods and folder restrictions. |
| `MetadataOutput` | `dict`, `list`, `frame`, `data`, `array`, `smart`; `Stoner/folders/metadata.py::MetadataProxy.slice` | Separate from `FitOutput`. Current code also accepts Python types and lowercases strings; preserve accepted type alternatives and `values_only` default selection. Verify the actual implementation rather than copying inconsistent docstring type names. |
| `PandasFormat` | `legacy`, `plain`; `Stoner/core/storage.py::DataStorage.to_pandas` | Share between storage packages and the public `Data.to_pandas` wrapper. This is table representation, not text formatting. |
| `MaskedExport` | `nan`, `raw`; the same storage export path | Keep distinct from the boolean `masked` parameter of `to_numpy`; identical argument names do not establish a common domain. |
| `RowIndexPolicy` | `require_range`, `discard`; `Stoner/core/storage.py::DataStorage.from_pandas` | Reuse in `Data.from_pandas`. Retain explicit treatment of non-positional pandas indices. |
| `BinningMode` | Branches accept case-insensitive prefixes starting `lin`, `log`, `spa`; `Stoner/analysis/filtering.py::make_bins` | Current defaults differ between `bin` (`log`) and `make_bins` (`lin`). Do not change defaults while adding enums. Equal-population/spacing mode support depends on how bins are specified; audit before choosing a member name/value. |
| `DeduplicateAction` | `average`, `median`, `first`, `last`; `Stoner/analysis/filtering.py::deduplicate` | Distinct contract from approximate duplicate removal, even where labels overlap. |
| `DuplicateStrategy` | `keep first`, `average`; `Stoner/core/methods.py::remove_duplicates` | A member such as `KEEP_FIRST` can map to the existing space-containing value. Do not silently replace the token with `keep_first` or equate all duplicate-removal algorithms. |
| `OutlierAction` | `delete`, `mask`, `mask row`, or a callable; `Stoner/analysis/filtering.py::outlier_detection` | Enum covers named actions only. Preserve callback support, callback arguments and masking/deletion distinctions. |

The audit found no existing Python `Enum`/`StrEnum` declarations in `Stoner` at
review time. The `Enum` text recognised by metadata type parsing is not a Python
public choice enum and should not be changed merely because its name matches.

## Inconsistencies to resolve deliberately

1. Separate the human-friendly member name from its wire/string value. For example,
   `NumberFormat.ENGINEERING` should retain `"eng"`; a new enum should not force
   every persisted configuration or existing script to change spelling.
2. Record accepted aliases, case folding, prefix matching and type alternatives
   for each domain. Do not introduce universal `.lower()` normalisation, or
   replace permissive existing behaviour with strict enum construction silently.
3. Derive documentation of supported fitting outputs from a reviewed support
   matrix. Existing dictionary dispatch and attribute-based dispatch should not
   drift independently. A shared `FitOutput` standardises labels without erasing
   algorithm-specific result content.
4. Folder metadata output currently accepts more choices than its error message
   lists, and its docstring type alternatives need checking against dispatch.
   Correct those messages/docs in that bounded migration with regression coverage.
5. Decide whether a wrong-domain enum should be rejected explicitly. Different
   string enums can share equal string values, so ordinary equality or enum
   construction alone does not enforce semantic type separation. Recommended:
   accept the declared enum or a plain legacy string, and reject an enum from
   another domain, with a documented exception if existing API usage requires it.
6. Prefer useful `ValueError`/`TypeError` diagnostics for new validation, but audit
   existing exception classes and validation timing before changing them. An
   invalid option should not partially mutate measurements as a side effect of
   introducing earlier validation.

## Planned image accessor choices

The image sequence is [plot](IMAGE_PLOT_ACCESSOR_PLAN.md), then
[ROI](IMAGE_ROI_ACCESSOR_PLAN.md), then [fit](IMAGE_FIT_ACCESSOR_PLAN.md).
Plotting establishes `CoordinateMode` (native coordinates or explicit pixel
indices) for all three. Fitting adds `BackgroundReference` (zero or fitted-region
mean, with a finite numeric reference as a separate accepted argument type).
`CoordinateMode` is now implemented with native/pixel values and shared by the
image plotting grid resolver; BackgroundReference remains planned. Reuse `FitOutput`; do not add a
competing image-output enum or enums for extensible model registry names. Test
string parity, wrong-domain rejection and serialised values in those batches.
Freeze accepted domains/defaults and stored strings at the
[release phase gates](RELEASE_0_12_PLAN.md); beta compatibility and RC full freeze
apply to enums and selection/provenance records as well as method signatures.

## Domains to keep extensible

Do not replace arbitrary strings with enums for column labels/selectors,
metadata keys, prefixes, paths, regular expressions, Matplotlib format strings
such as `"ro-"`, styles, user text or template expressions.

Registered fitting model names and file loader/saver names are open registries;
enums must not prevent custom registrations. File extensions, image codecs,
interpolation methods and optimiser methods may be supplied or extended by
dependencies. Audit ownership before introducing any bounded convenience enum,
and continue forwarding valid external strings/callables where supported.

## Implementation batches and validation

- [ ] Complete the public-choice inventory using actual signatures, dispatch,
  tests and configuration consumers across `Data`, folders, images, formats and
  tools. Mark each domain closed, extensible, callback-capable or internal-only.
  This initial audit is not a claim of complete package coverage.
- [ ] Introduce the lightweight enum module and `FitOutput` with the fit accessor.
  Validate every member against each algorithm's real output contract and share
  dispatch/normalisation across accessor and direct entry points.
- [ ] Add text/number formatting enums as one batch, including annotation callers.
- [ ] Add storage representation/policy enums and folder metadata output enums
  in separate batches, preserving interchange and type-alternative behaviour.
- [ ] Review plotting and then cleaning/binning domains separately; do not fold
  unrelated scientific or default-behaviour changes into enum adoption.
- [ ] Test enum/string parity, defaults, aliases, invalid values, wrong-domain
  enums, signatures and exception behaviour. Cover dictionaries, match cases,
  attribute lookup, direct function calls and third-party forwarding as used.
- [ ] Test config round trips, JSON/metadata/pickle handling and folder worker
  transfer where these options actually cross those boundaries. Retain plain
  persisted string values and stable enum import paths.
- [ ] Document members and accepted legacy values together. Update representative
  examples to demonstrate enum discovery without making strings appear unsupported.
- [ ] Run focused scientific regressions, affected examples, then the full suite
  for shared changes; use the repository Sphinx warning/inventory checks and
  exact-commit hosted gates before release. Planning-only edits need structural
  document validation, not a claim that future enum behaviour has passed tests.

No source migration is complete merely because `StrEnum` compares equal to a
string. Each bounded batch must establish compatibility at its actual consumers.
