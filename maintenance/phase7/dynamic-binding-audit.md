# Data dynamic-binding audit

Audited 2026-09-12. No runtime package code changed in this batch.

## Results

All 85 functions attached from the seven Data/PlotMixin source modules are
present as the original function objects. Class and bound-instance signatures,
docstrings and annotations are preserved. Every method appears in both dir(Data)
and dir(Data()), and pydoc/help includes its source documentation. No collisions
occur between the selected source modules, and none replaces a method defined
in the Data class body. Data uses adaptor=None and no_long_names=True; the
image adaptor/wrapper mechanisms are not part of this binding path.

The checks pass on Python 3.14 both normally and with READTHEDOCS=True, and
on Python 3.11. The 85 names exactly match the latest verified Sphinx inventory
in maintenance/runs/phase7-tdi2-docs-audit.json. All existing annotations resolve
without get_type_hints raising an exception; see the placeholder caveat below.

## Source-level gaps, not binding defects

1. Annotation coverage: only 14/85 methods annotate every bound parameter and
   the return value. The remaining 71 have no return annotation. This measures
   presence, not correctness. Counts by module:

   | Module                            | Methods | Fully annotated |
   | --------------------------------- | ------- | --------------- |
   | Stoner.analysis.columns           |      10 |               9 |
   | Stoner.analysis.features          |       2 |               0 |
   | Stoner.analysis.filtering         |      12 |               0 |
   | Stoner.analysis.fitting.functions |       6 |               0 |
   | Stoner.analysis.functions         |       7 |               5 |
   | Stoner.core.methods               |      28 |               0 |
   | Stoner.plot.functions             |      20 |               0 |

2. Incorrect existing annotation: Stoner.analysis.functions.threshold declares
   -> Data, but a normal scalar crossing returns numpy.float64. A live probe
   with Data(np.column_stack((np.arange(5.), np.arange(5.))), setas='xy') and
   threshold(1.5) returned approximately 1.5, not a Data object. Other branches
   return arrays or an empty list. Its return annotation needs a review of all
   branches; runtime behaviour should not be changed to satisfy the annotation.

3. Runtime type aliases: Stoner.tools.typing deliberately constructs placeholder
   Data, Setas, ImageArray and ImageFile classes outside TYPE_CHECKING. Thus
   typing.get_type_hints(Data.add)['datafile'] is Stoner.tools.typing.Data,
   not the actual public Data class. This limits runtime annotation introspection;
   it does not prevent method calls. The TYPE_CHECKING image import also uses
   '..image.core', while the tracked package is Stoner/Image/core.py. That is
   a case-sensitive-path concern for static checking; no external type checker
   or Linux environment was run in this audit.

4. class_modifier's own docstring has stale argument names: cls is not a direct
   parameter; proxy should be proxy_cls; RTD_Restrictions should be
   RTD_restrictions; alias is undocumented. These do not affect binding.

## Evidence and scope

Reproduction script: maintenance/phase7/audit-binding.py. It inspects all 85
methods without executing their numerical/plotting operations. Run it through
conda run -n py314 python, passing an output JSON path. Set READTHEDOCS=True
before starting Python for the documentation-mode variant; remove it for the
normal variant. Repeat with -n py311-minimal for lower-environment checking.

Reports: maintenance/runs/phase7-binding-normal.json,
maintenance/runs/phase7-binding-rtd.json and
maintenance/runs/phase7-binding-py311.json. Each records method names, sources,
signatures, discovery/help checks and missing annotation names.
The threshold value probe above was a separate small runtime check.

This is an audit, not a comprehensive static-type or return-behaviour audit.
No binding refactor is justified by the findings. Proposed follow-up: review
and correct existing annotation errors and helper documentation before adding
annotations to the remaining methods. Changes require a separate decision.
The previously completed 456-test suite was not repeated for this audit.
