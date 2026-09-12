# Phase 6 fifth batch: fitting models and Airy limit fix (2026-09-12)

## Changes

Corrected the model descriptions across three families and the Airy helper docstring:

- `Lorentzian_diff` now distinguishes the underlying Lorentzian area from the
  derivative peak height, gives the width convention and displays the derivative
  equation with valid RST mathematics.
- `BlochLaw` and `blochs_law_bulk` now display the expression actually evaluated
  by the code, returning magnetisation rather than a normalised ratio. They
  describe units, parameter hints and the bulk low-temperature restriction.
- `Ic_B_Airy` and `ic_B_airy` now have valid mathematics and parameter descriptions,
  matching the corrected zero-field limit.

The class descriptions distinguish fitting parameters from constructor arguments.
Existing example directives and their cache output names are retained. The user
explicitly authorised fixing the discovered Airy defect during this batch.
The only runtime change is the body of `ic_B_airy`; other models and parameter
hints are unchanged (`batch5-change-check.json`).

## Airy numerical fix

Previously, the helper used one as the small-argument replacement for `J1(u) / u`,
then multiplied by two. For `Ic0=2` and arguments `[0, 1e-6, 1e-4]`, it returned
`[4, 4, 1.9999999974999985]`. Zero argument also emitted an invalid-division warning
because NumPy evaluates the unused division branch.

The corrected code uses `np.isclose(u, 0, atol=1e-5, rtol=0)` and `np.where` to
select a factor of one. It first substitutes a safe denominator for the near-zero
arguments, so both evaluated branches are safe. The result at the centre is now
`Ic0`. The explicit absolute tolerance retains the small-argument scale; its
boundary is now inclusive. The ordinary nonzero Bessel expression is preserved.

`batch5-source-check.json` retains the clearly labelled pre-fix reproduction.
The new regression module checks scalar and two-dimensional array inputs, both
signs around zero, the branch boundary, output shape, nonzero model evaluation
and the absence of divide/invalid floating-point errors.

## Validation

The initial regression run failed both central-limit cases on the original code;
the two nonzero model cases passed. After the fix, all five final regression cases
pass on Python 3.14 and the Python 3.11 lower-dependency environment. The full
fitting selection passes nine tests on Python 3.14, and the Airy, Bloch and
Lorentzian example selection passes all three scripts. The complete repository
suite was not repeated; hosted CI remains a separate validation boundary. The
run identifiers and JUnit counts are recorded in `batch5-tests.json`.

Focused probes also verify Lorentzian and Bloch parameter names and independent
variables, and the bulk Bloch zero-temperature result. The final AST comparison
confirms the Airy helper body is the sole executable package change.

The clean RTD-mode HTML build uses Python 3.14.7 and Sphinx 9.1.0, matching the
previous batch. Warnings fall from 443 to 438, removing five markup diagnostics;
other warning categories are unchanged. Categories and API coverage are in `batch5-after.json`.
All five primary classes and 85 dynamic Data methods remain documented, with no
lost inventory entries relative to batch 4. All 243 plot-cache hashes remain
unchanged (`batch5-cache-check.json`). No warning suppression was added.

Reproduce from the repository root in the activated documentation environment:

```powershell
$env:READTHEDOCS = 'True'
python -m sphinx -b html -E doc maintenance/runs/phase6-batch5-final -w maintenance/runs/phase6-batch5-final-warnings.log
python maintenance/audit-docs.py maintenance/runs/phase6-batch5-final maintenance/runs/phase6-batch5-final-warnings.log maintenance/phase6/batch5-after.json --compare maintenance/runs/phase6-batch4-final
python -m pytest tests/Stoner/analysis/fitting
```

## Next proposed documentation batch

Investigate the 57 missing autosummary stubs in the module indexes. Distinguish
incidental imported names from intentional public re-exports, then correct a
focused group of index directives or generation settings. Validate with fresh
generated API sources and the inventory audit so that real public documentation
is retained. Third-party markup and duplicate descriptions remain separate
warning categories. Phase 6 remains in progress.
