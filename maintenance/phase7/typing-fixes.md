# Annotation and binding-documentation fixes

Implemented 2026-09-12 following the dynamic-binding audit and maintainer approval.

- threshold accepts a scalar or sequence of thresholds. Its return annotation
  now covers floating scalars, ndarray subclasses (including DataArray), and
  lists for no-crossing results. Complete-row output is DataArray, not Data.
  Runtime checks corrected an initial test assumption that a singleton crossing
  with all_vals=True must remain an array: it can already return a scalar.
  The numerical executable body is AST-identical to HEAD.
- Replaced runtime placeholder classes in Stoner.tools.typing with module-aware
  ForwardRef objects. They resolve to the actual Data, Setas, ImageArray and
  ImageFile classes after package initialisation without eager circular imports.
  Qualified reference expressions avoid a same-name resolution cycle observed
  with Python 3.14 deferred function annotations. Corrected the TYPE_CHECKING
  import to the tracked Image package capitalisation.
- Corrected class_modifier's documented parameters and return type, and documented
  adaptor=None and alias. The binding implementation is unchanged.

## Validation

- Python 3.14: new typing/return regressions plus the existing threshold test,
  three passed, one warning, 8.16 seconds
  (`maintenance/runs/20260912-231239-focused-a512c751`).
- Python 3.11 minimal: the same three passed, 18 warnings, 10.59 seconds
  (`maintenance/runs/20260912-231251-focused-3e72bdb6`).
- The full 85-method signature/docstring/annotation/discovery audit passes again,
  with no annotation-resolution errors
  (`maintenance/runs/phase7-binding-fixed-normal.json`).
- All 458 tests collect successfully in 11.21 seconds
  (`maintenance/runs/20260912-231352-collect-38333d59`).
- Fresh cached Sphinx build succeeds with the existing 106 reviewed Windows
  warnings; final inventory audit is recorded alongside the build in
  `maintenance/runs/phase7-typing-docs*`.
- Executable-body comparison: `maintenance/runs/verify-threshold-body.py`.

The preceding full suite had 456 passes. It was not rerun for these annotation
and documentation changes; focused regressions, collection and import/binding
checks were used. No third-party type checker was run. Changes are uncommitted;
remote CI has not validated this batch. Broader annotation coverage remains
future work.

## Small column follow-up (2026-09-12)

The sole remaining public-column annotation gap was std's return type. Added
np.floating and corrected its column/uncertainty docstring copy-and-paste errors.
No calculation code changed. A live result matches the resolved return type.
Existing column regressions pass: 22 on Python 3.14 (one warning, 5.66 seconds,
maintenance/runs/20260912-231839-focused-876b78db) and 22 on Python 3.11
(18 warnings, 8.24 seconds, maintenance/runs/20260912-231850-focused-329fe4b6).
The full suite and Sphinx build were not repeated for this small follow-up.
All ten attached column methods now have complete annotation coverage; this
does not assert that every pre-existing annotation has been audited for correctness.
