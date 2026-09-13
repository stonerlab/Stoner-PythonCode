# Phase 7 bounded final review

Closure update (2026-09-13): Phase 7 is complete. All five hosted test-matrix
jobs, final result publishing, lower-dependency CI and package validation passed
at `d5c43882c`. See [the closure evidence](ci-fixture-paths.md#hosted-closure-2026-09-13).
This supersedes the commit/push and validation handoff below, which records the
earlier review state. Deferred work remains outside the completed phase's scope.

Reviewed 2026-09-13 on `stable`. This is a source and evidence review, with no
runtime changes. The selected maintenance batches and final triage are complete
locally. Phase 7 remains in progress until the accumulated changes have been
reviewed into cohesive commits and validated by hosted CI.

## Remaining format questions

A case-insensitive TODO/FIXME search of package Python finds two actual tasks:

- `Stoner/formats/data/generic.py`: retain the legacy metadata deepcopy
  workaround. Its re-inference behaviour is understood, but the original
  LabVIEW TDI 1.5 writer's rationale is not established. The authorised
  `copy_into` type-preservation fix is complete and independently tested;
  it does not change general deepcopy or remove this loader workaround.
  See `metadata-copy-review.md` and `tdi2-loading.md`.
- `Stoner/formats/data/instruments.py`: SPC's alternative multiple-x/y-curve
  layout remains an unimplemented feature. Require representative exports,
  a format specification and expected axes/data before selecting an extension.
  The completed truncation/error-translation fix does not implement that layout.

The `todo` word in a `setas.py` exception message is not a maintenance marker.
MAXIMUS's former markers are already documented in `maximus-limitations.md`;
multi-region support remains deferred under the maintainer's evidence rule.

## Compatibility and public API

`Stoner/compat.py` is actively imported across core, folders, images, loaders,
tools and tests. It contains useful shared type/path/encoding helpers and
optional dependency handling as well as obsolete version branches. Its name
and old Python 2 comments are not evidence that the module can be removed.

The pre-Python-3.7 regular-expression branch is obsolete under Python >=3.11.
Older NumPy branches and global NumPy/regular-expression aliases warrant a
separate dependency-aware review before removal: `np.bool8` still appears in
image branches guarded for NumPy 1.x before 1.24, and in-repository searches cannot establish what
optional dependencies need. Do not broaden the current batch into global
monkey-patch changes without targeted import/behaviour tests.

Deprecated image methods also retain callers: `correct_drifts` and Kerr stack
construction call `apply_all`. `Data.polyfit`, although documented as deprecated,
is still used in the user guide and runnable examples. Preserve these public
entry points; deprecation alone is not authorisation to break the API. This
review identifies usage, not exhaustive execution coverage of every shim.

## Duplication and structural cleanup

TDI 2.0 filename and stream loading now share `_read_tdi2`, providing a completed
focused reduction in duplicated parsing. Data/image generic format modules
still duplicate `catch_sysout`, its logging filter and `_delim_detect`.
Both cleanup contexts have regression coverage after the fixes. Consolidating
these helpers may be useful later, but requires retaining module entry points
and checking import dependencies; duplication alone does not justify another
runtime change today. MAXIMUS also retains parallel readers in its stack module
and format utilities; preserve their callers and real fixtures before any
future consolidation.

Further annotation coverage and large-module splitting are discretionary
future work, not conditions for closing this phase. No new demonstrated runtime
defect requiring an immediate fix was established by this bounded review. This
is not an exhaustive audit of all loaders or third-party dependency behaviour.

## Validation and next session

The latest full Python 3.14 suite remains **477 passed, 592 warnings, 425.71 s**
(`maintenance/runs/20260912-233949-serial-1a15941b`). The final loader regressions
also passed all 19 cases on Python 3.14 and shared `py311-minimal`; see
`loader-cleanup.md`. Tests were not rerun for this documentation-only review.

Next session: review and group the accumulated changes into cohesive commits,
then push when authorised and inspect hosted CI. The coverage-only-on-Ubuntu-
3.14 workflow change remains unverified remotely; it is not a claimed fix for
the intermittent Python 3.12 worker loss. Keep Phase 8 release validation
separate. Today's stopping point adds no runtime changes or commit/push.
