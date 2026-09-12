# Loader cleanup and fallback fixes

Implemented 2026-09-12 after maintainer approval of the exception audit.

## Changes

Both catch_sysout helpers now restore stdout/stderr and remove their logging
filter in finally, allowing the original exception to propagate. Each context
owns a filter instance, so nesting preserves the outer context's suppression.

The ZIP loader tracks ownership immediately after opening. Its finally closes
only loader-owned archives on success or failure; open caller-owned archives
remain open. Empty archives, unreadable members, unsupported TDI headers,
invalid UTF-8 and expected malformed-text errors reject the candidate with
StonerLoadError. ZIPs remain opened read-only. Genuine downstream failures
such as a RuntimeError in copy_into still propagate, with cleanup performed.

SPC binary parsing now runs inside a narrow context that converts struct.error
into StonerLoadError. This covers the main header and later subheaders/data
fields without changing parsing calculations. The file context still closes
the handle. Chained causes help direct calls/debugging but are not required
by the dispatcher, whose behaviour and exception catches are unchanged.

## Validation

- Regression baseline: 12 failed, five passed, one warning, 6.32 seconds
  (`maintenance/runs/20260912-233604-focused-99833563`).
- Existing file-format coverage plus initial regressions: 83 passed,
  13 warnings, 63.40 seconds
  (`maintenance/runs/20260912-233751-focused-5c4a612b`).
- Final Python 3.14 regressions: 19 passed, one warning, 6.18 seconds
  (`maintenance/runs/20260912-233858-focused-5cdf1bed`).
- Final Python 3.11 minimal regressions: 19 passed, 18 warnings, 8.09 seconds
  (`maintenance/runs/20260912-233910-focused-3085c190`).
- Full Python 3.14 serial suite: 477 passed, 592 warnings, 425.71 seconds
  (`maintenance/runs/20260912-233949-serial-1a15941b`).

Cases cover nested redirection, success and propagated failures, owned/borrowed
ZIP archives on malformed input, successful borrowed-archive use, internal
error propagation, three truncation points from the real Raman.spc fixture,
and reaching a sentinel next candidate through auto_load_classes.

No registry priorities, format-selection policy or optional dependencies were
changed. Runtime behaviour changes are limited to cleanup and expected
candidate rejection. No Sphinx rebuild was required for this batch's internal
control-flow changes. Changes are uncommitted; hosted CI remains pending.
