# Loader exception-boundary audit

Audited 2026-09-12. Runtime source is unchanged by this audit.

## Contract

StonerLoadError intentionally rejects the current candidate so auto_load_classes
can try the next loader. A control probe confirms that this works. Findings
below concern exceptions escaping that boundary or cleanup failing during an
otherwise legitimate rejection. Do not replace the dispatcher's narrow catch
with a blanket Exception catch.

## Reproduced findings

1. Both catch_sysout context managers (formats/data/generic.py and
   formats/image/generic.py) put restoration after yield without finally.
   Raising StonerLoadError inside either leaves stdout and stderr redirected
   and an extra hyperspy.io suppression filter installed. The audit restores
   all three explicitly after observing the failure. The data helper is used
   in the HyperSpy loader; the image duplicate was tested directly.
   Proposed fix: exception-safe cleanup with preserved filter/stream ownership.

2. load_zipfile has gaps after opening the archive. A member containing
   unsupported text raises RuntimeError('Not a TDI File') from the stream
   parser; the next candidate is not reached, and an archive opened by the
   loader remains open. Conversely, an empty archive supplied by the caller
   produces StonerLoadError but is closed by the broad opening-error handler.
   Proposed fix: consistent owned/borrowed archive cleanup and conversion of
   expected format-rejection errors at the ZIP boundary. Keep genuine internal
   errors distinguishable rather than broadly swallowing all exceptions.

3. load_spc unpacks its 512-byte header before validating the available bytes.
   The first 32 bytes of the retained real sample sample-data/Raman.spc cause
   struct.error('unpack requires a buffer of 512 bytes'). A controlled dispatch
   probe confirms that this aborts candidate iteration rather than reaching
   the next loader. Proposed fix: validate required binary field lengths and
   use StonerLoadError for truncation, retaining real-file regression coverage.

## Optional dependencies

A fresh-process import probe simulated absence of fabio, rsciio, nptdms and
hyperspy together. Stoner imports successfully; Fabio/RosettaSciIO flags are
disabled, HyperSpy is unavailable and the TDMS function is not registered.
No installation or environment changes were made. This verifies the missing
package path, not every broken installation or every third-party reader.

The broad HyperSpy and RosettaSciIO reader catches are not classified as bugs
merely because they translate third-party exceptions into StonerLoadError.
That is consistent with fallback. Narrowing them requires reader-specific
exception evidence. Existing HyperSpy exception chaining preserves diagnostics;
RosettaSciIO also prints the caught exception but raises a bare StonerLoadError.

## Evidence and boundaries

Reproduction: maintenance/phase7/audit-loader-exceptions.py, run through
conda run -n py314 python. Results are in
maintenance/runs/exception-audit/results.json. The script restores global
streams/log filters and closes all audit-owned archives after checking them.
It uses a sentinel second candidate to test the actual dispatcher loop without
relying on a particular installed optional-loader ordering.

Missing-dependency simulation and output:
maintenance/runs/exception-audit/missing-optional.py and missing-optional.json.
Search index: maintenance/runs/exception-audit-index.txt.

This is a focused first pass through dispatch, optional-import guards, broad
reader catches and selected malformed-file boundaries, not a complete audit
of every instrument format. No full suite or documentation build was repeated
for the audit. The real SPC fixture and measurement repository are unchanged.

Suggested order: output-redirection cleanup first, then ZIP ownership/fallback,
then SPC truncation checks. Each should be a small separately validated batch.
