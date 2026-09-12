# Text stream input and TDI 2.0

Reviewed and implemented 2026-09-12.

## Scope and findings

The vague stream TODO in `Data.__lshift__` did not describe a failure for
existing text inputs. Strings, lists of lines, generators, StringIO and open
text files already work with the retained `CoreTest.dat` and
`sample-data/TDI_Format_RT.txt` fixtures. No legacy parser changes were needed.
Tests verify data, headers, metadata, receiver preservation, consumption from
the current stream position, and caller ownership on success and failure.

The maintainer requested TDI 2.0 coverage and identified the existing real
export `sample-data/TDI_2.0_Format.txt`. The old operator rejected that header.
A separate TDI 2.0 parsing branch now reads all 401 rows and eight columns,
including Python literal metadata, without changing the TDI 1.0/1.5 branches.
The result is already a new instance of the receiver's class and is returned
directly for TDI 2.0: a redundant constructor copy would re-infer explicit
string values such as `counter.clear_filter='True'`.

TDI 2.0 metadata uses `ast.literal_eval`; explicit strings are assigned with
a String hint to avoid untyped coercion. Empty numeric cells are masked,
short rows are padded, and metadata-only rows do not create numerical rows.
Streams are consumed without closing or rewinding them. Input is decoded text.
The operator docstring and user guide describe these contracts.

Correction from the subsequent loader review: filename loading already had a
dedicated `TDI_2_0` loader in `Stoner/formats/data/tdi2.py`; the earlier claim
that filename support was absent was incorrect. The parser now lives there
and is shared with the stream operator, preserving the existing loader's
nested-dictionary/list metadata convention. Its registration is retained.
General metadata clone semantics remain a separate concern.
Non-literal Python object representations are outside this parser's contract.

## Writer provenance

The measurement project was read only throughout. Inspected
`D:/Programming/stoner_measurement/src/stoner_measurement/plugins/command/save.py`,
SHA256 `94CED4E150A223D479914E1DBC13E302C7049466EF3E4C78B69AE3EB6B58FFDE`.
Its `_flatten_to_metadata`, `SavePayload`, `BaseSaveWriter` and `TdiSaveWriter`
definitions were executed in isolation to generate `tests/stoner/tdi2-stream.txt`.
This supplementary fixture covers unequal column lengths, metadata longer than
data, None, booleans, numeric-looking strings and escaped tab/newline/equals.
It supplements the real export, rather than replacing it.
Generation script: `maintenance/runs/generate-tdi2-stream.py`.
Fixture SHA256: `9F681BCC23977B887C992474C183DDF086ED0B5DC193544E951BDC663E3F5746`.

## Validation

- Python 3.14 stream and existing operator tests: 30 passed, one warning,
  27.19 seconds (`maintenance/runs/20260912-222043-focused-ed4b2575`).
- Shared Python 3.11 minimal environment: 22 passed, 18 warnings,
  13.64 seconds (`maintenance/runs/20260912-222054-focused-26b387de`).
- Fresh cached Sphinx build and API audit pass: 106 reviewed Windows warnings,
  no unexpected warnings or removed API objects, and no tracked plot-cache
  changes (`maintenance/runs/phase7-stream-docs*`).
- Full suite: 417 passed, one failed, 592 warnings, 445.95 seconds
  (`maintenance/runs/20260912-222128-serial-da7e5af6`). The failure was the
  exact `dir(Data)` inventory test: the initial private parser method added
  an attribute. Moved the parser to module scope, retaining the existing
  inventory rather than changing the test's expectation.
- After that correction, all 76 core/stream tests passed on Python 3.14,
  one warning, 37.74 seconds
  (`maintenance/runs/20260912-222951-focused-4ca2dddf`).
- Final Python 3.11 stream and directory-inventory checks: 23 passed,
  18 warnings, 16.92 seconds
  (`maintenance/runs/20260912-223003-focused-4aee579d`).

The complete suite was not repeated after the helper relocation; the affected
core tests were rerun. Sphinx validation preceded that internal relocation,
which left the public docstrings and user guide unchanged. Hosted CI remains
pending, and the batch is uncommitted.

Initial test-authoring failures were corrected: Data requires a NumPy array
for this construction path; a multirow receiver avoids an unrelated singleton
comparison issue; NumPy reference loading needs comments=None because real
metadata contains hash characters. These were not stream defects.
