# Metadata-copy review

Reviewed 2026-09-12. Runtime behaviour remains unchanged pending confirmation
of the measurement writer's type semantics.

The TDI and PNG loaders in `Stoner/formats/data/generic.py` deep-copy metadata
to correct types. `TypeHintedDict` has no custom deep-copy implementation, so
reconstruction inserts values through `__setitem__` again. This infers Python
types and regenerates hints: a stored String `00123` becomes integer `123`,
String `True` becomes Boolean, and hints such as I16 and Single Float become
I32 and Double Float respectively.

These changes alone do not establish a defect. The maintainer prefers
appropriate Python types where possible. In particular, a String hint may
represent a measurement writer's fallback for untyped data rather than a
requirement to preserve textual representation.

## Evidence and provenance boundary

- TDI loading passes the first metadata cell directly to `import_key`, which
  passes its key and value to `__setitem__`. An explicit String hint in that
  cell therefore comes from the file, not a default added by this loader.
- Without an explicit hint, `__setitem__` already calls `string_to_type`
  before inferring the hint. It does not simply label all untyped input String.
- The real sample `sample-data/6221-Lockin-DAQ Temperature Control -30.0Deg 0.004T.txt`
  contains `Option:560 Filter High Pass.__text__{String}=0.03` at line 194,
  alongside `Option:560 Filter High Pass{U16}=0` at line 193. Similar pairs
  appear for other filter settings. These suggest textual selection labels,
  but the writer implementation has not been verified; the file alone cannot
  establish whether String was a fallback.
- The initial proposed tests used synthetic explicit hints, including an
  identifier appended to a real TDI fixture. Their run produced five failures
  and one pass in 5.69 seconds (`maintenance/runs/20260912-182530-focused-8ff09cf3`).
  They demonstrate conversion, not an agreed preservation contract, and were
  moved out of the test suite to
  `maintenance/runs/phase7-metadata-copy-proposed-tests.py`.

## TDI 2.0 comparison

The maintainer confirmed that the TDI 1.5 writer is local LabVIEW 2018 code
using Variant type descriptor introspection. Its original conversion rationale
remains unresolved; inspecting the Python successor cannot establish it.

The local successor was inspected at
`D:/Programming/stoner_measurement/src/stoner_measurement/plugins/command/save.py`
(`_flatten_to_metadata` and `SaveCommand._build_metadata`), corresponding to
the supplied [upstream source](https://github.com/gb119/stoner_measurement/blob/main/src/stoner_measurement/plugins/command/save.py).
This inspection used the local file; the remote page could not be fetched.

- TDI 2.0 writes each leaf with `type(obj).__name__` and `repr(obj)`, after
  converting objects exposing `.item()` to their scalar value.
- A Python string `00123` therefore becomes `{str}='00123'`, whereas an integer
  becomes `{int}=123`. There is no default String hint in this serialiser.
- The metadata includes sequence configuration and evaluated outputs. A string
  already produced upstream remains a string at the serialisation boundary;
  this does not prove how every upstream plugin chose its value type.
- Stoner's `Stoner/formats/data/tdi2.py::_parse_entry` uses `ast.literal_eval`
  on the representation, rather than using the captured type name to cast it.
  Quoted strings and native numbers are therefore distinguishable at parsing.
  This is a source inspection, not an end-to-end round-trip validation through
  metadata assignment and subsequent cloning.

## Next step

Retain the current TDI 1.5 loader workaround and describe its observed
re-inference behaviour without claiming that its original rationale is known.
Do not introduce a general type-preserving deep-copy method based solely on
the synthetic examples. Any future TDI 2.0 compatibility work should use the
Python writer as its source contract and validate the complete load/clone
path separately.

## Authorised follow-up (2026-09-12)

Real TDI 2.0 filename/stream validation subsequently exercised the constructor
copy boundary. The maintainer authorised preserving existing metadata values
and type hints in `copy_into`, with independent deep copies of nested values.
This fixes the constructor's re-inference of explicit Python-writer strings;
the general deepcopy implementation and legacy loader workaround are unchanged.
See `tdi2-loading.md` for regressions and validation.
