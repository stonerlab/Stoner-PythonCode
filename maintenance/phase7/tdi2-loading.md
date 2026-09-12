# TDI 2.0 length handling and metadata copy boundary

Implemented 2026-09-12, with explicit maintainer approval for the shared
`copy_into` metadata correction and a full regression run.

## Existing loader retained

The dedicated `Stoner/formats/data/tdi2.py` loader already supported filename
loading and reconstructed dotted paths/list indices as nested metadata.
Earlier statements that filename support was missing were incorrect. Its
`TDI_2_0` registration, patterns, MIME types, priority and nested-metadata
convention are retained. No competing loader was added to `generic.py`.

The new parser lives in the existing TDI 2.0 module and is shared with `<<`
through a local import, avoiding a core/formats import cycle. The previous
pandas reader retained metadata-only rows as extra numerical rows. The shared
parser consumes all metadata, excludes metadata-only rows from numerical data,
and pads absent numerical cells with masked values. Malformed input is rejected
with the established StonerLoadError at the filename-loader boundary.
The source stream is read inside FileManager rather than reopened separately.

The read-only measurement writer's TdiSaveWriter.build_rows explicitly uses
max(metadata count, longest column length), padding missing cells with empty
strings. Unequal data-column lengths are therefore a supported writer case.

## Authorised copy fix

The existing constructor calls copy_into after loading. Its generic deepcopy
re-inferred top-level metadata strings: '001', 'False', '1e3' and '2026-01-01'
became an integer, boolean, float and datetime respectively. Four regressions
reproduced this before the fix
(`maintenance/runs/20260912-224349-focused-65e05824`).

copy_into now starts from TypeHintedDict.copy(), which preserves type hints,
then deep-copies values through the parent mapping setter without re-running
inference. A shared memo preserves internal aliases while separating source
and destination nested objects. Other attribute-copy behaviour is unchanged.
The generic deepcopy implementation and the TDI 1.5 loader workaround are
unchanged. The maintainer explicitly approved this shared-boundary fix.

## Coverage

- 24 combinations of metadata count (zero, shorter, equal, longer) and column
  lengths (equal, unequal, either empty, both empty, single-row data).
- Existing real TDI 2.0 export: 401 rows and eight columns; the supplementary
  actual-writer fixture covers missing cells, escaped text and native types.
- Matching filename/stream values, masks, headers and nested metadata.
- Both TDI 1.0 and 1.5 with shorter/longer metadata and unequal column lengths.
- Rejection of invalid headers, malformed metadata and excess numeric columns;
  caller streams remain open.
- String/native type preservation, independent nested NumPy arrays, shared
  internal aliases, destination class, filename, masks and column roles.

## Validation

- Python 3.14 focused checks: 60 passed, one warning, 27.79 seconds
  (`maintenance/runs/20260912-224423-focused-0b1280ba`).
- Python 3.11 minimal environment: 60 passed, 18 warnings, 18.84 seconds
  (`maintenance/runs/20260912-224630-focused-796b35cf`).
- Fresh cached Sphinx build and audit pass: the same 106 reviewed Windows
  warnings, no unexpected warnings or lost API objects, and unchanged tracked
  plot cache (`maintenance/runs/phase7-tdi2-docs*`).
- Full Python 3.14 serial suite: 456 passed, 592 warnings, 450.25 seconds
  (`maintenance/runs/20260912-224618-serial-7674731c`). This includes the final
  shared copy fix, all format tests, folder/image operations and doc examples.

The measurement repository and real sample exports were not modified.
Changes remain uncommitted; hosted CI has not run for this batch.
