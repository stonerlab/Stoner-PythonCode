# Attocube TIFF TODO review

Reviewed 2026-09-12. Replaced the stale multipage TIFF implementation TODO in
the AttocubeScan class docstring with a description of its inherited support.

`ImageFolderMixin.to_tiff` serialises the marshalled images as multiple TIFF
pages and includes layout information. `from_tiff` reconstructs the images
and restores the layout. `tests/Stoner/test_FileFormats.py::test_attocube_scan`
already round-trips the real `sample-data/attocube_scan` fixture through these
methods and asserts that the output exists and the scan layout is preserved.
That assertion is not a claim of bit-exact pixel or metadata round-tripping.

The existing test passed in the preceding CI investigation on Python 3.12
and 3.14, including two-worker runs; see `ci-worker-crash.md` for exact logs
and results. No runtime tests were repeated for this docstring-only edit.
An AST comparison against HEAD, excluding docstrings, confirmed identical
executable code (`maintenance/runs/check-attocube-docstring.py`).
`git -c core.whitespace=cr-at-eol diff --check` passes.

Next remaining TODO review: establish which extra attributes folder groups
already inherit, and whether the add_group TODO still describes a defect.
