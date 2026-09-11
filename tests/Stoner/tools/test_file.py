# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.file (test_is_zip function)."""

import os
import tempfile
import zipfile

import pytest

from Stoner.tools import file as file_tools
from Stoner.tools.file import test_is_zip as is_zip_file


@pytest.mark.parametrize("kind", ["loader", "saver"])
def test_clear_routine_removes_all_indexes(monkeypatch, kind):
    """Remove a routine from every index while retaining unrelated entries."""
    routine = lambda: None
    other = lambda: None
    names = {"temporary": routine, "other": other}
    patterns = {".dat": [(1, routine), (2, other), (3, routine)]}
    mime_types = {"text/plain": [(1, routine), (2, other)]}
    monkeypatch.setattr(file_tools, f"_{kind}s_by_name", names)
    monkeypatch.setattr(file_tools, f"_{kind}s_by_pattern", patterns)
    if kind == "loader":
        monkeypatch.setattr(file_tools, "_loaders_by_type", mime_types)
    removed = file_tools.clear_routine("temporary", loader=kind == "loader", saver=kind == "saver")
    assert removed == {kind: routine}
    assert names == {"other": other}
    assert patterns == {".dat": [(2, other)]}
    if kind == "loader":
        assert mime_types == {"text/plain": [(2, other)]}


def test_is_zip_with_empty_string():
    assert is_zip_file("") is False, "test_is_zip should return False for empty string"


def test_is_zip_with_none():
    assert is_zip_file(None) is False, "test_is_zip should return False for None"


def test_is_zip_with_bytes_containing_null():
    assert is_zip_file(b"data\x00more") is False, "test_is_zip should return False for bytes with null"


def test_is_zip_with_real_zip():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with zipfile.ZipFile(tmp_name, "w") as zf:
            zf.writestr("hello.txt", "Hello, world!")
        result = is_zip_file(tmp_name)
        assert result is not False, "test_is_zip should detect a real zip file"
        assert result[0] == tmp_name, "test_is_zip should return the zip filename"
        assert result[1] == "", "test_is_zip should return empty member for direct zip"
    finally:
        os.unlink(tmp_name)


def test_is_zip_with_non_zip_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tmp.write("Not a zip file")
        tmp_name = tmp.name
    try:
        result = is_zip_file(tmp_name)
        assert result is False, "test_is_zip should return False for non-zip file"
    finally:
        os.unlink(tmp_name)


def test_is_zip_with_path_inside_zip():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with zipfile.ZipFile(tmp_name, "w") as zf:
            zf.writestr("subdir/data.txt", "content")
        # Test with a path that includes the zip file + member path
        result = is_zip_file(os.path.join(tmp_name, "subdir", "data.txt"))
        assert result is not False, "test_is_zip should find zip when path goes through a zip"
        assert result[0] == tmp_name, "test_is_zip should find the zip file path"
    finally:
        os.unlink(tmp_name)


def test_mime_failure_falls_back_to_filename_matching(monkeypatch, tmp_path):
    """Return no MIME type when the native magic database cannot be loaded."""

    class BrokenMagic:
        """Model a native libmagic failure while identifying a file."""

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def id_filename(self, _filename):
            raise RuntimeError("invalid character range in magic database")

    monkeypatch.setattr(file_tools, "filemagic", BrokenMagic)
    monkeypatch.setattr(file_tools, "magic_errors", (RuntimeError,))

    assert file_tools.get_mime_type(tmp_path / "example.dat") is None


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
