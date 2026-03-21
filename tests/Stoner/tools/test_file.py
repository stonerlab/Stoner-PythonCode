# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.file (test_is_zip function)."""

import os
import tempfile
import zipfile

import pytest

from Stoner.tools.file import test_is_zip as is_zip_file


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


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
