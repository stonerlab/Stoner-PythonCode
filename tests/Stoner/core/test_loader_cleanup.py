"""Protect candidate fallback and cleanup after loader failures."""
import importlib
import logging
from pathlib import Path
import sys
import zipfile

import pytest

from Stoner import Data
from Stoner.core.exceptions import StonerLoadError
from Stoner.formats.data.instruments import load_spc
from Stoner.formats.data.zip import load_zipfile
from Stoner.tools import file as file_tools


@pytest.mark.parametrize("module", ["Stoner.formats.data.generic", "Stoner.formats.image.generic"])
@pytest.mark.parametrize("error", [None, StonerLoadError, RuntimeError])
def test_output_cleanup(module, error):
    manager = importlib.import_module(module).catch_sysout
    stdout, stderr = sys.stdout, sys.stderr
    logger = logging.getLogger("hyperspy.io")
    filters = list(logger.filters)
    try:
        try:
            with manager():
                outer = sys.stdout
                outer_filters = list(logger.filters)
                with manager():
                    assert sys.stdout is not outer
                assert sys.stdout is outer
                assert logger.filters == outer_filters
                if error:
                    raise error("original exception")
            assert error is None, "The original exception was swallowed"
        except Exception as exc:
            assert type(exc) is error
            assert str(exc) == "original exception"
        assert sys.stdout is stdout
        assert sys.stderr is stderr
        assert logger.filters == filters
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        logger.filters[:] = filters


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("payload", [None, b"Not TDI\n", b"\xff\xfe"])
def test_zip_failure_ownership(tmp_path, monkeypatch, borrowed, payload):
    filename = tmp_path / "invalid.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        if payload is not None:
            archive.writestr("data.txt", payload)
    opened = []
    original = zipfile.ZipFile

    class TrackedZip(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(zipfile, "ZipFile", TrackedZip)
    handle = TrackedZip(filename) if borrowed else filename
    try:
        with pytest.raises(StonerLoadError):
            load_zipfile(Data(), handle)
        assert all((archive.fp is not None) == borrowed for archive in opened)
    finally:
        for archive in opened:
            archive.close()


@pytest.mark.parametrize("length", [32, 512, 544])
def test_spc_truncation(tmp_path, length):
    source = Path(__file__).resolve().parents[3] / "sample-data/Raman.spc"
    filename = tmp_path / "truncated.spc"
    filename.write_bytes(source.read_bytes()[:length])
    with pytest.raises(StonerLoadError):
        load_spc(Data(), filename)


@pytest.mark.parametrize("kind", ["zip", "spc"])
def test_bad_candidate_reaches_next_loader(tmp_path, monkeypatch, kind):
    filename = tmp_path / ("invalid." + kind)
    if kind == "zip":
        with zipfile.ZipFile(filename, "w") as archive:
            archive.writestr("data.txt", "Not TDI\n")
        loader = load_zipfile
    else:
        filename.write_bytes(b"short")
        loader = load_spc

    def next_candidate(data, filename, *args, **kwargs):
        return data

    next_candidate.name = "next_candidate"
    monkeypatch.setattr(file_tools, "next_filer", lambda *args, **kwargs: iter([loader, next_candidate]))
    monkeypatch.setattr(file_tools, "get_mime_type", lambda *args, **kwargs: None)
    assert file_tools.auto_load_classes(filename, "Data")["Loaded as"] == "next_candidate"


def test_zip_success_keeps_caller_archive_open(tmp_path):
    source = Path(__file__).resolve().parents[1] / "CoreTest.dat"
    filename = tmp_path / "valid.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.write(source, "data.txt")
    with zipfile.ZipFile(filename) as archive:
        result = load_zipfile(Data(), archive)
        assert archive.fp is not None
        assert result.shape == Data(source).shape


def test_zip_internal_error_is_not_a_format_rejection(tmp_path, monkeypatch):
    module = importlib.import_module("Stoner.formats.data.zip")
    source = Path(__file__).resolve().parents[1] / "CoreTest.dat"
    filename = tmp_path / "valid.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.write(source, "data.txt")

    def broken_copy(*args):
        raise RuntimeError("internal error")

    monkeypatch.setattr(module, "copy_into", broken_copy)
    with pytest.raises(RuntimeError, match="internal error"):
        load_zipfile(Data(), filename)
