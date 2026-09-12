"""Verify text stream input against retained TDI fixtures."""

from contextlib import ExitStack
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from Stoner import Data


@pytest.mark.parametrize("fixture", ["tests/stoner/CoreTest.dat", "sample-data/TDI_Format_RT.txt"])
@pytest.mark.parametrize("kind", ["string", "lines", "generator", "stringio", "file"])
def test_text_input(fixture, kind):
    source = Path(__file__).resolve().parents[3] / fixture
    expected = Data(source)
    receiver = Data(np.array([[42, 43], [44, 45]]), column_headers=["original x", "original y"], setas="xy")
    receiver["original"] = True
    before = receiver.clone
    with ExitStack() as stack:
        text = source.read_text()
        inputs = {
            "string": text,
            "lines": text.splitlines(keepends=True),
            "generator": (line for line in text.splitlines(keepends=True)),
            "stringio": stack.enter_context(StringIO(text)),
            "file": stack.enter_context(source.open()),
        }
        stream = inputs[kind]
        result = receiver << stream
        if kind in {"stringio", "file"}:
            assert not stream.closed
            assert stream.read() == ""
        elif kind == "generator":
            assert list(stream) == []
    np.testing.assert_equal(result.data, expected.data)
    assert result.column_headers == expected.column_headers
    # Loader provenance is added by Data(filename), not by the text operator.
    result["Loaded as"] = expected["Loaded as"]
    assert result.metadata == expected.metadata
    assert result is not receiver
    assert receiver == before
    assert receiver.setas == before.setas


def test_text_stream_starts_at_current_position():
    source = Path(__file__).resolve().parents[3] / "tests/stoner/CoreTest.dat"
    with StringIO("prefix\n" + source.read_text()) as stream:
        stream.readline()
        result = Data() << stream
        assert not stream.closed
    np.testing.assert_equal(result.data, Data(source).data)


def test_invalid_text_leaves_stream_open_and_receiver_unchanged():
    receiver = Data(np.array([[1, 2], [3, 4]]))
    before = receiver.clone
    with StringIO("Not TDI\n") as stream:
        with pytest.raises(RuntimeError, match="Not a TDI File"):
            receiver << stream
        assert not stream.closed
    assert receiver == before


@pytest.mark.parametrize("kind", ["string", "lines", "generator", "stringio", "file"])
def test_tdi2_writer_output(kind):
    source = Path(__file__).resolve().parents[1] / "tdi2-stream.txt"
    with ExitStack() as stack:
        text = source.read_text(encoding="utf-8")
        inputs = {
            "string": text,
            "lines": text.splitlines(keepends=True),
            "generator": (line for line in text.splitlines(keepends=True)),
            "stringio": stack.enter_context(StringIO(text)),
            "file": stack.enter_context(source.open(encoding="utf-8")),
        }
        result = Data() << inputs[kind]
        if kind in {"stringio", "file"}:
            assert not inputs[kind].closed
    assert result.shape == (3, 2)
    np.testing.assert_equal(result.data[:, 0], [1., 2., 3.])
    np.testing.assert_equal(result.data[:2, 1], [4., 5.])
    assert np.ma.getmaskarray(result.data)[2, 1]
    assert result.column_headers == ["Voltage (V)", "Current (A)"]
    assert result["count"] == 3 and type(result["count"]) is int
    assert result["enabled"] is False
    assert result["label"] == "001"
    assert result["note"] == "a=b\tline\nnext"
    assert result["unset"] is None
    assert result["nested"]["gain"] == 1.25
    assert result["TDI Format"] == 2.0


@pytest.mark.parametrize("kind", ["string", "lines", "generator", "stringio", "file"])
def test_real_tdi2_export(kind):
    source = Path(__file__).resolve().parents[3] / "sample-data/TDI_2.0_Format.txt"
    with ExitStack() as stack:
        text = source.read_text(encoding="utf-8")
        inputs = {
            "string": text,
            "lines": text.splitlines(keepends=True),
            "generator": (line for line in text.splitlines(keepends=True)),
            "stringio": stack.enter_context(StringIO(text)),
            "file": stack.enter_context(source.open(encoding="utf-8")),
        }
        result = Data() << inputs[kind]
        if kind in {"stringio", "file"}:
            assert not inputs[kind].closed
    assert result.shape == (401, 8)
    np.testing.assert_equal(result.data[:, 0], np.arange(401))
    assert result["counter"]["collect_data"] is True
    assert result["counter"]["clear_filter"] == "True"
    assert result["alert"]["message_expr"] == "'Alert'"
    assert result.column_headers == text.splitlines()[0].split("\t")[1:]
    expected = np.loadtxt(source, delimiter="\t", skiprows=1, usecols=range(1, 9), comments=None)
    np.testing.assert_allclose(result.data, expected)
