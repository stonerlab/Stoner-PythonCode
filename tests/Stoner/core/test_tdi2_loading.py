"""Check filename and stream loading of rectangular and ragged TDI exports."""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from Stoner import Data
from Stoner.core.exceptions import StonerLoadError
from Stoner.formats.data.tdi2 import load_tdi2_format


@pytest.mark.parametrize("metadata_count", [0, 1, 3, 6])
@pytest.mark.parametrize("lengths", [(3, 3), (3, 1), (0, 3), (3, 0), (0, 0), (1, 1)])
def test_tdi2_independent_lengths(tmp_path, metadata_count, lengths):
    """Follow the writer's empty-cell layout with independently sized metadata and columns."""
    columns = [np.arange(size) + 10 * index for index, size in enumerate(lengths)]
    rows = ["TDI Format 2.0\tx\ty"]
    for i in range(max(metadata_count, *lengths)):
        metadata = f"entry{i}{{str}}='00{i}'" if i < metadata_count else ""
        values = [str(column[i]) if i < len(column) else "" for column in columns]
        rows.append("\t".join([metadata, *values]))
    source = tmp_path / "lengths.txt"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = Data(source)
    streamed = Data() << source.read_text()
    for result in (loaded, streamed):
        assert result.shape == (max(lengths), 2)
        assert result.column_headers == ["x", "y"]
        assert result["TDI Format"] == 2.0
        for i in range(metadata_count):
            assert result[f"entry{i}"] == f"00{i}"
        for index, column in enumerate(columns):
            np.testing.assert_equal(result.data[:len(column), index], column)
            np.testing.assert_array_equal(np.ma.getmaskarray(result.data)[:, index],
                                          np.arange(max(lengths)) >= len(column))


@pytest.mark.parametrize("fixture", ["sample-data/TDI_2.0_Format.txt", "tests/stoner/tdi2-stream.txt"])
def test_tdi2_real_and_writer_fixtures(fixture):
    source = Path(__file__).resolve().parents[3] / fixture
    loaded = Data(source)
    streamed = Data() << source.read_text(encoding="utf-8")
    assert loaded["Loaded as"] == "TDI_2_0"
    assert Path(loaded.filename) == source
    np.testing.assert_equal(loaded.data, streamed.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(loaded.data), np.ma.getmaskarray(streamed.data))
    assert loaded.column_headers == streamed.column_headers
    for key, value in streamed.metadata.items():
        assert loaded[key] == value
        assert type(loaded[key]) is type(value)


@pytest.mark.parametrize("header", ["TDI Format=Text 1.0", "TDI Format 1.5"])
@pytest.mark.parametrize("metadata_count", [1, 5])
def test_legacy_loader_length_cases(tmp_path, header, metadata_count):
    rows = [f"{header}\tx\ty"]
    for i in range(max(metadata_count, 3)):
        metadata = f"entry{i}{{Double Float}}={i}" if i < metadata_count else ""
        rows.append("\t".join([metadata, str(i) if i < 3 else "", str(i + 10) if i < 2 else ""]))
    source = tmp_path / "legacy.txt"
    source.write_text("\n".join(rows) + "\n")
    loaded = Data(source)
    assert loaded.shape == (3, 2)
    assert loaded["Loaded as"] == "DataFile"
    for i in range(metadata_count):
        assert loaded[f"entry{i}"] == i
    np.testing.assert_equal(loaded.data[:, 0], [0, 1, 2])
    assert np.ma.getmaskarray(loaded.data)[2, 1]


@pytest.mark.parametrize("text", ["Not TDI\tx\n", "TDI Format 2.0\tx\nbad\t1\n",
                                "TDI Format 2.0\tx\n\t1\t2\n"])
def test_loader_rejects_invalid_tdi(text):
    with StringIO(text) as stream:
        with pytest.raises(StonerLoadError):
            load_tdi2_format(Data(), stream)
        assert not stream.closed
