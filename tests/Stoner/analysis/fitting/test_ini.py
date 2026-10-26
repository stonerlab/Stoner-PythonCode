"""Exercise CSV layout forwarding and historical fitting-INI compatibility."""

from io import StringIO

import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.fitting.models import cfg_data_from_ini


def load_config(tmp_path, text, options="", columns="x: V\ny: G", **kwargs):
    """Load a real CSV through a text INI stream without user interaction."""
    filename = tmp_path / "measurement.csv"
    filename.write_text(text, encoding="utf-8")
    config = StringIO(f"[Data]\ntype: CSVFile\n{columns}\n{options}\n")
    data = cfg_data_from_ini(config, filename=filename, **kwargs)
    assert not config.closed
    return data


def test_omitted_layout_uses_csv_defaults(tmp_path):
    data = load_config(tmp_path, "V,G\n-1,2\n0,3\n1,4\n")
    assert data.column_headers == ["V", "G"]
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2], [0, 3], [1, 4]])
    assert list(data.setas) == list("xy")


@pytest.mark.parametrize("uncertainty", [False, True])
def test_preamble_intervening_lines_and_independent_separators(tmp_path, uncertainty):
    text = "Preamble\nSecond preamble\nV|G|dG\nIntervening line\n-1;2;0.1\n0;3;0.2\n1;4;0.3\n"
    columns = "x: V\ny: G" + ("\ne: dG" if uncertainty else "")
    options = "header_line: 2\ndata_line: 4\nheader_delim: |\ndata_delim: ;"
    data = load_config(tmp_path, text, options, columns)
    direct = Data.load(tmp_path / "measurement.csv", filetype="CSVFile", header_line=2, data_line=4,
                       header_delim="|", data_delim=";")
    assert data.column_headers == direct.column_headers == ["V", "G", "dG"]
    np.testing.assert_array_equal(data.to_numpy(), direct.to_numpy())
    assert list(data.setas) == list("xye" if uncertainty else "xy.")


def test_legacy_rows_are_one_based(tmp_path):
    data = load_config(tmp_path, "Preamble\nSecond preamble\nV;G\nIntervening line\n-1;2\n0;3\n1;4\n",
                       "header_line: 3\nstart_line: 5\nseparator: ;")
    assert data.column_headers == ["V", "G"]
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2], [0, 3], [1, 4]])


def test_legacy_original_first_header_and_second_data_rows(tmp_path):
    data = load_config(tmp_path, "V,G\n-1,2\n0,3\n1,4\n", "header_line: 1\nstart_line: 2\nseparator: ,")
    assert data.column_headers == ["V", "G"]
    assert data.shape == (3, 2) and data[0, 0] == -1


def test_modern_rows_override_legacy_rows(tmp_path):
    data = load_config(tmp_path, "V,G\n-1,2\n0,3\n1,4\n", "header_line: 0\ndata_line: 1\nstart_line: invalid")
    assert data.column_headers == ["V", "G"]
    assert data.shape == (3, 2)


@pytest.mark.parametrize("options,text", [
    ("header_line: 0\ndata_line: 1\nseparator: ;\nheader_delim: |", "V|G\n-1;2\n0;3\n1;4\n"),
    ("header_line: 0\ndata_line: 1\nseparator: |\ndata_delim: ;", "V|G\n-1;2\n0;3\n1;4\n"),
    ("header_line: 0\ndata_line: 1\nseparator: invalid\nheader_delim: |\ndata_delim: ;",
     "V|G\n-1;2\n0;3\n1;4\n"),
])
def test_modern_delimiters_override_separator_independently(tmp_path, options, text):
    data = load_config(tmp_path, text, options)
    assert data.column_headers == ["V", "G"]
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2], [0, 3], [1, 4]])


@pytest.mark.parametrize("options", [
    "header_line: None\ndata_line: 0\nheader_delim: None\ndata_delim: None",
    "header_line: none\nstart_line: 1\nseparator: None",
])
def test_headerless_numeric_selectors_and_detection(tmp_path, options):
    data = load_config(tmp_path, "-1;2;0.1\n0;3;0.2\n1;4;0.3\n", options, "x: 0\ny: 1\ne: 2")
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2, 0.1], [0, 3, 0.2], [1, 4, 0.3]])
    assert list(data.setas) == list("xye")


def test_header_and_data_delimiters_are_detected_independently(tmp_path):
    data = load_config(tmp_path, "Preamble\nV,G\nSkip\n-1;2\n0;3\n1;4\n",
                       "header_line: 1\ndata_line: 3\nheader_delim: None\ndata_delim: None")
    assert data.column_headers == ["V", "G"]
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2], [0, 3], [1, 4]])


@pytest.mark.parametrize("delimiter, option", [("\t", r"\t"), (" ", '" "'), ("%", "%")])
def test_literal_and_escaped_delimiters(tmp_path, delimiter, option):
    text = delimiter.join(["V", "G"]) + "\n" + "\n".join(delimiter.join(row) for row in [
        ["-1", "2"], ["0", "3"], ["1", "4"]]) + "\n"
    data = load_config(tmp_path, text, f"separator: {option}")
    assert data.column_headers == ["V", "G"]
    np.testing.assert_array_equal(data.to_numpy(), [[-1, 2], [0, 3], [1, 4]])


def test_loader_keywords_override_ini_values(tmp_path):
    data = load_config(tmp_path, "Preamble\nV|G\nSkip\n-1;2\n0;3\n1;4\n",
                       "header_line: invalid\ndata_line: invalid\nheader_delim: invalid\ndata_delim: invalid",
                       header_line=1, data_line=3, header_delim="|", data_delim=";")
    assert data.column_headers == ["V", "G"]
    assert data.shape == (3, 2)


def test_keyword_override_keeps_other_legacy_rows_normalised(tmp_path):
    data = load_config(tmp_path, "Preamble\nV;G\nSkip\n-1;2\n0;3\n1;4\n",
                       "header_line: 2\nstart_line: 3\nseparator: ;", data_line=3)
    assert data.column_headers == ["V", "G"]
    assert data.shape == (3, 2)


@pytest.mark.parametrize("options, name", [
    ("header_line: -1", "header_line"),
    ("data_line: -1", "data_line"),
    ("data_line: None", "data_line"),
    ("header_line: text", "header_line"),
    ("start_line: 0", "data_line"),
    ("header_line: 0\nstart_line: 2", "header_line"),
    ("header_delim: multiple", "header_delim"),
    ("data_delim:", "data_delim"),
])
def test_invalid_layout_reports_the_option(tmp_path, options, name):
    with pytest.raises(ValueError, match=rf"\[Data\] {name}"):
        load_config(tmp_path, "V,G\n-1,2\n0,3\n1,4\n", options)


def test_other_loader_ignores_csv_layout_entries(tmp_path):
    path = tmp_path / "measurement.tdi"
    Data(np.arange(6.).reshape(3, 2), column_headers=["V", "G"]).save(path, as_loaded="DataFile")
    config = StringIO("[Data]\ntype: DataFile\nx: V\ny: G\nheader_line: invalid\nseparator: invalid\n")
    data = cfg_data_from_ini(config, filename=path)
    assert data.shape == (3, 2) and data.column_headers == ["V", "G"]
