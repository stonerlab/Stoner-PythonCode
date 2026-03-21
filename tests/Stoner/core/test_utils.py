# -*- coding: utf-8 -*-
"""Tests for Stoner.core.utils"""

import csv

import pytest

from Stoner.core.utils import Tab_Delimited, decode_string


def test_decode_string_simple():
    assert decode_string("xxyy") == "xxyy", "decode_string with no patterns should return unchanged"


def test_decode_string_repeated_x():
    assert decode_string("3x") == "xxx", "decode_string failed to expand 3x"


def test_decode_string_repeated_y():
    assert decode_string("2y") == "yy", "decode_string failed to expand 2y"


def test_decode_string_mixed():
    result = decode_string("x2yz")
    assert result == "xyyz", "decode_string failed for mixed pattern x2yz"


def test_decode_string_dots_and_dashes():
    assert decode_string("3.") == "...", "decode_string failed to expand 3."
    assert decode_string("2-") == "--", "decode_string failed to expand 2-"


def test_decode_string_multiple_patterns():
    result = decode_string("2x3y")
    assert result == "xxyyy", "decode_string failed for multiple patterns 2x3y"


def test_Tab_Delimited_is_csv_dialect():
    assert issubclass(Tab_Delimited, csv.Dialect), "Tab_Delimited should subclass csv.Dialect"
    assert Tab_Delimited.delimiter == "\t", "Tab_Delimited delimiter should be a tab"
    assert Tab_Delimited.quoting == csv.QUOTE_NONE, "Tab_Delimited quoting should be QUOTE_NONE"
    assert Tab_Delimited.doublequote is False, "Tab_Delimited doublequote should be False"
    assert Tab_Delimited.lineterminator == "\r\n", "Tab_Delimited lineterminator should be CRLF"


def test_Tab_Delimited_roundtrip():
    import io

    output = io.StringIO()
    writer = csv.writer(output, dialect=Tab_Delimited)
    writer.writerow(["a", "b", "c"])
    output.seek(0)
    reader = csv.reader(output, dialect=Tab_Delimited)
    row = next(reader)
    assert row == ["a", "b", "c"], "Tab_Delimited roundtrip failed"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
