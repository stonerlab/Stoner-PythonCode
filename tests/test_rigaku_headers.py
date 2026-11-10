"""Preserve sparse indexed Rigaku metadata when a scalar header is promoted."""

import re

import numpy as np
import pytest

from Stoner import Data
from Stoner.formats.data.instruments import _parse_header

pytestmark = pytest.mark.formats


@pytest.mark.parametrize("index", [2, 3])
def test_sparse_string_header_preserves_first_value(index):
    """Retain a scalar first value and pad missing indexed string entries."""
    data = Data()
    data["axis"] = "first"
    key = f"axis-{index}"
    matched = re.match(r"(.*)-(\d+)$", key)

    assert _parse_header(data, matched, '"later"', key) is data
    assert data["axis"] == ["first"] + [""] * (index - 1) + ["later"]


def test_sparse_numeric_header_preserves_dtype_and_missing_values():
    """Keep floating-point metadata and NaN padding during scalar promotion."""
    data = Data()
    data["temperature"] = 1.5
    key = "temperature-3"

    _parse_header(data, re.match(r"(.*)-(\d+)$", key), '"2.5"', key)

    np.testing.assert_array_equal(data["temperature"], [1.5, np.nan, np.nan, 2.5])
    assert data["temperature"].dtype == np.dtype(float)


def test_unindexed_header_missing_marker_remains_nan():
    """Treat the instrument's quoted missing-value marker as numeric NaN."""
    data = Data()

    _parse_header(data, None, '"-"', "temperature")

    assert np.isnan(data["temperature"])
