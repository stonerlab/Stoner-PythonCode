"""Regression coverage for row selection through Data.del_rows."""

import numpy as np
import pytest

from Stoner import Data


def make_data():
    """Create identifiable rows with column roles, metadata and a cell mask."""
    data = Data(np.column_stack((np.arange(5), np.arange(5) * 10)),
                column_headers=["Position", "Signal"], setas="xy")
    data["Run"] = "row deletion"
    data.data.mask = np.zeros(data.shape, dtype=bool)
    data.data.mask[2, 1] = True
    return data


def assert_retained(data, values, mask):
    """Check row contents and the surrounding Data contract."""
    np.testing.assert_array_equal(data.data.data, values)
    np.testing.assert_array_equal(data.data.mask, mask)
    assert data.column_headers == ["Position", "Signal"]
    assert str(data.setas) == "xy"
    assert data["Run"] == "row deletion"


@pytest.mark.parametrize("row", [0, 2, 4, -1, -5])
def test_keep_single_row(row):
    """An inverted scalar row index keeps that row, including its mask."""
    data = make_data()
    values = data.data.data[[row]].copy()
    mask = data.data.mask[[row]].copy()
    assert data.del_rows(row, invert=True) is data
    assert_retained(data, values, mask)


@pytest.mark.parametrize("row", [5, -6])
def test_invalid_single_row_preserves_data(row):
    """Reject an out-of-range row before deleting any data."""
    data = make_data()
    values = data.data.data.copy()
    mask = data.data.mask.copy()
    with pytest.raises(IndexError):
        data.del_rows(row, invert=True)
    assert_retained(data, values, mask)


@pytest.mark.parametrize("bounds", [(1, 3), (3, 1)])
@pytest.mark.parametrize("invert", [False, True])
def test_range_deletion(bounds, invert):
    """Range endpoints are inclusive and their order does not affect selection."""
    data = make_data()
    rows = [1, 2, 3] if invert else [0, 4]
    values = data.data.data[rows].copy()
    mask = data.data.mask[rows].copy()
    assert data.del_rows("Position", bounds, invert=invert) is data
    assert_retained(data, values, mask)
