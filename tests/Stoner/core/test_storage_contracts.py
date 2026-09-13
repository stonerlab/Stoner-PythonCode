"""Characterise storage boundaries before replacing DataArray.

Tests named ``legacy`` record observed quirks, not requirements to reproduce those
quirks in the new backend. See STORAGE_CONTRACTS.md for the migration decisions.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from Stoner import Data


@pytest.fixture
def measurement():
    """Return distinguishable columns, including duplicate and literal-pattern names."""
    return Data(
        np.arange(24).reshape(4, 6),
        column_headers=["Field", "Moment", "Moment error", "Moment", "2", "["],
        setas="xyey..",
    )


@pytest.mark.parametrize(
    "selector, expected",
    [
        ("Moment", 1), ("Moment error", 2), ("err", 2), ("Mom", 1),
        ("2", 4), ("[", 5), (-1, 5), (0, 0),
        (slice(1, 5, 2), [1, 3]), (["Field", -1], [0, 5]),
        (re.compile("field", re.IGNORECASE), [0]),
    ],
)
def test_column_resolution(measurement, selector, expected):
    """Resolve exact names before patterns and retain positional selection forms."""
    assert measurement.find_col(selector) == expected


@pytest.mark.parametrize(
    "selector, exception",
    [("absent", KeyError), (re.compile("absent"), KeyError), ("(", re.error),
     (6, IndexError), (None, TypeError)],
)
def test_column_resolution_errors(measurement, selector, exception):
    """Distinguish missing names, invalid patterns and invalid positions."""
    with pytest.raises(exception):
        measurement.find_col(selector)


def test_force_list_and_compiled_pattern_shape(measurement):
    """Keep scalar and multi-column result shapes explicit."""
    assert measurement.find_col("Field", force_list=True) == [0]
    assert measurement.column("Field").shape == (4,)
    assert measurement.column(re.compile("Field")).shape == (4, 1)
    assert measurement[0, "Field"] == 0
    assert measurement[0].shape == (6,)


def test_metadata_precedence_and_column_assignment(measurement):
    """Keep string metadata assignment distinct from explicit column assignment."""
    original = measurement.column("Moment").copy()
    measurement["Moment"] = "metadata value"
    assert measurement["Moment"] == "metadata value"
    np.testing.assert_array_equal(measurement.column("Moment"), original)
    measurement[:, "Moment"] = [30, 31, 32, 33]
    np.testing.assert_array_equal(measurement.column(1), [30, 31, 32, 33])
    np.testing.assert_array_equal(measurement.column(3), [3, 9, 15, 21])
    assert measurement.metadata["Moment"] == "metadata value"


def test_row_iteration_and_slice_writeback(measurement):
    """Record row iteration and basic-view versus advanced-copy mutation."""
    np.testing.assert_array_equal(np.array(list(measurement)), measurement.data)
    column = measurement.column("Moment")
    column[1] = 100
    assert measurement[1, 1] == 100
    rows = measurement[1:3]
    rows[0, 0] = 200
    assert measurement[1, 0] == 200
    selected = measurement[[1, 2]]
    selected[0, 0] = 300
    assert measurement[1, 0] == 200


def test_mask_unmask_and_nan_are_distinct():
    """Preserve excluded values and distinguish exclusion from an unmasked NaN."""
    data = Data(np.array([[0, 10], [1, 100], [2, 30]], dtype=np.int16), setas="xy")
    data.mask = np.zeros(data.shape, dtype=bool)
    data.mask[1, 1] = True
    assert data.dtype == np.dtype("int16")
    assert data.column(1).mean() == 20
    assert data.data.data[1, 1] == 100
    data.mask[1, 1] = False
    assert data[1, 1] == 100
    floating = Data(np.array([[0, np.nan], [1, 10.0]]), setas="xy")
    assert not np.ma.getmaskarray(floating.data).any()
    assert np.isnan(floating.column(1).mean())


def test_clone_owns_values_masks_headers_roles_and_metadata(measurement):
    """Make a clone independent across numerical and descriptive state."""
    measurement.mask = np.zeros(measurement.shape, dtype=bool)
    measurement.mask[0, 1] = True
    measurement.metadata["history"] = [1, 2]
    clone = measurement.clone
    clone[1, 0] = -1
    clone.mask[0, 1] = False
    clone.column_headers[0] = "Changed"
    clone.setas = "......"
    clone.metadata["history"].append(3)
    assert measurement[1, 0] == 6
    assert measurement.mask[0, 1]
    assert measurement.column_headers[0] == "Field"
    assert measurement.setas.to_string() == "xyey.."
    assert measurement.metadata["history"] == [1, 2]


def test_roles_cover_errors_vectors_and_multiple_dependents():
    """Resolve all role families without treating role assignments as unique."""
    data = Data(np.zeros((3, 10)), setas="xdyezfuvwy")
    columns = data.setas._get_cols()
    expected = {"xcol": 0, "xerr": 1, "ycol": [2, 9], "yerr": [3],
                "zcol": [4], "zerr": [5], "ucol": [6], "vcol": [7], "wcol": [8]}
    for key, value in expected.items():
        assert columns[key] == value
    assert columns.axes == 6


def test_real_fixture_pandas_roundtrip():
    """Retain ordinary values and roles through current interchange."""
    data = Data(Path(__file__).parents[1] / "CoreTest.dat", setas="xy")
    restored = Data(data.to_pandas())
    np.testing.assert_array_equal(restored.data, data.data)
    assert restored.column_headers == data.column_headers
    assert restored.setas.to_string() == "xy"


@pytest.mark.skipif(pd.__version__.split(".")[0] != "3", reason="Observed with the pandas 3 accessor lifecycle")
def test_legacy_pandas_metadata_loss(measurement):
    """Record missing user metadata after round-trip on the baseline pandas version."""
    measurement.metadata["Run"] = 7
    restored = Data(measurement.to_pandas())
    assert measurement.metadata["Run"] == 7
    assert "Run" not in restored.metadata


def test_legacy_duplicate_regex_and_negative_overflow(measurement):
    """Record duplicate-name collapse and modulo indexing for later API review."""
    assert measurement.find_col(re.compile("Moment")) == [1, 2, 1]
    assert measurement.find_col(-7) == 5


def test_legacy_numeric_string_fallback(measurement):
    """Expose the broken non-negative numeric fallback rather than promise it works."""
    with pytest.raises(AttributeError):
        measurement.find_col("3")
    with pytest.raises(KeyError):
        measurement.find_col("-1")


def test_legacy_column_slice_roles(measurement):
    """Record headers following selection while roles retain their old prefix."""
    selected = measurement[:, [3, 0, 2]]
    assert selected.column_headers == ["Moment", "Field", "Moment error"]
    np.testing.assert_array_equal(selected, np.asarray(measurement.data)[:, [3, 0, 2]])
    assert selected.setas.to_string() == "xye"  # Correct selected roles would be yxe.


def test_legacy_masked_pandas_export(measurement):
    """Record loss of exclusions and hidden values in the existing pandas bridge."""
    measurement.mask = np.zeros(measurement.shape, dtype=bool)
    measurement.mask[0, 1] = True
    frame = measurement.to_pandas()
    assert np.isnan(frame.iloc[0, 1])
    restored = Data(frame)
    assert not np.ma.getmaskarray(restored.data).any()
    assert np.isnan(restored[0, 1])
    assert measurement.data.data[0, 1] == 1


def test_sort_preserves_mask_roles_and_returns_self():
    """Sort measurements positionally with their exclusions, retaining chaining."""
    data = Data(np.array([[2, 20], [0, 0], [1, 10]]), column_headers=["Field", "Moment"], setas="xy")
    data.mask = np.zeros(data.shape, dtype=bool)
    data.mask[0, 1] = True
    assert data.sort("Field") is data
    np.testing.assert_array_equal(data.data.data, [[0, 0], [1, 10], [2, 20]])
    np.testing.assert_array_equal(data.mask, [[False, False], [False, False], [False, True]])
    assert data.column_headers == ["Field", "Moment"]
    assert data.setas.to_string() == "xy"


def test_legacy_structural_edits_drop_state():
    """Record mask loss on reorder/add and role loss on column deletion."""
    data = Data(np.arange(9).reshape(3, 3), column_headers=["Field", "Moment", "Error"], setas="xye")
    data.mask = np.zeros(data.shape, dtype=bool)
    data.mask[0, 1] = True
    reordered, extended, deleted = data.clone, data.clone, data.clone
    order = [2, 0, 1]
    assert reordered.reorder_columns(order) is reordered
    assert reordered.setas.to_string() == "exy"
    assert not np.ma.getmaskarray(reordered.data).any()
    assert order == [0, 1]  # The caller's list is also consumed in part.
    assert extended.add_column([10, 11, 12], header="Extra", setas="y") is extended
    assert extended.setas.to_string() == "xyey"
    assert not np.ma.getmaskarray(extended.data).any()
    assert deleted.del_column(0) is deleted
    assert deleted.mask[0, 0]
    assert deleted.setas.to_string() == ".."


def test_legacy_second_x_group_error_index():
    """Record the relative x-error index in the second group for later correction."""
    data = Data(np.zeros((2, 8)), setas="xdyexdye")
    group = data.setas._get_cols(startx=4)
    assert group.xcol == 4
    assert group.ycol == [6]
    assert group.yerr == [7]
    assert group.xerr == 1  # The corresponding absolute column would be 5.


def test_legacy_masked_scalar_returns_fill_value(measurement):
    """Record scalar extraction losing the exclusion flag."""
    measurement.mask = np.zeros(measurement.shape, dtype=bool)
    measurement.mask[0, 1] = True
    measurement.data.fill_value = -999
    assert measurement[0, 1] == -999
    assert measurement.data.data[0, 1] == 1
