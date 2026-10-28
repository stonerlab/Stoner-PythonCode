"""Protect matching precedence, consumer cardinality and multi-column mutations."""

import re

import numpy as np
import pytest
from scipy.integrate import cumulative_simpson

from Stoner import Data, __datapath__
from Stoner.core import storage


@pytest.fixture
def measurement():
    """Return two distinguishable channels with duplicated names and independent masks."""
    x = np.arange(10, dtype=float)
    result = Data(x, 2*x + 1, 3*x + 2, np.full(10, 0.2), setas="xyye",
                  column_headers=["Field", "Signal", "Signal", "sigma"])
    result.mask[4, 2] = True
    result.metadata["run{String}"] = "001"
    return result


@pytest.mark.parametrize("selector, expected", [("Signal", [1, 2]), ("Sig", [1, 2]),
                                               ("sigma", 3)])
def test_exact_and_substring_do_not_compile_regex(measurement, monkeypatch, selector, expected):
    """Keep successful literal lookup independent of regex compilation."""
    schema = measurement.export_storage().schema
    compiled = []
    original = re.compile

    def record(pattern, *args, **kwargs):
        compiled.append(pattern)
        return original(pattern, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.re, "compile", record)
        result = storage.resolve_columns(schema, selector)
    assert result == expected and compiled == []


def test_successful_glob_does_not_also_compile_fallback_regex(measurement, monkeypatch):
    """Permit fnmatch's internal translation while deferring fallback regex work."""
    schema = measurement.export_storage().schema
    compiled = []
    original = re.compile

    def record(pattern, *args, **kwargs):
        compiled.append(pattern)
        return original(pattern, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.re, "compile", record)
        positions = storage.resolve_columns(schema, "Sign?l")
    assert positions == [1, 2] and "Sign?l" not in compiled


@pytest.mark.parametrize("selector, expected", [("sig", 1), ("sig*", 1),
                                               (re.compile("sig", re.I), [0, 1]),
                                               ("[", 2), ("2", 3)])
def test_case_literal_brackets_and_numeric_headers(selector, expected):
    """Use platform-independent case and keep unmatched brackets/numbers literal."""
    data = Data(np.ones((3, 4)), column_headers=["Signal", "signal", "H [T]", "2"])
    assert data.find_col(selector) == expected


@pytest.mark.parametrize("method, args, kwargs", [
    ("closest", (2,), {"xcol": "Signal"}),
    ("search_index", ("Signal", 2), {}),
    ("max", ("Signal",), {}),
    ("min", ("Signal",), {}),
    ("mean", ("Signal",), {}),
    ("std", ("Signal",), {}),
    ("clip", ((0, 2),), {"column": "Signal"}),
    ("del_rows", ("Signal", 2), {}),
    ("interpolate", ([1, 2],), {"xcol": "Signal"}),
    ("polyfit", (), {"xcol": "Signal", "ycol": 0}),
    ("normalise", (0,), {"base": "Signal"}),
    ("SG_Filter", (), {"xcol": "Signal", "col": 0}),
    ("spline", (), {"xcol": 0, "ycol": "Signal"}),
    ("peaks", (), {"xcol": 0, "ycol": "Signal"}),
    ("find_peaks", (), {"xcol": 0, "ycol": "Signal"}),
    ("threshold", (2,), {"col": "Signal", "xcol": 0}),
    ("add_column", (np.ones(10),), {"index": "Signal"}),
    ("swap_column", (("Signal", 0),), {}),
])
def test_scalar_inputs_reject_ambiguous_explicit_columns_before_mutation(measurement, method, args, kwargs):
    """Require an explicit choice before algorithms can flatten or truncate channels."""
    before = measurement.export_storage()
    with pytest.raises(ValueError, match="exactly one.*Signal.*1.*2"):
        getattr(measurement, method)(*args, **kwargs)
    after = measurement.export_storage()
    assert after.schema == before.schema and after.metadata == before.metadata
    np.testing.assert_array_equal(after.to_numpy(masked=False), before.to_numpy(masked=False))
    np.testing.assert_array_equal(after.excluded, before.excluded)


def test_role_writes_use_positions_instead_of_resolving_duplicate_names_again(measurement):
    """Write each assigned y once even when both channels have the same header."""
    values = np.column_stack((np.arange(10), -np.arange(10)))
    ids = measurement.column_ids
    measurement.y = values
    np.testing.assert_array_equal(measurement.to_numpy(masked=False)[:, 1:3], values)
    assert measurement.column_ids == ids and measurement.setas == "xyye"


def test_role_sequence_length_error_is_atomic(measurement):
    """Validate an expanded role assignment before applying any of its values."""
    before = measurement.export_storage()
    with pytest.raises(ValueError, match="count"):
        measurement.setas["Signal"] = ["z"]
    assert measurement.export_storage().schema == before.schema
    measurement.setas[[1, 2, 1]] = ["y", "z", "e"]
    assert measurement.setas == "xeze"


def test_mapping_append_overlap_uses_mapping_order_and_rolls_back_failure(measurement):
    """Broadcast matches, then let a later positional key override one channel."""
    before = measurement.export_storage()
    with pytest.raises(ValueError):
        measurement += {"Signal": [1, 2, 3]}
    assert len(measurement) == len(before.values)
    np.testing.assert_array_equal(measurement.to_numpy(masked=False), before.to_numpy(masked=False))
    measurement += {"Signal": 9, 2: 12}
    np.testing.assert_array_equal(measurement.to_numpy(masked=False)[-1, 1:3], [9, 12])
    np.testing.assert_array_equal(measurement.mask[-1], [True, False, False, True])


def test_duplicate_cleanup_and_count_include_every_exact_match(measurement):
    """Reduce all selected channels and retain only first occurrences on request."""
    np.testing.assert_array_equal(measurement.count(col="Signal"), [10, 9])
    np.testing.assert_array_equal(measurement.count(value=2, col="Signal"), [0, 1])
    assert np.isscalar(measurement.count(col=1))
    ids = measurement.column_ids
    assert measurement.del_column("Signal", duplicates=True) is measurement
    assert measurement.column_ids == (ids[0], ids[1], ids[3])


@pytest.mark.parametrize("replace", [True, False])
def test_normalise_uses_every_target_and_preserves_mask_and_roles(measurement, replace):
    """Normalise reversed target positions against one independently selected base."""
    original = measurement.export_storage()
    assert measurement.normalise([2, 1], replace=replace, scale=(0., 1.)) is measurement
    positions = [1, 2] if replace else [1, 3]
    for position in positions:
        np.testing.assert_allclose(measurement.column(position).compressed(),
                                   np.linspace(0, 1, 10)[~measurement.mask[:, position]])
    assert measurement.metadata["run"] == "001"
    if replace:
        assert measurement.column_ids == tuple(column.id for column in original.schema)
        np.testing.assert_array_equal(measurement.mask, original.excluded)


def test_smooth_and_integrate_keep_both_channels(measurement):
    """Compare multi-channel results with explicit independent calculations."""
    from scipy.signal import convolve

    measurement.mask = False
    before = measurement.to_numpy()
    smooth = measurement.smooth(xcol=0, ycol="Signal", size=3, result=False)
    assert smooth.shape == (10, 2)
    for i, position in enumerate([1, 2]):
        np.testing.assert_allclose(smooth[:, i], convolve(before[:, position], np.ones(3), mode="same") / 3)
    expected = [cumulative_simpson(before[:, i], x=before[:, 0])[-1] for i in [1, 2]]
    actual = measurement.integrate(xcol=0, ycol="Signal", output="result")
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(measurement.to_numpy(), before)


def test_arithmetic_first_operand_does_not_infer_an_uncertainty_pair(measurement):
    """Keep documented arithmetic first selection separate from explicit error tuples."""
    expected = measurement.column(1) + measurement.column(0)
    measurement.add("Signal", 0, header="sum")
    np.testing.assert_array_equal(measurement.column("sum"), expected)


def test_real_instrument_headers_match_as_literals_before_regex():
    """Retain actual facility header text containing units and TeX metacharacters."""
    data = Data(__datapath__ / "Cu_resistivity_vs_T.txt")
    positions = data.find_col("T (K)", force_list=True)
    assert len(positions) == 1 and data.column_headers[positions[0]] == "T (K)"
    header = next(name for name in data.column_headers if "rho" in name)
    assert data.find_col(header) == data.find_col("rho", first=True)
