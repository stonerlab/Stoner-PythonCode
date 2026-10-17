"""Characterise sorting permutations and scientific state at the owner boundary."""

import re

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.storage import DataStorage


def measurement():
    """Retain duplicate headers, excluded raw keys, ties and non-finite values."""
    values = np.ma.array([
        [2, 1, 0], [1, 3, 1], [1, 2, 2], [np.nan, 0, 3], [1, 2, 4],
        [-np.inf, 5, 5], [np.inf, 6, 6], [np.nan, -1, 7], [0, 9, 8], [-0., 8, 9],
    ], fill_value=-99)
    values.mask = np.zeros(values.shape, dtype=bool)
    values.mask[2, 0] = True
    values.mask[7, :] = True
    data = Data.from_storage(DataStorage.from_numpy(values, headers=["key", "key", "row"], roles="xy."))
    data.metadata["run{String}"] = "001"
    data.metadata["parent"] = data
    return data


@pytest.mark.parametrize("order,expected", [
    ((0,), [5, 8, 9, 1, 2, 4, 0, 6, 3, 7]),
    (("key",), [5, 8, 9, 1, 2, 4, 0, 6, 3, 7]),
    ((0, 1), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
    (([0, 1],), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
    (((0, 1),), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
    ((re.compile("^key$"),), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
    ((0, 0, 1), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
    ((), [5, 9, 8, 2, 4, 1, 0, 6, 7, 3]),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_sort_permutation_and_owner_state(order, expected, reverse):
    """Move raw exclusions with observations and reverse the complete permutation."""
    data = measurement()
    original = data.to_numpy()
    ids = data.column_ids
    metadata = data._require_storage_owner()._state.metadata
    native = data.data
    positions = expected[::-1] if reverse else expected
    assert data.sort(*order, reverse=reverse) is data
    result = data.to_numpy()
    np.testing.assert_array_equal(result.data, original.data[positions])
    np.testing.assert_array_equal(result.mask, original.mask[positions])
    assert result.fill_value == -99 and result.dtype == original.dtype
    assert data.column_ids == ids and list(data.column_headers) == ["key", "key", "row"]
    assert data.setas == "xy." and data.metadata["parent"] is data
    assert data.metadata["run"] == "001" and data.metadata.type("run") == "String"
    assert data._require_storage_owner()._state.metadata is metadata
    assert data.data.index.equals(pd.RangeIndex(10))
    np.testing.assert_array_equal(native.to_numpy(), original.data)


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.uint64, np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("keys", [[0], [0, 1], [1, 0], [0, 0]])
def test_numeric_domains_match_legacy_lexsort(dtype, keys):
    """Compare supported numeric domains with the recorded pre-delegation contract."""
    rng = np.random.default_rng(47)
    values = rng.integers(0, 6, size=(31, 3)).astype(dtype)
    if dtype == np.uint64:
        values[:4, 0] = [2**64 - 1, 2**63 + 7, 2**53 + 1, 2**53]
    elif np.dtype(dtype).kind == "f":
        values[:5, 0] = [np.nan, np.inf, -np.inf, -0., 0.]
        values[5:8, 1] = np.nan
    elif np.dtype(dtype).kind == "c":
        values[:6, 0] = [1+2j, 1+1j, complex(1, np.nan), complex(np.nan, 2),
                         complex(np.nan, np.nan), -1+2j]
    expected = np.lexsort(tuple(values[:, key] for key in reversed(keys)))
    data = Data.from_storage(DataStorage.from_numpy(np.ma.array(values, fill_value=0)))
    assert data.sort(keys) is data
    np.testing.assert_array_equal(data.to_numpy(masked=False), values[expected])
    assert data.dtype == values.dtype


@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 0)])
@pytest.mark.parametrize("reverse", [False, True])
def test_empty_sort_and_no_key_identity(shape, reverse):
    """Keep empty rectangles and reverse keyless rows without inventing keys."""
    data = Data(np.empty(shape, dtype=np.int16))
    ids = data.column_ids
    assert data.sort(reverse=reverse) is data
    assert data.shape == shape and data.column_ids == ids and data.dtype == np.dtype(np.int16)


def test_default_role_priority_and_unassigned_columns():
    """Resolve x then y/z roles, or all columns when no roles are assigned."""
    values = np.array([[1, 2, 3], [0, 3, 1], [0, 2, 2], [1, 1, 0]])
    for roles, keys in [(".yx", [2, 1]), ("...", [0, 1, 2])]:
        data = Data(values, setas=roles)
        expected = np.lexsort(tuple(values[:, key] for key in reversed(keys)))
        data.sort()
        np.testing.assert_array_equal(data.to_numpy(masked=False), values[expected])


def test_comparator_receives_raw_rows_and_stays_stable():
    """Preserve the two-row comparator contract and recoverable excluded inputs."""
    data = measurement()
    seen = []

    def compare(left, right):
        seen.append((left.copy(), right.copy()))
        return int(left[2] > right[2]) - int(left[2] < right[2])

    data.sort(compare, reverse=True)
    np.testing.assert_array_equal(data.column(2).data, np.arange(9, -1, -1))
    assert seen and not any(np.ma.isMaskedArray(row) for pair in seen for row in pair)
    assert any(row[0] == 1 and row[2] == 2 for pair in seen for row in pair)
    tied = measurement()
    tied.sort(lambda left, right: 0, reverse=True)
    np.testing.assert_array_equal(tied.column(2).data, np.arange(9, -1, -1))


@pytest.mark.parametrize("selector,error", [("absent", KeyError), (True, TypeError), (99, IndexError)])
def test_bad_sort_key_is_atomic(selector, error):
    """Leave native values and ownership unchanged after selector failure."""
    data = measurement()
    native = data.data
    with pytest.raises(error):
        data.sort(selector)
    assert data.data is native


def test_comparator_failure_is_atomic():
    """Do not commit a partial permutation when a comparator fails."""
    data = measurement()
    native = data.data

    def fail(left, right):
        raise RuntimeError("comparison failed")

    with pytest.raises(RuntimeError, match="comparison failed"):
        data.sort(fail)
    assert data.data is native


def test_numeric_delegation_only_exports_at_commit(monkeypatch):
    """Resolve real keys natively; numerical export belongs to the owner commit."""
    data = measurement()
    owner = data._require_storage_owner()
    original_export = owner.to_numpy
    exports = []

    def export(*, masked=True, dtype=None):
        exports.append(masked)
        return original_export(masked=masked, dtype=dtype)

    monkeypatch.setattr(owner, "to_numpy", export)
    data.sort(0, 1)
    assert exports == [True]
    np.testing.assert_array_equal(data.column(2).data, [5, 9, 8, 2, 4, 1, 0, 6, 7, 3])


@pytest.mark.parametrize("dtype", [np.float16, np.complex128, ">i4", ">f8"])
def test_numerical_fallback_does_not_call_pandas(dtype, monkeypatch):
    """Keep unproven or incompatible dtype orderings on the numerical path."""
    values = np.array([[2, 1], [1, 3], [1, 2]], dtype=dtype)
    data = Data.from_storage(DataStorage.from_numpy(np.ma.array(values, fill_value=0)))
    expected = np.lexsort((values[:, 1], values[:, 0]))

    def fail(*args, **kwargs):
        pytest.fail("unsupported dtype delegated to pandas")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail)
    data.sort(0, 1)
    np.testing.assert_array_equal(data.to_numpy(masked=False), values[expected])


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_editing_guard_precedes_comparison(context):
    """Reject active drafts before callbacks can produce external side effects."""
    data = measurement()

    def fail(*args):
        pytest.fail("comparator called during active edit")

    with getattr(data, context)():
        with pytest.raises(RuntimeError, match="editing"):
            data.sort(fail)
        with pytest.raises(RuntimeError, match="editing"):
            data.sort(0)


def test_native_drift_precedes_sorting():
    """Reject detectable schema changes before preparing a permutation."""
    data = measurement()
    data.data.columns = ["changed", "key", "row"]
    with pytest.raises(ValueError, match="Native DataFrame"):
        data.sort(0)
