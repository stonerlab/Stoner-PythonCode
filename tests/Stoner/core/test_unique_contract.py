"""Characterise Data.unique's masked NumPy results and owner read boundary."""

import re

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.storage import DataStorage


def measurement(values, mask=False, *, headers=None):
    """Import explicit fill state without normalising scientific values or types."""
    array = np.ma.array(values, mask=mask, fill_value=0)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    data = Data.from_storage(DataStorage.from_numpy(array, headers=headers))
    data.metadata["run{String}"] = "001"
    data.metadata["parent"] = data
    return data


def parts(result, first, inverse):
    """Unpack the public optional-result combinations without changing their shape."""
    if not first and not inverse:
        assert type(result) is np.ma.MaskedArray
        return result, None, None
    assert isinstance(result, tuple) and len(result) == 1 + first + inverse
    values = result[0]
    assert type(values) is np.ma.MaskedArray
    indices = result[1] if first else None
    mapping = result[-1] if inverse else None
    for extra in result[1:]:
        assert type(extra) is np.ndarray and extra.dtype.kind == "i"
    return values, indices, mapping


@pytest.mark.parametrize("first,inverse", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mask,expected,excluded,indices,mapping", [
    ([False]*4, [1, 2, 3], [False]*3, [1, 3, 0], [2, 0, 2, 1]),
    ([False, True, False, True], [3, 0], [False, True], [0, 1], [0, 1, 0, 1]),
    ([True]*4, [0], [True], [0], [0, 0, 0, 0]),
])
def test_finite_values_masks_and_optional_results(first, inverse, mask, expected, excluded, indices, mapping):
    """Keep excluded observations represented without exposing their raw keys."""
    data = measurement([3, 1, 3, 2], mask)
    native = data.data
    owner = data._require_storage_owner()
    metadata = owner._state.metadata
    schema = data.column_ids
    result = data.unique(0, first, inverse)
    values, found, groups = parts(result, first, inverse)
    np.testing.assert_array_equal(values.mask, excluded)
    np.testing.assert_array_equal(values.compressed(), np.ma.array(expected, mask=excluded).compressed())
    assert values.dtype == data.dtype and values.fill_value == 0 and values.flags.writeable
    if first:
        np.testing.assert_array_equal(found, indices)
    if inverse:
        np.testing.assert_array_equal(groups, mapping)
    assert data.data is native and data.column_ids == schema
    assert owner._state.metadata is metadata and data.metadata["parent"] is data
    assert data.metadata["run"] == "001" and data.metadata.type("run") == "String"
    np.testing.assert_array_equal(data.to_numpy(masked=False).ravel(), [3, 1, 3, 2])
    np.testing.assert_array_equal(data.mask[:, 0], mask)
    # Result data/masks and index arrays are caller-owned, even for a single group.
    values[:] = 99
    if first:
        found[:] = 0
    if inverse:
        groups[:] = 0
    np.testing.assert_array_equal(data.to_numpy(masked=False).ravel(), [3, 1, 3, 2])
    np.testing.assert_array_equal(data.mask[:, 0], mask)


@pytest.mark.parametrize("first,inverse", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("selector,indices,mapping", [
    ("key", [1, 0], [1, 0, 1]),
    ([0], [1, 0], [[1], [0], [1]]),
    ([0, 1], [1, 2, 0], [[2, 0], [1, 0], [2, 1]]),
    (re.compile("^key$"), [1, 2, 0], [[2, 0], [1, 0], [2, 1]]),
])
def test_flattened_elements_and_shaped_inverse(first, inverse, selector, indices, mapping):
    """Deduplicate selected elements, with inverse shape matching the selection."""
    data = measurement([[3, 1], [2, 1], [3, 2]], headers=["key", "key"])
    result = data.unique(selector, return_index=first, return_inverse=inverse)
    values, found, groups = parts(result, first, inverse)
    expected = [2, 3] if selector == "key" or selector == [0] else [1, 2, 3]
    np.testing.assert_array_equal(values, expected)
    if first:
        np.testing.assert_array_equal(found, indices)
        np.testing.assert_array_equal(values.data, data.column(selector).data.ravel()[found])
    if inverse:
        np.testing.assert_array_equal(groups, mapping)
        np.testing.assert_array_equal(values[groups], data.column(selector))


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.uint64, np.float16,
                                   np.float32, np.float64, np.complex64, np.complex128])
def test_homogeneous_numeric_types_and_precision(dtype):
    """Retain exact dtype and full-width integers without pandas coercion."""
    values = np.array([3, 1, 3, 2], dtype=dtype)
    if dtype == np.uint64:
        values = np.array([2**64-1, 2**53+1, 2**64-1, 2**53], dtype=dtype)
    elif np.dtype(dtype).kind == "c":
        values = np.array([1+2j, 2+1j, 1+2j, 1+1j], dtype=dtype)
    data = measurement(values)
    distinct, indices, inverse = data.unique(0, True, True)
    expected = np.sort(values[[1, 3, 0]]) if dtype != np.bool_ else np.array([True])
    np.testing.assert_array_equal(distinct, expected)
    assert distinct.dtype == values.dtype
    np.testing.assert_array_equal(values[indices], distinct)
    np.testing.assert_array_equal(distinct[inverse], values)


@pytest.mark.parametrize("first,inverse", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mask", [[False]*5, [False, True, False, False, True]])
def test_nan_and_exclusion_contract(first, inverse, mask):
    """Preserve NumPy's existing mixed NaN/mask groups rather than normalise them."""
    data = measurement([3., np.nan, 3., np.nan, 1.], mask)
    values, found, groups = parts(data.unique(0, first, inverse), first, inverse)
    if any(mask):
        np.testing.assert_array_equal(values.data, [3., np.nan, np.nan, 1.])
        np.testing.assert_array_equal(values.mask, [False, True, False, True])
        expected_indices, expected_inverse = [0, 1, 3, 4], [0, 1, 0, 2, 3]
    else:
        np.testing.assert_array_equal(values.data, [1., 3., np.nan])
        assert not values.mask.any()
        expected_indices, expected_inverse = [4, 0, 1], [1, 2, 1, 2, 0]
    if first:
        np.testing.assert_array_equal(found, expected_indices)
    if inverse:
        np.testing.assert_array_equal(groups, expected_inverse)
    assert np.isnan(data.data.iloc[1, 0]) and data.data.iloc[4, 0] == 1


def test_complex_nan_equivalence_keeps_numpy_policy():
    """Collapse complex NaN forms using NumPy's representative and inverse policy."""
    data = measurement([complex(1, np.nan), complex(np.nan, 2), 2+1j, 1+2j])
    values, indices, inverse = data.unique(0, True, True)
    np.testing.assert_array_equal(values[:2], [1+2j, 2+1j])
    assert values.size == 3 and np.isnan(values[-1]) and not values.mask.any()
    np.testing.assert_array_equal(indices, [3, 2, 0])
    np.testing.assert_array_equal(inverse, [2, 2, 1, 0])


@pytest.mark.parametrize("selector,shape", [(0, (0,)), ([0], (0, 1)), ([], (0, 0))])
def test_empty_selection_results(selector, shape):
    """Keep empty unique values, flat first indices and correctly shaped inverse."""
    data = measurement(np.empty((0, 1), dtype=np.int16))
    values, indices, inverse = data.unique(selector, True, True)
    assert values.shape == indices.shape == (0,) and inverse.shape == shape
    assert values.dtype == data.dtype


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_unique_reads_committed_values_during_draft(context):
    """Numerical reads use committed state until an explicit draft is published."""
    data = measurement([3, 1, 3, 2])
    with getattr(data, context)() as draft:
        if context == "edit_numpy":
            draft[0, 0] = 4
        else:
            draft.iloc[0, 0] = 4
        np.testing.assert_array_equal(data.unique(0), [1, 2, 3])
    np.testing.assert_array_equal(data.unique(0), [1, 2, 3, 4])


def test_unique_reads_masked_column_without_native_getter(monkeypatch):
    """Keep explicit numerical reads independent of the public native accessor."""
    data = measurement([3, 1, 3, 2])

    def fail(_):
        pytest.fail("unique used the native public getter")

    monkeypatch.setattr(Data, "data", property(fail, Data.data.fset))
    np.testing.assert_array_equal(data.unique(0), [1, 2, 3])


def test_unique_rejects_drift_and_unsupported_selection():
    """Fail before returning results from altered native schema or unknown options."""
    data = measurement([3, 1, 3, 2])
    native = data.data
    with pytest.raises(TypeError):
        data.unique(0, where=np.ones(4, dtype=bool))
    assert data.data is native
    native.columns = ["changed"]
    with pytest.raises(ValueError, match="Native DataFrame"):
        data.unique(0)


def test_raw_pandas_uniqueness_has_a_different_contract():
    """Keep the reviewed backend distinction explicit at Stoner's public boundary."""
    data = measurement([3, 1, 3, 2], [False, True, False, True])
    np.testing.assert_array_equal(pd.unique(data.data.iloc[:, 0]), [3, 1, 2])
    result = data.unique(0)
    assert result.size == 2 and result.mask.tolist() == [False, True]
