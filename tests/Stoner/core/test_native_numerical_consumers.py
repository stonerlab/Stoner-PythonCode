"""Protect the first numerical consumer batch independently of the public data getter."""

import re

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.methods import search_index
from Stoner.core.storage import DataStorage


@pytest.fixture
def measurement():
    """Provide distinguishable duplicate channels, exclusions and typed metadata."""
    values = np.ma.array([[0, 10, 100], [1, 11, 101], [2, 12, 102], [3, 13, 103], [4, 14, 104]],
                         dtype=np.int16, mask=False, fill_value=-9)
    values.mask[1, 1] = values.mask[2, 2] = values.mask[4, 0] = True
    data = Data.from_storage(DataStorage.from_numpy(values, headers=["field", "signal", "signal"], roles="xye"))
    data.metadata["run{String}"] = "001"
    data.metadata["history"] = [1]
    return data


@pytest.fixture(params=["snapshot", "native"])
def exposed(request, measurement, monkeypatch):
    """Exercise the same measurement with current and prospective native property types."""
    if request.param == "native":
        descriptor = Data.data
        monkeypatch.setattr(Data, "data", property(lambda obj: obj._require_storage_owner().frame, descriptor.fset))
    return measurement


def test_array_export_keeps_snapshot_contract_and_annotations(exposed):
    """Keep explicit array reads detached and read-only even with native public storage."""
    result = exposed.asarray()
    assert type(result) is np.ma.MaskedArray
    assert not result.flags.writeable and not result.mask.flags.writeable
    assert result.column_headers == ["field", "signal", "signal"]
    assert result.setas == list("xye")
    np.testing.assert_array_equal(result.i, np.arange(5))
    assert result.fill_value == -9 and result.data[1, 1] == 11 and result.mask[1, 1]
    result.setflags(write=True)
    result.data[0, 0] = 80
    assert exposed[0, 0] == 0


@pytest.mark.parametrize("columns, positions, shape", [(None, [0, 1, 2], (2, 3)),
                                                       (1, [1], (2,)), ([1], [1], (2,)),
                                                       ("signal", [1, 2], (2, 2)),
                                                       (re.compile("signal"), [1, 2], (2, 2)),
                                                       ([2, 1, 2], [2, 1, 2], (2, 3)),
                                                       (slice(1, None), [1, 2], (2, 2))])
def test_search_keeps_rectangular_masks_raw_values_and_shapes(exposed, columns, positions, shape):
    """Select duplicate/repeated columns positionally and preserve every exclusion."""
    before = exposed.export_storage()
    result = exposed.search("field", (0, 2), columns=columns)
    expected = before.to_numpy()[[1, 2]][:, positions]
    if len(shape) == 1:
        expected = expected[:, 0]
    assert type(result) is np.ma.MaskedArray and result.shape == shape
    assert result.dtype == np.dtype("int16") and result.fill_value == -9
    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, expected.mask)
    result.data[...] = 70
    result.mask = False
    pd.testing.assert_frame_equal(exposed.export_storage().values, before.values)
    np.testing.assert_array_equal(exposed.mask, before.excluded)
    assert exposed.column_ids == tuple(column.id for column in before.schema)
    assert exposed.metadata["run"] == "001"


@pytest.mark.parametrize("selector, rows", [(2, [2]), (np.int16(2), [2]), (np.float32(2), [2]),
                                          ([0, 1, 3], [0, 1, 3]),
                                          (np.array([1, 3]), [1, 3]),
                                          (lambda x, row: row.i in (0, 3), [0, 3]),
                                          ([True, False, True, False, False], [0, 2]),
                                          ([True, False], [0]),
                                          (np.array([False, True]), [1]), ([], [])])
def test_search_selection_forms_and_short_boolean_padding(exposed, selector, rows):
    """Keep numeric/callback selection and honour positional Boolean lists and arrays."""
    result = exposed.search("field", selector)
    expected = exposed.to_numpy()[rows, :]
    assert result.shape == (len(rows), 3)
    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, expected.mask)


def test_search_predicates_preserve_exclusions_and_interval_endpoints(exposed):
    """Retain masked predicates, the open-lower/closed-upper interval and inversion."""
    predicate = search_index(exposed, "field", (0, 2))
    np.testing.assert_array_equal(predicate[:4], [False, True, True, False])
    legacy_pair = search_index(exposed, "field", [0, 2])
    np.testing.assert_array_equal(legacy_pair.data, predicate.data)
    np.testing.assert_array_equal(legacy_pair.mask, predicate.mask)
    assert np.ma.getmaskarray(predicate)[4]
    inverted = search_index(exposed, "field", [True, False], invert=True)
    np.testing.assert_array_equal(inverted, [False, True, True, True, True])
    exact = exposed.search("field", 1.1, accuracy=0.2, columns=0)
    np.testing.assert_array_equal(exact, [1])
    empty = exposed.search("field", 99, columns=[1, 2])
    assert empty.shape == (0, 2) and empty.dtype == np.dtype("int16")
    with pytest.raises(IndexError):
        exposed.search("field", 1, columns=[])


def test_search_assembles_result_with_one_full_export(exposed, monkeypatch):
    """Do not repeatedly convert a whole table while assembling several output columns."""
    original = Data.to_numpy
    calls = []

    def counted(obj, **kwargs):
        calls.append(obj)
        return original(obj, **kwargs)

    monkeypatch.setattr(Data, "to_numpy", counted)
    result = exposed.search("field", (0, 3), columns=[2, 1, 2])
    assert result.shape == (3, 3) and len(calls) == 1 and calls[0] is exposed


def test_columns_preserve_masks_reset_and_iteration_lifetime(exposed):
    """Capture mask eligibility once while returning current, detached column values."""
    columns = list(exposed.columns())
    assert len(columns) == 3
    for position, result in enumerate(columns):
        assert type(result) is np.ma.MaskedArray and not result.flags.writeable
        np.testing.assert_array_equal(result.data, exposed.column(position).data)
        np.testing.assert_array_equal(result.mask, np.asarray(exposed.mask)[:, position])
    assert list(exposed.columns(reset=True)) == []
    assert list(exposed.columns(not_masked=True)) == []
    exposed.mask[:, 0] = False
    exposed.mask[:, 2] = False
    iterator = exposed.columns(not_masked=True)
    first = next(iterator)
    exposed.mask[1, 2] = True
    exposed[0, 2] = 900
    second = next(iterator)
    assert first[0] == 0 and second[0] == 900 and second.mask[1]
    with pytest.raises(StopIteration):
        next(iterator)


def test_column_extraction_keeps_duplicate_resolution_and_readonly_results(exposed):
    """Retain all duplicate matches and make first-column selection explicit."""
    single = exposed.column(exposed.find_col("signal", first=True))
    multiple = exposed // re.compile("signal")
    assert single.shape == (5,) and multiple.shape == (5, 2)
    np.testing.assert_array_equal(exposed.column("signal"), multiple)
    assert single.mask[1] and multiple.mask[2, 1]
    assert not single.flags.writeable and not multiple.flags.writeable


def test_selected_consumers_do_not_read_public_data(measurement, monkeypatch):
    """Require explicit conversion boundaries instead of accepting a lucky property coercion."""
    other = measurement.clone
    descriptor = Data.data

    def forbidden(obj):
        pytest.fail("Numerical consumer read the public data property")

    monkeypatch.setattr(Data, "data", property(forbidden, descriptor.fset))
    assert measurement.asarray().shape == (5, 3)
    assert measurement.column("signal").shape == (5, 2)
    assert measurement.search("field", (0, 2), columns=[1, 2]).shape == (2, 2)
    assert len(list(measurement.columns())) == 3
    assert measurement == other


@pytest.mark.parametrize("change, equal", [("none", True), ("hidden", True), ("roles", True),
                                         ("value", False), ("header", False), ("metadata", False)])
def test_equality_keeps_positional_mask_header_and_metadata_rules(exposed, change, equal):
    """Keep current equality semantics independently of native labels and schema identities."""
    other = Data.from_storage(exposed.export_storage())
    if change == "hidden":
        other._storage_owner.frame.iloc[1, 1] = 999
    elif change == "roles":
        other.setas = "xyy"
    elif change == "value":
        other[0, 1] = 300
    elif change == "header":
        other.column_headers[1] = "different"
    elif change == "metadata":
        other.metadata["run{String}"] = "002"
    assert (exposed == other) is equal
    assert (exposed == 1) is False
    assert exposed.metadata["run"] == "001"


def test_equality_exports_each_operand_once_and_shortcuts_shape(measurement, monkeypatch):
    """Bound numerical conversion count and avoid conversion for incompatible geometry."""
    other = measurement.clone
    shorter = Data(np.ones((1, 3)), column_headers=list(measurement.column_headers))
    original = Data.to_numpy
    calls = []

    def counted(obj, **kwargs):
        calls.append(obj)
        return original(obj, **kwargs)

    monkeypatch.setattr(Data, "to_numpy", counted)
    assert measurement == other
    assert len(calls) == 2 and calls[0] is measurement and calls[1] is other
    calls.clear()
    assert not (measurement == shorter)
    assert calls == []


def test_equality_retains_existing_independent_nan_filtering(monkeypatch):
    """Characterise the retained equality rule without adding pandas alignment semantics."""
    left = Data(np.array([[1.0, np.nan], [2.0, 3.0]]), column_headers=["a", "b"])
    right = Data(np.array([[1.0, 2.0], [np.nan, 3.0]]), column_headers=["a", "b"])
    descriptor = Data.data
    monkeypatch.setattr(Data, "data", property(lambda obj: obj._require_storage_owner().frame, descriptor.fset))
    assert left == right
