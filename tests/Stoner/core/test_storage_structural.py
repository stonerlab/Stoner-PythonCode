"""Validate the default frame backend and atomic structural mutation contracts."""

from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Stoner import Data


def measurement():
    """Make duplicate columns and recoverable exclusions distinguishable."""
    result = Data(np.arange(12, dtype=np.int16).reshape(4, 3),
                  column_headers=["x", "signal", "signal"], setas="xye")
    result.mask[1, 2] = True
    result.metadata["run{String}"] = "001"
    return result


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint64", "float32", "complex128"])
def test_ordinary_constructor_owns_frame(dtype):
    """Construction detaches numeric input while retaining its homogeneous dtype."""
    source = np.ones((3, 2), dtype=dtype)
    data = Data(source, setas="xy")
    source[0, 0] = 0
    assert data[0, 0] == 1
    assert data.dtype == np.dtype(dtype)
    assert isinstance(data.__dict__["_storage_owner"]._state.values, pd.DataFrame)
    assert "_data" not in data.__dict__ and "_metadata" not in data.__dict__


def test_copy_constructors_preserve_ids_and_metadata_cycles():
    """Copy construction retains IDs in an independent namespace, including cycles."""
    source = measurement()
    source.metadata["parent"] = source
    for cloned in (Data(source), copy(source), deepcopy(source), source.clone):
        assert cloned.column_ids == source.column_ids
        assert cloned.metadata["parent"] is cloned
        assert cloned.metadata["run"] == "001"
        cloned[0, 0] = 70
        assert source[0, 0] == 0


def test_repeated_columns_delete_and_sort():
    """Keep column state together and regenerate only repeated selection IDs."""
    data = measurement()
    ids = data.column_ids
    order = [2, 0, 2, 1]
    data.reorder_columns(order)
    assert order == [2, 0, 2, 1]
    assert data.column_ids[:2] == (ids[2], ids[0])
    assert data.column_ids[2] not in ids
    assert data.column_ids[3] == ids[1]
    assert data.setas == "exey"
    assert data.mask[1, 0] and data.mask[1, 2]
    data.del_column(0).sort("x", reverse=True)
    assert data.setas == "xey"
    assert data.mask[2, 1]
    assert data.export_storage().values.index.equals(pd.RangeIndex(4))


def test_insert_replace_and_rows_keep_raw_values():
    """Preserve masks on both existing and inserted data and retain replaced IDs."""
    data = measurement()
    ids = data.column_ids
    new = np.ma.array([20, 21, 22, 23], mask=[False, True, False, False])
    data.add_column(new, header="extra", index=1, setas="z")
    assert data.column_ids[0] == ids[0] and data.column_ids[2:] == ids[1:]
    assert data.mask[1, 1] and data.mask[1, 3]
    inserted_id = data.column_ids[1]
    data.add_column(new + 10, header="changed", index=1, replace=True, setas="y")
    assert data.column_ids[1] == inserted_id
    row = np.ma.array([50, 51, 52, 53], mask=[False, False, True, False])
    data.insert_rows(1, row)
    assert data.mask[1, 2] and data.mask[2, 3]
    assert data.to_numpy(masked=False)[1, 2] == 52
    data.del_rows(1)
    assert data.mask[1, 3]


def test_row_append_occurrences_and_left_metadata():
    """Match each duplicate header separately, preserving mask and metadata types."""
    left = measurement()
    right = Data(np.ma.array([[90, 91, 92]], mask=[[False, True, False]]),
                 column_headers=["signal", "x", "signal"], setas="yxe")
    right.metadata["run"] = "different"
    right.metadata["new{String}"] = "002"
    ids = left.column_ids
    left += right
    np.testing.assert_array_equal(left.to_numpy(masked=False)[-1], [91, 90, 92])
    assert left.mask[-1, 0] and left.mask[1, 2]
    assert left.column_ids == ids
    assert left.metadata["run"] == "001" and left.metadata["new"] == "002"


def test_column_concat_remaps_clone_ids():
    """Concatenating a clone must not create duplicate owner column identities."""
    source = measurement()
    result = source & source.clone
    assert result.column_ids[:3] == source.column_ids
    assert len(set(result.column_ids)) == 6
    assert result.setas == "xyexye"
    assert result.mask[1, 2] and result.mask[1, 5]


def test_copy_into_and_inplace_metadata_identity():
    """Retain recursive ownership through copy helpers and structural commits."""
    from Stoner.tools import copy_into

    source = measurement()
    source.metadata["parent"] = source
    target = Data()
    copy_into(source, target)
    assert target.column_ids == source.column_ids
    assert target.metadata["parent"] is target
    target.sort(0)
    target += Data(np.ones((1, 3)), column_headers=list(target.column_headers))
    assert target.metadata["parent"] is target
    target += {"x": 20}
    assert target.metadata["parent"] is target


def test_saved_role_interface_supports_slices_and_headers():
    """Resolve compound assignments atomically through durable role interfaces."""
    data = measurement()
    roles = data.setas
    roles[1:] = "ey"
    assert roles == "xey"
    roles.column_headers = ["Field", "Error", "Signal"]
    data.del_column(1)
    assert roles.to_list() == ["x", "y"]
    assert roles.column_headers == ["Field", "Signal"]
    roles[:] = "yy"
    assert roles["#y"] == [0, 1]


def test_row_snapshots_keep_scalar_positions_and_roles():
    """Iteration and indexed row snapshots retain the observation's position."""
    data = measurement()
    for i, row in enumerate(data.rows()):
        assert row.i == i
        assert row.setas == "xye"
        with pytest.raises(ValueError):
            row[0] = 99
    assert data[-1].i == 3
    data._set_mask(lambda row: 0 < row.i < 3, invert=True)
    np.testing.assert_array_equal(np.asarray(data.mask)[:, 0], [True, False, False, True])


def test_structural_failures_leave_all_state_unchanged():
    """Bad schemas, indices and active drafts cannot partly modify an owner."""
    data = measurement()
    before = data.export_storage()
    for operation in (lambda: data.add_column([1, 2, 3, 4], header="bad", setas="!"),
                      lambda: data.reorder_columns([0, 9]),
                      lambda: data.insert_rows(0, [1, 2]),
                      lambda: data.del_rows(99),
                      lambda: data.reorder_columns([1, 0, 2], headers_too=False)):
        with pytest.raises((ValueError, IndexError)):
            operation()
        np.testing.assert_array_equal(data.to_numpy(masked=False), before.to_numpy(masked=False))
        assert data.column_ids == tuple(c.id for c in before.schema)
        np.testing.assert_array_equal(data.mask, before.excluded)
    with data.edit_numpy():
        for operation in (lambda: data.del_rows(0), lambda: data.add_column([1]*4),
                          lambda: data.insert_rows(0, [1, 2, 3])):
            with pytest.raises(RuntimeError):
                operation()


def test_zero_sized_axes_and_real_loader_constructor():
    """Keep empty schema/row dimensions explicit and finish loader construction in a frame."""
    assert Data().shape == (0, 0)
    data = measurement()
    data.del_rows(slice(None))
    assert data.shape == (0, 3)
    ids = data.column_ids
    data.insert_rows(0, [1, 2, 3])
    assert data.shape == (1, 3) and data.column_ids == ids
    data.del_column([0, 1, 2])
    assert data.shape == (1, 0) and len(data) == 1
    real = Data(Path(__file__).parents[1] / "CoreTest.dat", setas="xy")
    assert "_storage_owner" in real.__dict__ and len(real) > 0
    assert real.setas == "xy"


def test_pandas_constructor_uses_explicit_import_policy():
    """Native frame construction rejects implicit index discard and mixed dtypes."""
    assert Data(pd.DataFrame([[1, 2]], columns=["x", "y"])).shape == (1, 2)
    with pytest.raises(ValueError):
        Data(pd.DataFrame([[1]], columns=["x"], index=[9]))
    with pytest.raises(TypeError):
        Data(pd.DataFrame({"x": [1], "y": [2.5]}))
