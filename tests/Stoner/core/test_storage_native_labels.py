"""Protect the Stage 8c owner-label/package boundary before public native access."""

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.storage import DataStorage
from Stoner.core.storage_owner import DataOwner


@pytest.fixture
def data():
    """Make duplicate headers, distinct roles and an excluded integer observable."""
    values = np.ma.array([[0, 10, 11], [1, 100, 101]], dtype=np.int16,
                         mask=[[False, False, False], [False, True, False]], fill_value=-9)
    result = Data.from_storage(DataStorage.from_numpy(values, headers=["field", "signal", "signal"], roles="xye"))
    result.metadata["run{String}"] = "001"
    result.metadata["history"] = [1]
    return result


def test_owner_headers_and_lossless_package_ids(data):
    """Keep useful native labels and portable identities on their respective boundaries."""
    owner = data._storage_owner
    frame = owner._state.values
    assert list(frame.columns) == ["field", "signal", "signal"]
    assert isinstance(frame["signal"], pd.DataFrame)
    assert frame.iloc[1, 1] == 100
    package = data.export_storage()
    package.validate()
    assert package.version == 1
    assert list(package.values.columns) == list(data.column_ids)
    assert package.dtype == np.dtype("int16") and package.fill_value == -9
    assert package.excluded[1, 1] and package.values.iloc[1, 1] == 100
    assert package.metadata.type("run") == "String" and package.metadata["run"] == "001"
    package.values.iloc[0, 0] = 88
    package.excluded[:] = False
    package.metadata["history"].append(2)
    assert frame.iloc[0, 0] == 0 and data.mask[1, 1]
    assert data.metadata["history"] == [1]
    restored = Data.from_storage(data.export_storage())
    assert list(restored._storage_owner._state.values.columns) == list(frame.columns)
    assert restored.column_ids == data.column_ids
    restored.mask[1, 1] = False
    assert restored[1, 1] == 100 and data.mask[1, 1]


def test_header_edits_and_roles_keep_identity(data):
    """Synchronise displayed headers atomically without relabelling portable IDs."""
    ids = data.column_ids
    data.column_headers[1] = "moment"
    data.setas = "xyy"
    assert list(data._storage_owner._state.values.columns) == ["field", "moment", "signal"]
    assert list(data.export_storage().values.columns) == list(ids)
    assert data.column_ids == ids
    with pytest.raises(TypeError):
        data.column_headers[:] = ["changed", 3, "changed"]
    assert list(data.column_headers) == ["field", "moment", "signal"]
    assert list(data._storage_owner._state.values.columns) == list(data.column_headers)


def test_named_pandas_drafts_preserve_exclusions_and_detach(data):
    """Commit raw-value edits by header, including duplicates, without clearing masks."""
    ids = data.column_ids
    with data.edit_pandas() as draft:
        draft.loc[:, "signal"] = np.array([[20, 21], [30, 31]], dtype=np.int16)
        draft.loc[0, "field"] = 5
    assert data[0, "field"] == 5
    assert data.to_numpy(masked=False)[1, 1] == 30 and data.mask[1, 1]
    assert data.column_ids == ids and data.setas == "xye"
    draft.iloc[1, 1] = 70
    assert data.to_numpy(masked=False)[1, 1] == 30
    data.mask[1, 1] = False
    assert data[1, 1] == 30


@pytest.mark.parametrize("change", ["labels", "index", "dtype", "exception"])
def test_invalid_pandas_drafts_leave_labels_and_side_state_unchanged(data, change):
    """Reject detectable schema changes and release the transaction after rollback."""
    before = data.export_storage()
    with pytest.raises((ValueError, RuntimeError)):
        with data.edit_pandas() as draft:
            draft.iloc[0, 0] = 7
            if change == "labels":
                draft.columns = ["other", "signal", "signal"]
            elif change == "index":
                draft.index = [2, 3]
            elif change == "dtype":
                draft["field"] = draft["field"].astype(float)
            else:
                raise RuntimeError("abort")
    after = data.export_storage()
    pd.testing.assert_frame_equal(after.values, before.values)
    np.testing.assert_array_equal(after.excluded, before.excluded)
    assert after.schema == before.schema
    with data.edit_pandas() as draft:
        draft.iloc[0, 0] = 6
    assert data[0, 0] == 6


def test_structural_edits_keep_duplicate_labels_masks_and_identities(data):
    """Move schema and exclusions positionally through repeated selection and row edits."""
    ids = data.column_ids
    owner = data._storage_owner
    selected = owner.select(rows=[1, 0], columns=[2, 0, 2, 1])
    assert list(selected._state.values.columns) == ["signal", "field", "signal", "signal"]
    assert selected.column_ids[:2] == (ids[2], ids[0])
    assert selected.column_ids[2] not in ids and selected.column_ids[3] == ids[1]
    assert selected.mask[0, 3]
    selected.take(rows=[1, 0], columns=[3, 1])
    selected.insert_rows(1, np.array([[80, 81]], dtype=np.int16))
    assert list(selected._state.values.columns) == ["signal", "field"]
    assert selected._state.values.index.equals(pd.RangeIndex(3))
    assert selected.mask[2, 0] and selected.to_numpy(masked=False)[2, 0] == 100
    selected.insert_columns(1, np.array([2, 3, 4], dtype=np.int16), headers=["new"], roles=["z"])
    assert list(selected._state.values.columns) == ["signal", "new", "field"]
    assert list(selected.roles) == ["y", "z", "x"] and selected.mask[2, 0]
    imported = DataOwner(selected.export_storage())
    assert imported.column_ids == selected.column_ids
    pd.testing.assert_frame_equal(imported._state.values, selected._state.values)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (3, 0)])
def test_empty_axes_keep_dtype_and_labels(shape):
    """Do not infer away the declared dtype or schema when an axis is empty."""
    data = Data(np.empty(shape, dtype=np.uint16))
    owner = data._storage_owner
    assert list(owner._state.values.columns) == list(data.column_headers)
    restored = Data.from_storage(data.export_storage())
    assert restored.shape == shape and restored.dtype == np.dtype("uint16")
    assert restored.column_ids == data.column_ids


def test_explicit_snapshot_stays_detached_with_native_storage(data):
    """Retain the explicit snapshot contract after the native getter switch."""
    snapshot = data.asarray()
    assert type(snapshot) is np.ma.MaskedArray and not snapshot.flags.writeable
    with data.edit_pandas() as draft:
        draft.iloc[0, 0] = 4
    assert snapshot[0, 0] == 0 and data[0, 0] == 4
