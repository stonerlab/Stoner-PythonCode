"""Exercise native-owner validation, replacement, locking and reference lifetime."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from Stoner import Data, ImageFile, ImageStack
from Stoner.core.storage import DataStorage
from Stoner.core.storage_owner import DataOwner


@pytest.fixture
def owner():
    """Provide duplicate headers, separate roles and a recoverable excluded integer."""
    values = np.ma.array([[0, 10, 11], [1, 100, 101]], dtype=np.int16,
                         mask=[[False, False, False], [False, True, False]], fill_value=-9)
    result = DataOwner(DataStorage.from_numpy(values, headers=["field", "signal", "signal"], roles="xye"))
    result.metadata["run{String}"] = "001"
    result.metadata["history"] = [1]
    return result


def test_native_getter_returns_stored_frame_without_export_or_copy(owner, monkeypatch):
    """Read the actual ordinary DataFrame without constructing a substitute or NumPy export."""
    stored = owner._state.values

    def forbidden(*args, **kwargs):
        pytest.fail("Native access copied or exported the frame")

    monkeypatch.setattr(pd.DataFrame, "__init__", forbidden)
    monkeypatch.setattr(pd.DataFrame, "copy", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_numpy", forbidden)
    monkeypatch.setattr(DataStorage, "to_numpy", forbidden)
    assert type(owner.frame) is pd.DataFrame
    assert owner.frame is stored


def test_native_cell_edits_keep_raw_values_and_independent_exclusions(owner):
    """Honour native duplicate-name access without clearing Stoner exclusions."""
    frame = owner.frame
    assert isinstance(frame["signal"], pd.DataFrame)
    frame.iloc[1, 1] = 120
    frame.loc[0, "field"] = 5
    assert owner[0, "field"] == 5
    assert owner[1, 1] is np.ma.masked and frame.iloc[1, 1] == 120
    owner.mask[1, 1] = False
    assert owner[1, 1] == 120 and owner.to_numpy().dtype == np.dtype("int16")
    assert owner.metadata["run"] == "001" and list(owner.roles) == list("xye")


def test_replacement_preserves_side_state_and_detaches_input(owner):
    """Replace raw values without importing attrs as Stoner state or retaining input aliases."""
    old = owner.frame
    ids = owner.column_ids
    metadata, mask = owner.metadata, owner.mask
    history = metadata["history"]
    replacement = old.copy(deep=True)
    replacement.iloc[1, 1] = 300
    replacement.attrs.update(metadata={"run": "untrusted"}, excluded=np.zeros(replacement.shape, bool))
    owner.replace_pandas(replacement)
    assert owner.frame is not old and owner.frame is not replacement
    assert owner.column_ids == ids and list(owner.roles) == list("xye")
    assert owner[1, 1] is np.ma.masked and mask[1, 1]
    assert metadata["history"] is history and metadata.type("run") == "String"
    assert metadata["run"] == "001"
    package = owner.export_storage()
    assert package.fill_value == -9 and package.values.iloc[1, 1] == 300
    replacement.iloc[1, 1] = 400
    old.iloc[1, 1] = 500
    assert owner.frame.iloc[1, 1] == 300
    mask[1, 1] = False
    assert owner[1, 1] == 300


@pytest.mark.parametrize("defect", ["type", "headers", "rows", "columns", "index", "axis_name", "dtype", "extension"])
def test_invalid_replacement_leaves_current_frame_live(owner, defect):
    """Validate an entire replacement before retiring the stored frame or altering side state."""
    original = owner.frame
    before = owner.export_storage()
    replacement = original.copy(deep=True)
    if defect == "type":
        replacement = np.zeros(original.shape)
    elif defect == "headers":
        replacement.columns = ["field", "other", "signal"]
    elif defect == "rows":
        replacement = replacement.iloc[:1]
    elif defect == "columns":
        replacement = replacement.iloc[:, :2]
    elif defect == "index":
        replacement.index = pd.RangeIndex(1, 3)
    elif defect == "axis_name":
        replacement = replacement.rename_axis("observation")
    elif defect == "dtype":
        replacement = replacement.astype(float)
    else:
        replacement["field"] = replacement["field"].astype("Int16")
    with pytest.raises((TypeError, ValueError)):
        owner.replace_pandas(replacement)
    assert owner.frame is original
    pd.testing.assert_frame_equal(owner.export_storage().values, before.values)
    np.testing.assert_array_equal(owner.mask, before.excluded)
    assert owner.column_ids == tuple(column.id for column in before.schema)
    original.iloc[0, 0] = 9
    assert owner[0, 0] == 9


def corrupt(frame, defect):
    """Perform unsupported native structural changes to test the next owner boundary."""
    if defect == "headers":
        frame.columns = ["wrong", "signal", "signal"]
    elif defect == "row_labels":
        frame.index = pd.RangeIndex(1, 3)
    elif defect == "index_type":
        frame.index = pd.Index([0, 1], dtype=np.int64)
    elif defect == "rows":
        frame.loc[2] = [2, 20, 21]
    elif defect == "columns":
        frame["extra"] = np.array([1, 2], dtype=np.int16)
    elif defect == "deletion":
        frame.drop(columns="field", inplace=True)
    elif defect == "dtype":
        frame["field"] = frame["field"].astype(float)
    else:
        frame["field"] = frame["field"].astype("Int16")


@pytest.mark.parametrize("defect", ["headers", "row_labels", "index_type", "rows", "columns", "deletion",
                                    "dtype", "extension"])
def test_detectable_drift_blocks_owner_reads_and_writes(owner, defect):
    """Reject drift before reading misleading results or publishing partial side-state edits."""
    frame = owner.frame
    replacement = frame.copy(deep=True)
    mask, headers, roles, metadata = owner.mask, owner.headers, owner.roles, owner.metadata
    original_mask = np.asarray(mask)
    original_schema = list(owner._state.schema)
    corrupt(frame, defect)
    for operation in (lambda: owner.frame, owner.validate, owner.to_numpy, owner.export_storage,
                      lambda: owner[0, 0], lambda: owner.__setitem__((0, 0), 9),
                      lambda: mask[0, 0], lambda: mask.__setitem__((0, 0), True),
                      lambda: list(headers), lambda: headers.__setitem__(0, "new"),
                      lambda: list(roles), lambda: roles.__setitem__(0, "y"),
                      lambda: metadata["run"], lambda: metadata.__setitem__("run", "new"),
                      lambda: owner.replace_pandas(replacement), lambda: owner.take(rows=[0])):
        with pytest.raises(ValueError, match="Native DataFrame state is inconsistent"):
            operation()
    assert owner._state.values is frame and owner._state.schema == original_schema
    np.testing.assert_array_equal(owner._state.excluded, original_mask)
    assert owner._state.metadata["run"] == "001"


def test_data_entry_points_reject_detectable_drift(owner):
    """Guard the current public snapshot, shape, selection and schema interfaces too."""
    data = Data.from_storage(owner.export_storage())
    data._storage_owner.frame.columns = ["wrong", "signal", "signal"]
    for operation in (lambda: data.data, lambda: data.shape, lambda: data.dtype, lambda: len(data),
                      lambda: data[0, 0], lambda: data.column("field"), lambda: data.find_col("field"),
                      lambda: data.setas.shape, lambda: data.setas.to_list(), data.to_numpy, data.export_storage):
        with pytest.raises(ValueError, match="Native DataFrame state is inconsistent"):
            operation()


@pytest.mark.parametrize("operation", ["assignment", "numpy", "pandas", "replacement", "structural"])
def test_reacquired_frame_retires_previous_values_after_commits(owner, operation):
    """Do not let writes to old root references mutate committed or structurally changed state."""
    previous = owner.frame
    if operation == "assignment":
        owner[0, 0] = 7
    elif operation == "numpy":
        with owner.edit_numpy() as draft:
            draft[0, 0] = 7
    elif operation == "pandas":
        with owner.edit_pandas() as draft:
            draft.iloc[0, 0] = 7
    elif operation == "replacement":
        owner.replace_pandas(previous)
    else:
        owner.take(rows=[1, 0], columns=[2, 0, 1])
    current = owner.frame
    expected = owner.to_numpy(masked=False)
    assert current is not previous
    previous.iloc[0, 0] = 90
    np.testing.assert_array_equal(owner.to_numpy(masked=False), expected)


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_transactions_block_new_native_references_and_replacement(owner, context):
    """Keep owner locks and draft rollback without promising control of previously returned objects."""
    previous = owner.frame
    with pytest.raises(RuntimeError, match="abort"):
        with getattr(owner, context)() as draft:
            for operation in (lambda: owner.frame, lambda: owner.replace_pandas(previous),
                              lambda: owner.mask.__setitem__((0, 0), True)):
                with pytest.raises(RuntimeError, match="active editing"):
                    operation()
            if context == "edit_numpy":
                draft[0, 0] = 7
            else:
                draft.iloc[0, 0] = 7
            raise RuntimeError("abort")
    assert owner.frame is previous and owner[0, 0] == 0
    owner.replace_pandas(previous)


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_retained_native_writes_are_outside_draft_rollback(owner, context):
    """An aborted draft cannot undo an independent write through a previously acquired frame."""
    previous = owner.frame
    with pytest.raises(RuntimeError, match="abort"):
        with getattr(owner, context)():
            previous.iloc[0, 0] = 8  # Deliberately violate the documented concurrent-write restriction.
            raise RuntimeError("abort")
    assert owner.frame is previous and owner[0, 0] == 8


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_drift_during_draft_is_rejected_before_commit_and_unlocks(owner, context):
    """Do not silently heal externally corrupted axes by committing a stale draft."""
    previous = owner.frame
    columns = previous.columns.copy()
    with pytest.raises(ValueError, match="Native DataFrame state is inconsistent"):
        with getattr(owner, context)() as draft:
            if context == "edit_numpy":
                draft[0, 0] = 7
            else:
                draft.iloc[0, 0] = 7
            previous.columns = ["wrong", "signal", "signal"]
    assert owner._state.values is previous and previous.iloc[0, 0] == 0
    previous.columns = columns  # Restore the deliberately corrupted native labels for the next valid edit.
    with owner.edit_pandas() as draft:
        draft.iloc[0, 0] = 9
    assert owner[0, 0] == 9


def test_clones_and_exports_remain_independent_of_live_native_writes(owner):
    """Detach raw values, masks and recursive typed metadata at clone/export boundaries."""
    owner.metadata["parent"] = owner
    cloned = deepcopy(owner)
    package = owner.export_storage()
    owner.frame.iloc[1, 1] = 200
    assert cloned.metadata["parent"] is cloned and cloned.metadata["run"] == "001"
    assert cloned.frame.iloc[1, 1] == package.values.iloc[1, 1] == 100
    cloned.frame.iloc[0, 0] = 3
    assert owner[0, 0] == 0


def test_shared_image_metadata_does_not_require_dataframe_validation():
    """Keep image/stack metadata proxies independent of the Data-only native validator."""
    image = ImageFile(np.ones((2, 3), dtype=np.int16))
    stack = ImageStack([image])
    for obj in (image._image_owner, stack._stack_owner):
        obj.metadata["run{String}"] = "001"
        copied = obj.metadata.copy()
        assert copied["run"] == "001" and copied.type("run") == "String"
        assert "run" in list(obj.metadata) and len(obj.metadata) > 0
        assert deepcopy(obj).metadata["run"] == "001"


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (3, 0)])
def test_native_empty_axes_preserve_fixed_dtype_and_replacement(shape):
    """Retain explicit dtype, positional geometry and IDs where pandas cannot infer them."""
    owner = DataOwner(DataStorage.from_numpy(np.empty(shape, dtype=np.uint16)))
    frame, ids = owner.frame, owner.column_ids
    owner.replace_pandas(frame)
    assert owner.frame is not frame and owner.frame.shape == shape
    assert owner.column_ids == ids and owner.to_numpy().dtype == np.dtype("uint16")
