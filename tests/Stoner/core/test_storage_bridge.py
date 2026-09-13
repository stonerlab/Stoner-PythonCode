"""Exercise real Data instances constructed through the explicit frame boundary."""

import re

import numpy as np
import pandas as pd
import pytest

from Stoner import Data


@pytest.fixture
def data():
    """Import a legacy measurement into an authoritative frame-backed Data."""
    source = Data(np.array([[0, 10, 11], [1, 100, 101]], dtype=np.int16),
                  column_headers=["x", "y", "y"], setas="xyy")
    source.mask[1, 1] = True
    source.metadata["run{String}"] = "001"
    return Data.from_storage(source.export_storage())


def test_real_data_owns_only_frame_state(data):
    """Keep the public class while removing persistent legacy array state."""
    assert isinstance(data, Data)
    assert "_data" not in data.__dict__
    assert "_metadata" not in data.__dict__
    assert data.shape == (2, 3)
    assert data[1, 1] is np.ma.masked
    data.mask[1, 1] = False
    assert data[1, 1] == 100
    assert data.metadata["run"] == "001"
    assert data.metadata.type("run") == "String"


def test_assignment_and_roles_preserve_identity(data):
    """Commit schema syntax and value writes to the same owner."""
    ids = data.column_ids
    data.setas = "xye"
    assert data.setas.to_string() == "xye"
    assert data.setas(y=2, reset=False) is data
    assert data.setas.to_string() == "xyy"
    data.setas[1] = "e"
    assert data.setas.to_string() == "xey"
    data.column_headers[1] = "error"
    data[0, "error"] = 0.5
    assert data[0, 1] == 0.5
    assert data.column_ids == ids
    assert data.find_col(re.compile("y")) == [2]


def test_snapshot_and_clone_are_independent(data):
    """Prevent legacy numerical reads and cloned owners from writing through."""
    snapshot = data.data
    with pytest.raises(ValueError):
        snapshot[0, 0] = 9
    with pytest.raises(ValueError):
        data.column(0)[0] = 9
    with pytest.raises(ValueError):
        data.column(re.compile("y"))[0, 0] = 9
    cloned = data.clone
    assert "_data" not in cloned.__dict__
    assert "_metadata" not in cloned.__dict__
    assert cloned.column_ids == data.column_ids
    cloned[0, 0] = 50
    assert data[0, 0] == 0
    cloned.metadata["run{String}"] = "002"
    assert data.metadata["run"] == "001"


def test_edit_context_routes_public_writes(data):
    """Guard public schema, mask, metadata and value writes during editing."""
    roles = data.setas
    with data.edit_numpy() as draft:
        draft[0, 0] = 7
        for write in (lambda: data.__setitem__((0, 0), 4),
                      lambda: roles.__setitem__(1, "e"),
                      lambda: data.metadata.__setitem__("run", 7)):
            with pytest.raises(RuntimeError):
                write()
    draft[0, 0] = 8
    assert data[0, 0] == 7
    with data.edit_pandas() as frame:
        frame.iloc[0, 0] = 10
    assert data[0, 0] == 10


def test_whole_array_replacement_and_private_boundary(data):
    """Replace numerical state explicitly and reject private legacy replacement."""
    replacement = data.to_numpy()
    replacement[0, 1] = 40
    data.data = replacement
    assert data[0, 1] == 40
    ids = data.column_ids
    data.data = np.zeros((4, 3))
    assert data.shape == (4, 3)
    assert data.column_ids == ids
    with pytest.raises(RuntimeError, match="Private legacy"):
        data._data = np.zeros((2, 3))
    ordinary = Data(np.ones((2, 3)))
    with ordinary.edit_numpy() as draft:
        draft[0, 0] = 4
    assert ordinary[0, 0] == 4
    assert "_storage_owner" in ordinary.__dict__


def test_assignment_parser_and_failed_schema_edit(data):
    """Support compressed/mapping assignments without partially committing errors."""
    data.setas = "x2y"
    assert data.setas.to_string() == "xyy"
    data.setas({"y": [1, 2], "x": 0})
    assert data.setas.to_string() == "xyy"
    with pytest.raises(ValueError):
        data.setas = "xy!"
    assert data.setas.to_string() == "xyy"
    ids = data.column_ids
    data.setas.clear()
    assert data.setas.to_string() == "..."
    assert data.column_ids == ids
    assert "from_storage" in dir(data)


def test_native_frame_constructor_and_export(data):
    """Expose deliberate lossy export and explicit native row-index import."""
    frame = pd.DataFrame([[0, 2], [0, 3]], columns=["x", "y"], index=[9, 9])
    with pytest.raises(ValueError):
        Data.from_pandas(frame)
    imported = Data.from_pandas(frame, index="discard", setas="xy")
    assert "_storage_owner" in imported.__dict__
    assert imported.setas.to_string() == "xy"
    with pytest.warns(UserWarning, match="lossless"):
        exported = data.to_pandas(format="plain", masked="raw")
    assert list(exported.columns) == ["x", "y", "y"]
    assert exported.iloc[1, 1] == 100


def test_clone_cyclic_metadata_and_active_draft(data):
    """Retain recursive metadata and clone committed state without its edit lock."""
    data.metadata["parent"] = data
    with data.edit_numpy() as draft:
        draft[0, 0] = 90
        clone = data.clone
        assert clone.metadata["parent"] is clone
        assert clone[0, 0] == 0
        clone[0, 0] = 80
    assert data[0, 0] == 90
    assert clone[0, 0] == 80
