"""Check the existing configuration inheritance contract for new groups."""

from os import path

import numpy as np
import pytest

from Stoner import Data, DataFolder


def test_group_inherits_configuration(tmp_path):
    """Groups inherit declared options and constructor attributes, not members."""
    folder = DataFolder(readlist=False, pattern="*.dat", experiment="sample A")
    folder.root = str(tmp_path)
    folder.extra_args = {"filetype": "CSVFile"}
    folder.append(Data(np.array([[1, 2]]), column_headers=["x", "y"]))
    assert folder.add_group("child") is folder
    child = folder.groups["child"]
    assert type(child) is type(folder)
    assert list(child.pattern) == list(folder.pattern)
    assert child.extra_args == folder.extra_args
    assert child.experiment == "sample A"
    assert child.key == "child"
    assert child.root == path.join(str(tmp_path), "child")
    assert len(child) == 0
    assert len(folder) == 1
    assert not child.groups
    child.experiment = "changed child"
    folder.add_group("child")
    assert folder.groups["child"] is child
    assert child.experiment == "changed child"


def test_group_inherits_current_constructor_configuration():
    """Declared constructor attributes propagate their current values."""
    folder = DataFolder(readlist=False, experiment="initial")
    folder.experiment = "updated"
    folder.add_group("child")
    assert folder.groups["child"].experiment == "updated"


def test_added_attributes_propagate_to_nested_groups():
    """Later public attributes propagate with their latest assigned value."""
    folder = DataFolder(readlist=False)
    folder.experiment = "initial"
    folder.experiment = "updated"
    folder._private_cache = object()
    folder.add_group("child")
    child = folder.groups["child"]
    assert child.experiment == "updated"
    assert "experiment" in child._instance_attrs
    assert not hasattr(child, "_private_cache")
    child.add_group("grandchild")
    assert child.groups["grandchild"].experiment == "updated"
    assert not ({"args", "kwargs", "executor", "directory", "pattern"} & folder._instance_attrs)


@pytest.mark.parametrize("constructor", [False, True])
def test_deleted_attribute_is_not_restored(constructor):
    """Deletion removes tracking and prevents constructor values reappearing."""
    folder = DataFolder(readlist=False, **({"experiment": "initial"} if constructor else {}))
    folder.experiment = "updated"
    del folder.experiment
    assert "experiment" not in folder._instance_attrs
    with pytest.raises(AttributeError):
        del folder.experiment
    folder.add_group("child")
    assert not hasattr(folder.groups["child"], "experiment")
    folder.experiment = "re-added"
    folder.add_group("second")
    assert folder.groups["second"].experiment == "re-added"


def test_properties_are_not_tracked_as_added_attributes():
    """Declared properties retain their setter and failure behaviour."""
    folder = DataFolder(readlist=False)
    folder.debug = True
    assert "debug" not in folder._instance_attrs
    with pytest.raises(AttributeError):
        folder.defaults = {}
    assert "defaults" not in folder._instance_attrs
    with pytest.raises(AttributeError, match="protected"):
        del folder.debug
    assert folder.debug is True


def test_each_deletion_uses_folder_attribute_tracking():
    """The existing each deletion route removes tracked folder attributes."""
    folder = DataFolder(readlist=False)
    folder.experiment = "sample A"
    del folder.each.experiment
    assert not hasattr(folder, "experiment")
    assert "experiment" not in folder._instance_attrs
    folder.add_group("child")
    assert not hasattr(folder.groups["child"], "experiment")
