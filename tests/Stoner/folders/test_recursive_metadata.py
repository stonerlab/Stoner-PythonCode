"""Check group traversal and opt-in recursive metadata slicing."""

import numpy as np
import pytest

from Stoner import Data, DataFolder


def hierarchy():
    """Create direct members at three levels and two sibling branches."""
    root = DataFolder(readlist=False)

    def add(folder, value):
        data = Data()
        data["value"] = value
        folder.append(data)

    add(root, 1)
    root.add_group("first")
    first = root.groups["first"]
    add(first, 2)
    first.add_group("nested")
    add(first.groups["nested"], 3)
    root.add_group("second")
    add(root.groups["second"], 4)
    return root


@pytest.mark.parametrize("only_terminal,expected", [(True, [3, 4]), (False, [3, 2, 4, 1])])
def test_walk_members(only_terminal, expected):
    root = hierarchy()
    visited = []
    root.walk_groups(lambda data, trail: visited.append((data["value"], tuple(trail))),
                     only_terminal=only_terminal)
    assert [value for value, _ in visited] == expected
    paths = {3: ("first", "nested"), 2: ("first",), 4: ("second",), 1: ()}
    assert all(trail == paths[value] for value, trail in visited)
    assert len(root) == 1 and len(root.groups) == 2
    if only_terminal:
        default_visited = []
        result = root.walk_groups(lambda data, trail: default_visited.append(data["value"]))
        assert default_visited == expected
        assert result == []  # Preserve the existing non-terminal root return contract.


def test_walk_groups_includes_nonterminal_groups():
    root = hierarchy()
    visited = []
    root.walk_groups(lambda group, trail: visited.append(tuple(trail)), group=True, only_terminal=False)
    assert visited == [("first", "nested"), ("first",), ("second",), ()]


def test_replacing_terminal_groups_does_not_revisit_parent():
    root = hierarchy()
    visited = []

    def replace(group, trail):
        visited.append(tuple(trail))
        return Data()

    root.walk_groups(replace, group=True, replace_terminal=True)
    assert visited == [("first", "nested"), ("second",)]
    assert list(root.groups) == ["first"]


def test_recursive_slice_preserves_default_and_order():
    root = hierarchy()
    assert root.metadata.slice("value", values_only=True) == [1]
    assert root.metadata.slice("value", recurse=True, values_only=True) == [3, 2, 4, 1]
    assert root.slice_metadata("value", recurse=True) == [3, 2, 4, 1]
    assert root.metadata.slice("value", recurse=True) == [{"value": x} for x in (3, 2, 4, 1)]
    assert len(root) == 1 and len(root.groups) == 2


@pytest.mark.parametrize("output", ["array", "data", "frame"])
def test_recursive_numeric_formats(output):
    result = hierarchy().metadata.slice("value", recurse=True, output=output)
    values = result.values if output == "frame" else result.data if output == "data" else result
    np.testing.assert_array_equal(np.asarray(values).ravel(), [3, 2, 4, 1])


def test_recursive_missing_keys_are_resolved_across_all_members():
    root = hierarchy()
    root[0]["root_only"] = 9
    with pytest.raises(KeyError):
        root.metadata.slice("root_only", recurse=True)
    assert root.metadata.slice("root_only", recurse=True, mask_missing=True, output="list") == [None, None, None, 9]


def test_recursive_slice_of_empty_folder():
    root = DataFolder(readlist=False)
    assert root.metadata.slice(recurse=True) == []
    with pytest.raises(KeyError):
        root.metadata.slice("missing", recurse=True)


def test_recursive_wildcard_and_vector_metadata():
    root = hierarchy()

    def add_vector(data, trail):
        data["vector"] = [data["value"], data["value"] + 10]

    root.walk_groups(add_vector, only_terminal=False)
    result = root.metadata.slice("v*", recurse=True)
    assert result == [
        {"value": value, "vector[0]": value, "vector[1]": value + 10}
        for value in (3, 2, 4, 1)
    ]
