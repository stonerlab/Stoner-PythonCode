# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.json"""

import pytest

from Stoner.tools.json import find_parent_dicts, find_paths, flatten_json


def test_flatten_json_simple_dict():
    data = {"a": 1, "b": 2}
    result = flatten_json(data)
    assert result == {"a": 1, "b": 2}, "Simple dict flattening failed"


def test_flatten_json_nested_dict():
    data = {"a": {"b": 1}, "c": {"d": {"e": 2}}}
    result = flatten_json(data)
    assert result == {"a.b": 1, "c.d.e": 2}, "Nested dict flattening failed"


def test_flatten_json_with_list():
    data = {"a": {"b": 1}, "c": [10, 20]}
    result = flatten_json(data)
    assert result == {"a.b": 1, "c[0]": 10, "c[1]": 20}, "List flattening failed"


def test_flatten_json_scalar():
    result = flatten_json(42, parent_key="x")
    assert result == {"x": 42}, "Scalar flattening failed"


def test_flatten_json_nested_list():
    data = {"items": [{"HasData": True}, {"HasData": False}]}
    result = flatten_json(data)
    assert "items[0].HasData" in result
    assert result["items[0].HasData"] is True
    assert result["items[1].HasData"] is False


def test_flatten_json_bool_and_none():
    data = {"flag": True, "nothing": None}
    result = flatten_json(data)
    assert result == {"flag": True, "nothing": None}, "Bool and None flattening failed"


def test_find_paths_simple():
    data = {"A": {"B": {"HasData": True}}}
    result = list(find_paths(data, "HasData", True))
    assert result == [["A", "B", "HasData"]], "Simple nested path not found"


def test_find_paths_in_list():
    data = {"items": [{"HasData": True}, {"HasData": False}]}
    result = list(find_paths(data, "HasData", True))
    assert len(result) == 1
    assert result[0][-1] == "HasData", "Path through list not found correctly"


def test_find_paths_no_match():
    data = {"A": {"B": {"HasData": False}}}
    result = list(find_paths(data, "HasData", True))
    assert result == [], "find_paths returned results for non-matching value"


def test_find_paths_multiple_matches():
    data = {"A": {"HasData": True}, "B": {"HasData": True}}
    result = list(find_paths(data, "HasData", True))
    assert len(result) == 2, "find_paths should find multiple matches"


def test_find_paths_scalar_input():
    result = list(find_paths(42, "key", "val"))
    assert result == [], "find_paths with scalar should return empty"


def test_find_parent_dicts_simple():
    data = {"A": {"B": {"HasData": True, "Other": 5}}}
    result = list(find_parent_dicts(data, "HasData", True))
    assert len(result) == 1
    assert result[0] == {"HasData": True, "Other": 5}, "Parent dict not found"


def test_find_parent_dicts_in_list():
    data = {"items": [{"HasData": True}, {"HasData": False}]}
    result = list(find_parent_dicts(data, "HasData", True))
    assert len(result) == 1
    assert result[0] == {"HasData": True}, "Parent dict in list not found"


def test_find_parent_dicts_no_match():
    data = {"A": {"B": {"HasData": False}}}
    result = list(find_parent_dicts(data, "HasData", True))
    assert result == [], "find_parent_dicts returned results for non-matching value"


def test_find_parent_dicts_multiple_matches():
    data = {"A": {"HasData": True, "v": 1}, "B": {"HasData": True, "v": 2}}
    result = list(find_parent_dicts(data, "HasData", True))
    assert len(result) == 2, "find_parent_dicts should find multiple parent dicts"


def test_find_parent_dicts_scalar_input():
    result = list(find_parent_dicts(42, "key", "val"))
    assert result == [], "find_parent_dicts with scalar should return empty"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
