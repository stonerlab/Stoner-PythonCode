"""Keep ordered metadata lookup contracts intact across fallback refactoring."""

import pytest

from Stoner.core.base import RegexpDict, TypeHintedDict


@pytest.mark.parametrize("mapping_type", [RegexpDict, TypeHintedDict])
def test_exact_metadata_key_precedes_regular_expression(mapping_type):
    """Treat literal punctuation as an exact key before considering regex matches."""
    mapping = mapping_type({"Field T": 1, "Field (T)": 2})

    assert mapping["Field (T)"] == 2
    assert mapping.__lookup__("Field (T)", multiple=True) == ["Field (T)"]


@pytest.mark.parametrize("mapping_type, expected", [
    (RegexpDict, 2),
    (TypeHintedDict, {"signal.main": 2, "signal.secondary": 3}),
])
def test_metadata_prefix_matches_precede_search_matches(mapping_type, expected):
    """Prefer complete prefix matches while retaining their stored ordering."""
    mapping = mapping_type({"tail.signal": 1, "signal.main": 2, "signal.secondary": 3})

    assert mapping["signal"] == expected
    assert mapping.__lookup__("signal", multiple=True) == ["signal.main", "signal.secondary"]
    assert mapping[r"\.signal$"] == 1


@pytest.mark.parametrize("mapping_type", [RegexpDict, TypeHintedDict])
def test_integer_metadata_lookup_uses_sorted_keys(mapping_type):
    """Keep positional lookup independent of the dictionary's insertion order."""
    mapping = mapping_type({"z": 1, "a": 2})

    assert mapping[0] == 2
    assert mapping[-1] == 1


@pytest.mark.parametrize("mapping_type", [RegexpDict, TypeHintedDict])
def test_metadata_exact_and_invalid_pattern_failures(mapping_type):
    """Reject incomplete exact names and invalid unmatched patterns with KeyError."""
    mapping = mapping_type({"signal.main": 2})

    with pytest.raises(KeyError, match="exact match requested"):
        mapping.__lookup__("signal", exact=True)
    with pytest.raises(KeyError):
        mapping["("]


def test_typed_metadata_fallback_keeps_explicit_string_hint():
    """Retain explicitly typed values when resolving their canonical key by a pattern."""
    mapping = TypeHintedDict()
    mapping["run{String}"] = "001"

    assert mapping["ru"] == "001"
    assert mapping.type("run") == "String"
