"""Protect typed metadata and independent storage at the loader copy boundary."""

import numpy as np
import pytest

from Stoner import Data
from Stoner.tools import copy_into


@pytest.mark.parametrize("value", ["001", "False", "1e3", "2026-01-01"])
def test_copy_into_preserves_explicit_strings(value):
    source = Data()
    source.metadata["label{String}"] = value
    destination = Data()
    assert copy_into(source, destination) is destination
    assert destination["label"] == value
    assert type(destination["label"]) is str
    assert destination.metadata.type("label") == source.metadata.type("label")


def test_copy_into_preserves_native_types_and_independent_storage():
    class Destination(Data):
        """A destination whose class must be preserved."""

    source = Data(np.array([[1., 2.], [3., 4.]]), setas="xy")
    source.mask[0, 1] = True
    source.filename = "source.dat"
    for key, value in {"integer": 2, "boolean": False, "nothing": None, "real": 1.25}.items():
        source[key] = value
    shared = ["001", {"array": np.array([1., 2.])}]
    source["nested"] = {"first": shared, "second": shared}
    destination = Destination()
    copy_into(source, destination)
    assert type(destination) is Destination
    assert destination.filename == source.filename
    assert destination.setas == source.setas
    np.testing.assert_array_equal(destination.mask, source.mask)
    assert destination["nested"]["first"] is destination["nested"]["second"]
    for key in ("integer", "boolean", "nothing", "real", "nested"):
        assert type(destination[key]) is type(source[key])
        assert destination.metadata.type(key) == source.metadata.type(key)
    destination["nested"]["first"][1]["array"][0] = 99
    assert source["nested"]["first"][1]["array"][0] == 1
    destination.data[1, 0] = 99
    assert source.data[1, 0] == 3
