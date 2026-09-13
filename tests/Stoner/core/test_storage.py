"""Protect the production interchange boundary before migrating Data's owner."""

from copy import deepcopy
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.base import TypeHintedDict
from Stoner.core.storage import Column, DataStorage, resolve_columns


@pytest.fixture
def package():
    """Supply excluded integer values, duplicate headers and explicit metadata."""
    array = np.ma.array([[0, 10, 11], [0, 100, 101]], dtype=np.int16,
                        mask=[[False, False, False], [False, True, False]], fill_value=-999)
    metadata = TypeHintedDict()
    metadata["run{String}"] = "001"
    shared = [1, {"value": np.array([2.0])}]
    metadata["history"] = {"first": shared, "second": shared}
    return DataStorage.from_numpy(array, headers=["Field", "Moment", "Moment"], roles="xyy", metadata=metadata)


def test_lossless_copy_and_export_independence(package):
    """Retain hidden integers and typed/nested state without mutable aliases."""
    copied = package.copy()
    assert copied.schema == package.schema
    assert copied.to_numpy()[1, 1] is np.ma.masked
    assert copied.to_numpy().fill_value == -999
    assert copied.metadata["run"] == "001"
    assert copied.metadata.type("run") == "String"
    assert copied.metadata["history"]["first"] is copied.metadata["history"]["second"]
    copied.metadata["history"]["first"][1]["value"][0] = 8
    assert package.metadata["history"]["first"][1]["value"][0] == 2
    copied.excluded[1, 1] = False
    assert copied.to_numpy()[1, 1] == 100
    assert package.excluded[1, 1]
    values = package.to_numpy(masked=False)
    values[0, 0] = 40
    assert package.values.iloc[0, 0] == 0


def test_construction_detaches_inputs():
    """Factories own their raw values, exclusions and metadata."""
    array = np.ma.array([[1, 2]], mask=[[True, False]])
    package = DataStorage.from_numpy(array)
    array.data[0, 0] = 99
    array.mask[0, 0] = False
    assert package.to_numpy(masked=False)[0, 0] == 1
    assert package.excluded[0, 0]


@pytest.mark.parametrize("selector, expected", [("Moment", 1), (re.compile("Moment"), [1, 2]),
                                                (-1, 2), ([2, 0, 2], [2, 0, 2]),
                                                (slice(None, None, -1), [2, 1, 0]),
                                                (np.array([0, 2]), [0, 2])])
def test_resolution(package, selector, expected):
    """Resolve duplicate names and positional collections with stable shapes."""
    assert resolve_columns(package.schema, selector) == expected
    assert resolve_columns(package.schema, "Moment", force_list=True) == [1]


@pytest.mark.parametrize("selector, error", [("3", KeyError), (-4, IndexError), (3, IndexError),
                                             (True, TypeError), (b"Moment", TypeError), ("(", re.error),
                                             (np.array(0), TypeError), (None, TypeError)])
def test_resolver_failure_types(package, selector, error):
    """Use explicit failure types rather than legacy numeric fallback quirks."""
    with pytest.raises(error):
        resolve_columns(package.schema, selector)


def test_literal_pattern_headers():
    """Resolve exact literal regex characters before compiling a pattern."""
    schema = [Column(str(uuid4()), "["), Column(str(uuid4()), "2")]
    assert resolve_columns(schema, "[") == 0
    assert resolve_columns(schema, "2") == 1


@pytest.mark.parametrize("defect", ["version", "mask_shape", "mask_type", "index", "ids", "schema",
                                    "dtype", "frame_type", "fill_shape", "fill_fraction", "fill_overflow"])
def test_invalid_packages_fail_before_conversion(package, defect):
    """Reject edited inconsistent packages at every owning conversion boundary."""
    if defect == "version":
        package.version = True
    elif defect == "mask_shape":
        package.excluded = np.zeros((1, 3), bool)
    elif defect == "mask_type":
        package.excluded = package.excluded.astype(int)
    elif defect == "index":
        package.values.index = [7, 7]
    elif defect == "ids":
        package.schema[1] = package.schema[0]
    elif defect == "schema":
        package.schema[0] = "not a column"
    elif defect == "dtype":
        package.dtype = np.dtype(float)
    elif defect == "frame_type":
        package.values = np.zeros((2, 3))
    elif defect == "fill_shape":
        package.fill_value = [1]
    elif defect == "fill_fraction":
        package.fill_value = 1.5
    elif defect == "fill_overflow":
        package.fill_value = 999999
    for operation in (package.copy, package.to_numpy, package.to_pandas):
        with pytest.raises((TypeError, ValueError)):
            operation()


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.float64, np.complex128, bool])
def test_empty_shape_dtype_roundtrip(shape, dtype):
    """Retain dtype even where pandas has no columns from which to infer it."""
    source = DataStorage.from_numpy(np.zeros(shape, dtype=dtype))
    copied = source.copy().to_numpy()
    assert copied.shape == shape
    assert copied.dtype == np.dtype(dtype)


def test_pandas_loss_is_explicit(package):
    """Warn on lossy exports while retaining the source's recoverable values."""
    with pytest.warns(UserWarning, match="lossless"):
        legacy = package.to_pandas()
    assert legacy.columns.names == ["Headers", "Setas"]
    assert np.isnan(legacy.iloc[1, 1])
    with pytest.warns(UserWarning):
        raw = package.to_pandas(format="plain", masked="raw")
    assert list(raw.columns) == ["Field", "Moment", "Moment"]
    assert raw.iloc[1, 1] == 100
    restored = DataStorage.from_pandas(raw, mask=package.excluded, metadata=package.metadata, setas="xyy")
    assert restored.to_numpy()[1, 1] is np.ma.masked
    assert restored.metadata["run"] == "001"
    raw.iloc[0, 0] = 88
    assert restored.values.iloc[0, 0] == 0


def test_native_pandas_policies():
    """Require deliberate index discard and mixed numeric conversion."""
    frame = pd.DataFrame({"x": [1, 1], "y": [2.0, np.nan]}, index=[9, 9])
    with pytest.raises(ValueError):
        DataStorage.from_pandas(frame)
    with pytest.raises(TypeError):
        DataStorage.from_pandas(frame, index="discard")
    source = DataStorage.from_pandas(frame, index="discard", dtype=float)
    assert not source.excluded.any()
    assert np.isnan(source.to_numpy().mean())
    for dtype in ("Int64", "string", "category"):
        with pytest.raises(TypeError):
            DataStorage.from_pandas(pd.DataFrame({"x": pd.Series([1, 2], dtype=dtype)}))


def test_real_fixture_interchange():
    """Keep real scientific fixture values/roles across legacy frame conversion."""
    data = Data(Path(__file__).parents[1] / "CoreTest.dat", setas="xy")
    package = DataStorage.from_pandas(data.to_pandas())
    np.testing.assert_array_equal(package.to_numpy().data, data.data.data)
    assert [col.role for col in package.schema] == ["x", "y"]
    restored = Data(package.to_pandas())
    np.testing.assert_array_equal(restored.data, data.data)


def test_uncopyable_metadata_reports_key(package):
    """Identify an unsupported arbitrary metadata object without sharing it."""
    class Uncopyable:
        def __copy__(self):
            raise RuntimeError("uncopyable")

        def __deepcopy__(self, memo):
            raise RuntimeError("uncopyable")

    package.metadata["device"] = Uncopyable()
    with pytest.raises(TypeError, match="device"):
        package.copy()


def test_metadata_reference_to_package(package):
    """Use a shared deepcopy memo for recursive metadata references."""
    super(TypeHintedDict, package.metadata).__setitem__("owner", package)
    package.metadata.types["owner"] = "Invalid Type"
    copied = deepcopy(package)
    assert copied.metadata["owner"] is copied
