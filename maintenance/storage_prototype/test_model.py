"""Check scientific and ownership boundaries of the isolated storage prototype."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

from Stoner import Data
from Stoner.Image import ImageFile
from Stoner.core.base import TypeHintedDict
from .model import Column, Stack, Table, resolve, role_groups

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def table():
    """Supply duplicate headers and an excluded recoverable integer."""
    result = Table(np.array([[0, 10, 11], [0, 100, 101]], dtype=np.int16),
                   ["Field", "Moment", "Moment"], "xyy")
    result.mask[1, 1] = True
    result.metadata["run{String}"] = "001"
    result.metadata["nested"] = {"history": [1, 2]}
    return result


def test_table_lossless_roundtrip(table):
    """Retain hidden integers, explicit metadata types, roles and duplicate IDs."""
    package = table.export_storage()
    restored = Table.from_storage(package)
    assert restored.column_ids == table.column_ids
    assert restored.to_numpy().dtype == np.dtype("int16")
    assert restored.metadata["run"] == "001"
    assert restored.metadata.type("run") == table.metadata.type("run")
    assert restored[1, 1] is np.ma.masked
    restored.mask[1, 1] = False
    assert restored[1, 1] == 100 and table.mask[1, 1]
    restored.metadata["nested"]["history"].append(3)
    assert table.metadata["nested"]["history"] == [1, 2]
    package.values.iloc[0, 0] = 900
    assert restored[0, 0] == table[0, 0] == 0


def test_structural_identity_and_accessor(table):
    """Move masks/roles with both duplicates and allocate IDs for repetition."""
    order = [2, 0, 1]
    selected = table.select(order)
    assert order == [2, 0, 1]
    assert "".join(col.role for col in selected._state.schema) == "yxy"
    assert selected.mask[1, 2]
    assert selected.column_ids == tuple(table.column_ids[i] for i in order)
    repeated = table.select([1, 1])
    assert repeated.column_ids[0] == table.column_ids[1]
    assert len(set(repeated.column_ids)) == 2
    package = table.export_storage()
    output = package.values.stoner_prototype.column(re.compile("Moment"), schema=package.schema)
    assert output.shape == (2, 2)
    output.iloc[0, 0] = -1
    assert table[0, 1] == 10
    with pytest.raises(ValueError):
        package.values.iloc[:, ::-1].stoner_prototype.column("Moment", schema=package.schema)


@pytest.mark.parametrize("selector, expected", [("Moment", 1), (re.compile("Moment"), [1, 2]),
                                                (-1, 2), (slice(None, None, -1), [2, 1, 0])])
def test_resolver(table, selector, expected):
    """Keep exact-name precedence and distinct compiled-pattern positions."""
    assert resolve(table._state.schema, selector) == expected


@pytest.mark.parametrize("selector, error", [("3", KeyError), ("-1", KeyError), (-4, IndexError),
                                             (3, IndexError), ("(", re.error), (None, TypeError)])
def test_resolver_errors(table, selector, error):
    """Reject unsupported selectors and ordinary positional overflow."""
    with pytest.raises(error):
        resolve(table._state.schema, selector)


def test_transaction_independence_rollback_and_locks(table):
    """Commit drafts once and reject nested owner writes or structural edits."""
    saved_metadata = table.metadata
    with table.edit_numpy() as draft:
        draft[1, 1] = 99
        with pytest.raises(RuntimeError):
            table[0, 0] = 4
        with pytest.raises(RuntimeError):
            saved_metadata["run"] = 7
        with pytest.raises(RuntimeError):
            with table.edit_numpy():
                pass
    draft[1, 1] = 17
    assert table[1, 1] == 99
    with pytest.raises(RuntimeError, match="cancel"):
        with table.edit_pandas() as frame:
            frame.iloc[0, 2] = 999
            raise RuntimeError("cancel")
    assert table[0, 2] == 11
    with pytest.raises(ValueError):
        with table.edit_pandas() as frame:
            frame.columns = ["bad", "axes", "here"]
    assert table[0, 2] == 11
    with table.edit_pandas() as frame:
        frame.iloc[0, 2] = 12
    frame.iloc[0, 2] = 13
    assert table[0, 2] == 12


@pytest.mark.parametrize("defect", ["version", "mask", "ids", "roles", "index", "dtype"])
def test_malformed_table_packages(table, defect):
    """Fail closed on malformed exports rather than losing scientific state."""
    package = table.export_storage()
    if defect == "version":
        package.version = 7
    elif defect == "mask":
        package.excluded = np.zeros((1, 3), bool)
    elif defect == "ids":
        package.schema[1] = package.schema[0]
    elif defect == "roles":
        col = package.schema[0]
        package.schema[0] = Column(col.id, col.header, "bad")
    elif defect == "index":
        package.values.index = [8, 8]
    elif defect == "dtype":
        package.values = package.values.astype(str)
    with pytest.raises((ValueError, TypeError)):
        Table.from_storage(package)
    assert table.mask[1, 1]


def test_native_import_and_numeric_boundaries():
    """Require explicit index/dtype policy and preserve unmasked missing values."""
    frame = pd.DataFrame([[0, 1], [0, 2]], columns=["Field", "Moment"], index=[7, 7])
    with pytest.raises(ValueError):
        Table.from_pandas(frame)
    table = Table.from_pandas(frame, index="discard", setas="xy")
    assert table._state.values.index.equals(pd.RangeIndex(2))
    mixed = pd.DataFrame({"x": [1, 2], "y": [1.5, 2.5]})
    with pytest.raises(TypeError):
        Table.from_pandas(mixed)
    assert Table.from_pandas(mixed, dtype=float).to_numpy().dtype == float
    with pytest.raises(TypeError):
        Table.from_pandas(pd.DataFrame({"x": pd.array([1, None], dtype="Int64")}))
    nan = Table(np.array([[np.nan, 2.0]]))
    assert not nan.to_numpy().mask.any() and np.isnan(nan.to_numpy().mean())
    nan.mask[:] = True
    assert nan.to_numpy().mean() is np.ma.masked


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_table_shapes(shape):
    """Keep empty dimensions through package round trips."""
    table = Table(np.zeros(shape, dtype=np.int16))
    result = Table.from_storage(table.export_storage()).to_numpy()
    assert result.shape == shape
    assert result.dtype == np.dtype("int16")


def test_assignment_promotion_and_hidden_values():
    """Avoid integer overflow, float truncation and accidental mask destruction."""
    table = Table(np.array([[1, 2]], dtype=np.int8))
    table[0, 0] = 1000
    assert table[0, 0] == 1000
    table[0, 1] = 2.5
    assert table[0, 1] == 2.5
    table[0, 1] = np.ma.masked
    assert table.to_numpy(masked=False)[0, 1] == 2.5
    table.mask[0, 1] = False
    assert table[0, 1] == 2.5
    with pytest.raises(TypeError):
        table[0, 0] = "bad"
    assert table[0, 0] == 1000
    unsigned = Table(np.array([[1]], dtype=np.uint16))
    unsigned[0, 0] = -2
    assert unsigned[0, 0] == -2
    complex_table = Table(np.array([[1 + 2j]]))
    assert Table.from_storage(complex_table.export_storage())[0, 0] == 1 + 2j


def test_append_keeps_exclusions(table):
    """Append positional observations without aligning repeated scan values."""
    assert table.append([[0, 120, 121]]) is table
    assert table.to_numpy().shape == (3, 3)
    assert table.mask[1, 1]
    assert not table.mask[2, 1]


def test_real_fixture_multi_y_weighted_fit():
    """Compare coefficients/covariances after a real fixture crosses storage."""
    source = Data(ROOT / "tests/Stoner/CoreTest.dat", setas="xy")
    x, y = np.asarray(source.column(0)), np.asarray(source.column(1))
    sigma = np.linspace(0.1, 0.3, len(x))
    values = np.column_stack([x, y, sigma, y * 2, sigma * 2])
    mask = np.zeros(values.shape, bool)
    mask[2, 1] = True
    mask[4, 3] = True
    legacy = Data(values, setas="xyeye")
    legacy.mask = mask
    table = Table(values, ["x", "y", "error", "y", "error"], "xyeye", mask)

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    for ycol, error in [(1, 2), (3, 4)]:
        baseline = np.ma.array(legacy.data, copy=True)
        converted = Table.from_storage(table.export_storage()).to_numpy()
        outputs = []
        for array in (baseline, converted):
            valid = ~np.ma.getmaskarray(array[:, [0, ycol, error]]).any(axis=1)
            outputs.append(curve_fit(quadratic, array.data[valid, 0], array.data[valid, ycol],
                                     sigma=array.data[valid, error], absolute_sigma=True))
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-12, atol=1e-12)
    legacy_frame = source.to_pandas()
    imported = Table.from_pandas(legacy_frame)
    np.testing.assert_array_equal(imported.to_numpy().data, source.data.data)
    assert "".join(col.role for col in imported._state.schema) == "xy"


def test_absolute_error_positions_and_vector_roles():
    """Retain the second x group's error and repeated dependent role families."""
    table = Table(np.zeros((2, 8)), roles="xdyexdye")
    second = role_groups(table._state.schema)[1]
    assert second["xcol"] == 4 and second["xerr"] == 5
    assert second["ycol"] == [6] and second["yerr"] == [7]
    table = Table(np.zeros((2, 10)), roles="xdyezfuvwy")
    group = role_groups(table._state.schema)[0]
    assert group["ycol"] == [2, 9]
    assert [group[key] for key in ("zcol", "zerr", "ucol", "vcol", "wcol")] == [[4], [5], [6], [7], [8]]


@pytest.fixture
def stack():
    """Supply unequal integer frames with meaningful per-frame metadata."""
    result = Stack([np.ones((2, 3), np.uint16), np.ones((3, 4), np.uint16)], ["small", "large"])
    result["small"].metadata["Field"] = 10
    return result


def test_stack_roundtrip_padding_and_handles(stack):
    """Exclude padding and preserve writeback by identity across structural edits."""
    restored = Stack.from_storage(stack.export_storage())
    assert restored.to_numpy().count() == 18
    assert restored.to_numpy().mean() == 1
    dataset = restored.export_storage().dataset
    assert dataset.stoner_prototype.mean() == 1
    assert dataset.intensity.mean() == 0.75
    item = restored["small"]
    assert item.to_numpy().shape == (2, 3)
    item[1, 2] = 77
    assert restored["small"][1, 2] == 77
    assert stack["small"][1, 2] == 1
    item.mask[1, 2] = True
    copied = Stack.from_storage(restored.export_storage())
    copied["small"].mask[1, 2] = False
    assert copied["small"][1, 2] == 77
    assert restored["small"].mask[1, 2]
    restored.reorder([1, 0])
    assert item.metadata["Field"] == 10
    item[0, 0] = 12
    assert restored[1][0, 0] == 12
    restored.insert(0, np.full((4, 5), 8, np.uint16))
    assert item[0, 0] == 12
    restored.reorder([0, 1])
    with pytest.raises(ReferenceError):
        item.to_numpy()


@pytest.mark.parametrize("defect", ["version", "padding", "extents", "ids", "complex", "dimensions"])
def test_stack_invalid_packages(stack, defect):
    """Reject structural corruption before an imported stack becomes visible."""
    package = stack.export_storage()
    if defect == "version":
        package.version = 9
    elif defect == "padding":
        package.dataset.excluded.values[0, 2, 0] = False
    elif defect == "extents":
        package.dataset.valid_height.values[0] = 9
    elif defect == "ids":
        package.frames[1].id = package.frames[0].id
    elif defect == "complex":
        package.dataset["intensity"] = package.dataset.intensity.astype(complex)
    elif defect == "dimensions":
        package.dataset = package.dataset.transpose("frame", "x", "y")
    with pytest.raises((ValueError, TypeError)):
        Stack.from_storage(package)


def test_stack_context_rollback_and_parent_lock(stack):
    """Validate drafts and prevent stale parent or sibling handle commits."""
    with stack[0].edit_numpy() as draft:
        draft[0, 0] = 8
        with pytest.raises(RuntimeError):
            stack[1][0, 0] = 9
        with pytest.raises(RuntimeError):
            stack.insert(0, np.ones((2, 2)))
    draft[0, 0] = 11
    assert stack[0][0, 0] == 8
    with pytest.raises(ValueError):
        with stack.edit_xarray() as dataset:
            dataset.excluded.values[:] = False
    assert stack.to_numpy().count() == 18
    with pytest.raises(ValueError):
        with stack.edit_xarray() as dataset:
            dataset.coords["x"] = [8, 9, 10, 11]
    with stack.edit_xarray() as dataset:
        dataset.intensity.values[1, 0, 0] = 20
    dataset.intensity.values[1, 0, 0] = 21
    assert stack[1][0, 0] == 20


def test_real_image_with_explicit_calibration():
    """Retain real fixture intensity and supplied calibration through interchange."""
    image = ImageFile(ROOT / "tests/Stoner/Image/coretestdata/im1_annotated.png")
    array = np.ma.array(image.image, copy=True)
    stack = Stack([array])
    package = stack.export_storage()
    height, width = array.shape
    package.dataset = package.dataset.assign_coords(
        physical_y=(("frame", "y"), np.arange(height)[None, :] * 0.5),
        physical_x=(("frame", "x"), np.arange(width)[None, :] * 0.25))
    for coord in ("physical_y", "physical_x"):
        package.dataset[coord].attrs["units"] = "um"
    restored = Stack.from_storage(package)
    assert restored.export_storage().dataset.identical(package.dataset)
    with pytest.raises(NotImplementedError):
        restored.insert(0, array)
    np.testing.assert_array_equal(restored[0].to_numpy().data, array.data)
    np.testing.assert_allclose(restored[0].to_numpy().mean(), array.mean())


def test_empty_stack_and_fill_values():
    """Retain empty stacks and dtype-compatible fill sentinels."""
    assert Stack.from_storage(Stack([]).export_storage()).to_numpy().shape == (0, 0, 0)
    values = np.ma.array([[1, 2]], mask=[[False, True]], fill_value=-9)
    stack = Stack([values])
    assert Stack.from_storage(stack.export_storage())[0].to_numpy().fill_value == -9


def test_uncopyable_metadata_reports_key():
    """Do not silently share an arbitrary mutable object on failed copying."""
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")
    metadata = TypeHintedDict()
    metadata["device"] = Uncopyable()
    with pytest.raises(TypeError, match="device"):
        Table(np.ones((2, 2)), metadata=metadata)
