"""Check native row previews at Stoner's scientific and ownership boundaries."""

import numpy as np
import pandas as pd
import pytest

from Stoner import Data
from Stoner.core.storage import DataStorage
from Stoner.core.storage_owner import DataOwner


@pytest.fixture
def measurement():
    """Keep duplicate columns, raw excluded values and metadata cycles visible."""
    values = np.ma.array(np.arange(24, dtype=np.int16).reshape(8, 3), fill_value=-99)
    values[1, :] = np.ma.masked
    values[6, 2] = np.ma.masked
    data = Data.from_storage(DataStorage.from_numpy(values, headers=["x", "signal", "signal"], roles="xye"))
    data.metadata["run{String}"] = "001"
    data.metadata["nested"] = {"values": [1, 2]}
    data.metadata["parent"] = data
    data.filename = "preview-source.dat"
    return data


@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("count", [None, 0, 1, 5, 8, 12, -1, -5, -8, -12, np.int64(2)])
def test_preview_positions_and_scientific_state(measurement, method, count):
    """Retain exact positional observations and all associated scientific state."""
    source = measurement
    native = source.data
    original = source.to_numpy()
    n = 5 if count is None else int(count)
    positions = list(range(8))[:n] if method == "head" else list(range(8))[-n:] if n else []
    result = getattr(source, method)() if count is None else getattr(source, method)(count)
    actual = result.to_numpy()
    np.testing.assert_array_equal(actual.data, original.data[positions])
    np.testing.assert_array_equal(actual.mask, original.mask[positions])
    assert actual.dtype == original.dtype and actual.fill_value == -99
    assert list(result.column_headers) == ["x", "signal", "signal"]
    assert result.setas == "xye" and result.column_ids == source.column_ids
    assert result.metadata["run"] == "001" and result.metadata.type("run") == source.metadata.type("run")
    assert result.metadata["parent"] is result
    assert result.filename == source.filename
    assert result.data.index.equals(pd.RangeIndex(len(positions)))
    assert result is not source and source.data is native
    np.testing.assert_array_equal(source.to_numpy().data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_is_independent_in_both_directions(measurement, method):
    """Detach raw values, exclusions and nested metadata, including full previews."""
    result = getattr(measurement, method)(99)
    result[1, 0] = 300
    result.mask[1, 0] = False
    result.metadata["nested"]["values"].append(3)
    result.column_headers[0] = "changed"
    assert measurement.data.iloc[1, 0] == 3 and measurement.mask[1, 0]
    assert measurement.metadata["nested"]["values"] == [1, 2]
    assert measurement.column_headers[0] == "x"
    measurement[0, 0] = 400
    assert result[0, 0] == 0


@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 0)])
def test_preview_empty_axes(method, shape):
    """Keep rectangular schema and row counts when an axis has no values."""
    data = Data(np.empty(shape, dtype=np.float32))
    result = getattr(data, method)(2)
    assert result.shape == (min(2, shape[0]), shape[1])
    assert result.dtype == data.dtype and result.column_ids == data.column_ids


@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("dtype", [np.bool_, np.uint64, np.float32, np.complex128])
def test_preview_preserves_numeric_domain(method, dtype):
    """Avoid precision loss or coercion of supported homogeneous numeric types."""
    values = np.ones((3, 2), dtype=dtype)
    if dtype == np.uint64:
        values[1, 0] = 2**63 + 7
    elif dtype == np.complex128:
        values[1, 0] = 1 + 3j
    elif dtype == np.float32:
        values[1, 0] = np.nan
    result = getattr(Data(values), method)(3)
    np.testing.assert_array_equal(result.to_numpy(masked=False), values)
    assert result.dtype == values.dtype


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_subclass_and_guards(measurement, method):
    """Preserve the concrete class and reject schema drift and active drafts."""
    class Measurement(Data):
        """Represent a user-specialised Data class."""

    assert type(getattr(Measurement(np.ones((3, 2))), method)(2)) is Measurement
    with measurement.edit_pandas():
        with pytest.raises(RuntimeError, match="editing"):
            getattr(measurement, method)()
    measurement.data.columns = ["bad", "signal", "signal"]
    with pytest.raises(ValueError, match="Native DataFrame"):
        getattr(measurement, method)()


@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("count", [True, np.bool_(False), 1.5, "2", None])
def test_preview_rejects_noninteger_counts(measurement, method, count):
    """Reject ambiguous counts without changing the source."""
    native = measurement.data
    with pytest.raises(TypeError):
        getattr(measurement, method)(count)
    assert measurement.data is native and measurement.shape == (8, 3)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_rejects_unoffered_controls(measurement, method):
    """Keep selection and mutation explicit rather than forwarding keywords."""
    for options in ({"where": np.ones(8, dtype=bool)}, {"inplace": True}):
        with pytest.raises(TypeError):
            getattr(measurement, method)(**options)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_preserves_shared_attribute_metadata_graph(measurement, method):
    """Use one memo for ancillary attributes, owner aliases and recursive metadata."""
    owner = measurement._require_storage_owner()
    shared = measurement.metadata["nested"]
    measurement.metadata["second"] = shared
    shared["self"] = shared
    shared["parent"] = measurement
    shared["frame"] = owner._state.values
    shared["excluded"] = owner._state.excluded
    object.__setattr__(measurement, "extra", shared)
    object.__setattr__(measurement, "frame_alias", owner._state.values)
    object.__setattr__(measurement, "mask_alias", owner._state.excluded)
    object.__setattr__(measurement, "saved_mask", measurement.mask)
    result = getattr(measurement, method)(2)
    selected = result._require_storage_owner()
    assert result.extra is result.metadata["nested"] is result.metadata["second"]
    assert result.extra is result.extra["self"] and result.extra["parent"] is result
    assert result.extra is not shared and result.metadata["parent"] is result
    assert result.frame_alias is result.extra["frame"] and result.frame_alias.shape == measurement.shape
    assert result.mask_alias is result.extra["excluded"] and result.mask_alias.shape == measurement.shape
    assert result.frame_alias is not selected._state.values
    assert result.mask_alias is not selected._state.excluded
    assert np.asarray(result.saved_mask).shape == (2, 3)
    np.testing.assert_array_equal(result.frame_alias.to_numpy(), measurement.data.to_numpy())
    np.testing.assert_array_equal(result.mask_alias, owner._state.excluded)
    assert not np.shares_memory(result.frame_alias.to_numpy(), measurement.data.to_numpy())
    assert not np.shares_memory(result.mask_alias, owner._state.excluded)
    result.extra["values"].append(3)
    assert shared["values"] == [1, 2]


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_copies_only_selected_numerical_rows(monkeypatch, method):
    """Forbid full-table copies, NumPy exports and owner trimming on the fast path."""
    class Measurement(Data):
        """Inherit standard copying while carrying user instance attributes."""

    source = Measurement(np.arange(30000, dtype=np.float64).reshape(10000, 3))
    object.__setattr__(source, "calibration", {"scale": [1, 2]})
    original_copy = pd.DataFrame.copy

    def frame_copy(frame, deep=True):
        if deep is True and len(frame) == 10000:
            pytest.fail("preview deep-copied the complete frame")
        return original_copy(frame, deep=deep)

    def fail(*args, **kwargs):
        pytest.fail("preview exported or trimmed the complete table")

    monkeypatch.setattr(pd.DataFrame, "copy", frame_copy)
    monkeypatch.setattr(DataOwner, "to_numpy", fail)
    monkeypatch.setattr(DataOwner, "take", fail)
    result = getattr(source, method)(5)
    assert type(result) is Measurement and result.shape == (5, 3)
    assert result.calibration == source.calibration and result.calibration is not source.calibration
    assert result.data.iloc[0, 0] == (0 if method == "head" else 29985)


@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("hook", ["clone", "deepcopy"])
def test_custom_copy_hooks_retain_clone_then_trim(method, hook):
    """Respect user copy transformations through the established fallback path."""
    calls = []

    class CustomClone(Data):
        """Transform the copied object through a custom clone property."""

        @property
        def clone(self):
            calls.append("clone")
            result = super().clone
            result[0, 0] = 123
            return result

    class CustomDeepcopy(Data):
        """Transform the copied object through a custom deepcopy hook."""

        def __deepcopy__(self, memo):
            calls.append("deepcopy")
            result = super().__deepcopy__(memo)
            result[0, 0] = 123
            return result

    cls = CustomClone if hook == "clone" else CustomDeepcopy
    source = cls(np.arange(12).reshape(4, 3))
    result = getattr(source, method)(2)
    assert type(result) is cls and result.shape == (2, 3) and calls == [hook]
    assert result[0, 0] == (123 if method == "head" else 6)
    assert source[0, 0] == 0 and source.shape == (4, 3)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_copy_failure_does_not_change_owner(measurement, method):
    """Leave authoritative state and retained references intact on copying errors."""
    class CannotCopy:
        """Fail independently of the numerical preparation."""

        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")

    object.__setattr__(measurement, "extra", CannotCopy())
    native = measurement.data
    original = measurement.to_numpy()
    with pytest.raises(RuntimeError, match="copy failed"):
        getattr(measurement, method)(2)
    assert measurement.data is native
    np.testing.assert_array_equal(measurement.to_numpy().data, original.data)
    np.testing.assert_array_equal(measurement.mask, original.mask)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_preview_retains_canonical_result_frame_policy(method):
    """Keep the old trimming policy for non-authoritative native frame details."""
    source = Data(np.arange(12).reshape(4, 3))
    native = source.data
    native.attrs["note"] = "native"
    native.columns = pd.CategoricalIndex(list(source.column_headers), name="channels")
    native.index.name = "rows"
    native.flags.allows_duplicate_labels = False
    result = getattr(source, method)(2)
    assert result.data.attrs == {} and result.data.columns.name is None
    assert result.data.index.name is None and result.data.flags.allows_duplicate_labels
    assert not isinstance(result.data.columns, pd.CategoricalIndex)
    assert source.data is native and native.attrs == {"note": "native"}
    assert native.columns.name == "channels" and native.index.name == "rows"
    assert not native.flags.allows_duplicate_labels
