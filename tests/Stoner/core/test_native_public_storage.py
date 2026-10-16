"""Exercise the native public interface and explicit scientific boundaries."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from Stoner import Data, ImageFile, ImageStack
from Stoner.Image.storage import ImageStorage
from Stoner.core.storage import DataStorage


def test_dataframe_is_stored_object_and_does_not_export(monkeypatch):
    """Keep native reads cheap and raw exclusions recoverable."""
    data = Data(np.arange(6, dtype=np.int16).reshape(3, 2), column_headers=['x', 'x'])
    data.mask[1, 1] = True
    owner = data._storage_owner
    monkeypatch.setattr(owner, 'to_numpy', lambda **kw: pytest.fail('implicit NumPy export'))
    assert type(data.data) is pd.DataFrame and data.data is owner.frame
    assert data.data['x'].shape == (3, 2)
    data.data.iloc[1, 1] = 40
    assert data.data.iloc[1, 1] == 40 and data.mask[1, 1]
    ids = data.column_ids
    replacement = data.data.copy()
    data.data = replacement
    replacement.iloc[0, 0] = 90
    assert data.data.iloc[0, 0] == 0 and data.column_ids == ids
    with data.edit_pandas():
        with pytest.raises(RuntimeError):
            _ = data.data


@pytest.mark.parametrize('kind', ['image', 'stack'])
def test_dataset_is_stored_object_with_raw_masks_and_native_selection(kind, monkeypatch):
    """Expose actual datasets without exports while retaining scientific exclusions."""
    image = ImageFile(np.arange(6, dtype=np.int16).reshape(2, 3))
    obj = image if kind == 'image' else ImageStack([image])
    owner = obj._image_owner if kind == 'image' else obj._stack_owner
    attr = 'image' if kind == 'image' else 'stack'
    monkeypatch.setattr(owner, 'to_numpy', lambda **kw: pytest.fail('implicit NumPy export'))
    native = getattr(obj, attr)
    assert type(native) is xr.Dataset and native is owner._state.dataset
    key = (0, 1) if kind == 'image' else (0, 0, 1)
    native.intensity.values[key] = 20
    native.excluded.values[key] = True
    assert native.intensity.sel(x=1).size > 0
    assert native.excluded.values[key]
    replacement = native.copy(deep=True)
    setattr(obj, attr, replacement)
    replacement.intensity.values[key] = 99
    assert getattr(obj, attr).intensity.values[key] == 20
    with obj.edit_xarray():
        with pytest.raises(RuntimeError):
            getattr(obj, attr)


@pytest.mark.parametrize('change', ['dtype', 'coordinate', 'variable', 'attributes'])
def test_native_image_schema_drift_is_rejected(change):
    """Reject unsupported changes at numerical and metadata entry points."""
    image = ImageFile(np.ones((2, 3), dtype=np.int16))
    native = image.image
    if change == 'dtype':
        native['intensity'] = native.intensity.astype(float)
    elif change == 'coordinate':
        native.coords['x'] = [1, 2, 3]
    elif change == 'variable':
        native['extra'] = native.intensity
    else:
        native.attrs['calibration'] = 'changed'
    for read in (lambda: image.to_numpy(), lambda: image.shape, lambda: image.metadata['run']):
        with pytest.raises(ValueError, match='Native Dataset'):
            read()


def test_native_replacement_is_atomic_and_retained_references_retire():
    """Copy replacements, retain side state and reject altered calibration."""
    image = ImageFile(np.ones((2, 3), dtype=np.int16))
    image.metadata['run{String}'] = '001'
    native = image.image
    bad = native.assign_coords(x=[3, 4, 5])
    with pytest.raises(ValueError):
        image.image = bad
    assert image.image is native
    image[0, 0] = 5
    assert image.image is not native and native.intensity.values[0, 0] == 1
    assert image.metadata['run'] == '001'
    package = ImageStorage.from_images([np.ones((1, 2)), np.ones((2, 3))])
    stack = ImageStack.from_storage(package)
    stack.stack.excluded.values[0, 1, 2] = False
    with pytest.raises(ValueError):
        stack.to_numpy()


def test_explicit_numerical_properties_and_image_assignment_preserve_exclusions():
    """Keep numerical transpose/record APIs and masked whole-image replacements."""
    data = Data(np.arange(6).reshape(3, 2))
    assert not data.T.flags.writeable and data.records.dtype.names == tuple(data.column_headers)
    source = ImageFile(np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]]))
    target = ImageFile(np.zeros((2, 2), dtype=int))
    target.image = source
    assert target.to_numpy().data[0, 1] == 2 and target.to_numpy().mask[0, 1]


def test_empty_image_native_assignment_rejects_stack_geometry():
    """Do not turn a loader placeholder into a three-dimensional ImageFile."""
    image = ImageFile()
    stack = ImageStack([ImageFile(np.ones((2, 3)))])
    with pytest.raises(ValueError, match='y/x image Dataset'):
        image.image = stack.stack
    assert image.size == 0


def test_borrowed_frame_and_region_follow_owner_edits_and_deletion():
    """Keep stable handles and calibrated native views without promising raw writeback."""
    stack = ImageStack([ImageFile(np.arange(12).reshape(3, 4))])
    frame = stack[0]
    crop = frame.crop((1, 3, 0, 2), copy=False)
    assert crop.image.intensity.shape == (2, 2)
    crop[0, 0] = 50
    assert stack.stack.intensity.values[0, 0, 1] == 50
    assert stack[0].image.intensity.values[0, 1] == 50
    del stack[0]
    with pytest.raises(ReferenceError):
        _ = frame.image


@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('excluded', [False, True, 3])
def test_rolling_windows_preserve_masks_dtype_fill_and_live_reads(wrap, excluded):
    """Retain positional geometry without silently unmasking concatenated windows."""
    values = np.ma.array(np.arange(14, dtype=np.int16).reshape(7, 2), mask=False, fill_value=-7)
    values.mask[0, 1] = True
    data = Data.from_storage(DataStorage.from_numpy(values))
    windows = data.rolling_window(window=5, wrap=wrap, exclude_centre=excluded)
    first = next(windows)
    positions = ([5, 6] if wrap else []) + ([0, 1, 2] if not excluded else [1, 2] if excluded is True else [2])
    np.testing.assert_array_equal(first.data, values.data[positions])
    np.testing.assert_array_equal(first.mask, values.mask[positions])
    assert first.dtype == values.dtype and first.fill_value == -7 and not first.flags.writeable
    data[3, 0] = 99
    second = next(windows)
    assert 99 in second[:, 0]
    assert first.data[0, 0] != 99


def test_xor_equal_inputs_are_deterministic_boolean_comparisons():
    """Avoid uninitialised numerical output when two measurements are equal."""
    data = Data(np.arange(6).reshape(3, 2))
    result = data ^ data.clone
    assert result.dtype == bool and result.to_numpy().all()


@pytest.mark.parametrize('length, expected', [(0, []), (1, [[0, 0, 0]]),
                                            (2, [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])])
def test_short_rolling_windows_keep_single_wrap_geometry(length, expected):
    """Retain the legacy one-prefix/one-suffix rule on inputs shorter than a window."""
    data = Data(np.arange(length * 2, dtype=np.int16).reshape(length, 2))
    windows = list(data.rolling_window(window=5))
    assert len(windows) == len(expected)
    for result, positions in zip(windows, expected):
        np.testing.assert_array_equal(result, data.to_numpy()[positions])


@pytest.mark.parametrize('method', ['gaussian_filter', 'uniform_filter', 'median_filter'])
def test_neighbourhood_selection_is_rejected_without_mutation(method):
    """Do not forward an unsupported region into a neighbourhood backend."""
    image = ImageFile(np.arange(25, dtype=float).reshape(5, 5))
    before = image.to_numpy()
    with pytest.raises(TypeError, match='does not support operation selection'):
        getattr(image, method)(where=np.ones((5, 5), dtype=bool))
    np.testing.assert_array_equal(image.to_numpy(), before)


@pytest.mark.parametrize('method', ['lmfit', 'odr', 'differential_evolution'])
def test_other_fitting_engines_reject_where_explicitly(method):
    """Keep the bounded curve_fit selection API from leaking into optimiser kwargs."""
    data = Data(np.column_stack((np.arange(6.), 2 * np.arange(6.) + 1)), setas='xy')
    with pytest.raises(TypeError, match='does not support where'):
        getattr(data, method)(lambda x, a, b: a * x + b, where=np.ones(6, dtype=bool))


@pytest.mark.parametrize('method', ['apply', 'decompose', 'integrate', 'bin', 'outlier_detection',
                                   'peaks', 'find_peaks', 'section', 'swap_column'])
def test_unsupported_numerical_selections_fail_before_work_or_publication(method):
    """Reject where even when legacy keyword dictionaries would otherwise ignore it."""
    data = Data(np.column_stack((np.arange(7.), np.arange(7.)**2)), setas='xy')
    data.mask[0, 1] = True
    data.metadata['run{String}'] = '001'
    before = data.export_storage()

    def callback(row, **kwargs):
        pytest.fail('Unsupported selection reached the row callback')

    args = (callback,) if method == 'apply' else ((0, 1),) if method == 'swap_column' else ()
    with pytest.raises(TypeError, match='does not support where'):
        getattr(data, method)(*args, where=np.ones(len(data), dtype=bool))
    after = data.export_storage()
    pd.testing.assert_frame_equal(after.values, before.values)
    np.testing.assert_array_equal(after.excluded, before.excluded)
    assert after.schema == before.schema and data.metadata['run'] == '001'
    assert data.metadata.type('run') == 'String'


def test_hysteresis_utility_rejects_unsupported_selection():
    """Reject selection before a utility loads, fits or edits a measurement."""
    from Stoner.analysis.utils import hysteresis_correct

    data = Data(np.column_stack((np.arange(7.), np.arange(7.))), setas='xy')
    before = data.clone
    with pytest.raises(TypeError, match='does not support where'):
        hysteresis_correct(data, where=np.ones(len(data), dtype=bool))
    assert data == before


def test_select_retains_column_query_keyword_namespace():
    """Keep column queries and explicitly forwarded callback names independent."""
    data = Data(np.arange(6).reshape(3, 2), column_headers=['where', 'signal'])
    result = data.select(where=2)
    np.testing.assert_array_equal(result.to_numpy(), [[2, 3]])
    assert data.shape == (3, 2)

    def adjust(row, where):
        return row[1] + where

    data.apply(adjust, col=1, _extra={'where': 3})
    np.testing.assert_array_equal(data.column('adjust'), [4, 6, 8])
