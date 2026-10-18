"""Smoke-test an installed distribution while using checkout files only as fixtures."""

from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from Stoner import Data, ImageFile, ImageStack
import Stoner


def main():
    """Reject source imports, then exercise numerical, loader, plot and image paths."""
    checkout = Path(sys.argv[1]).resolve()
    installed = Path(Stoner.__file__).resolve()
    if installed.is_relative_to(checkout):
        raise RuntimeError(f'Imported checkout instead of installed distribution: {installed}')

    if version('Stoner') != Stoner.__version__:
        raise RuntimeError('Installed metadata and runtime versions differ')
    check_data_and_resources(checkout)
    check_reviewed_wrappers()
    print(f'Installed-package checks passed: {installed}')


def check_reviewed_wrappers():
    """Exercise the reviewed wrapper contracts from each installed archive."""
    data = Data(np.array([[2., 20.], [1., 10.], [1., 11.], [np.nan, 30.]]), setas='xy')
    data.mask[1, 0] = True
    data.metadata['run{String}'] = '001'
    data.metadata['parent'] = data
    for method, rows in ((data.head, slice(0, 2)), (data.tail, slice(2, 4))):
        preview = method(2)
        np.testing.assert_array_equal(preview.to_numpy().data, data.to_numpy().data[rows])
        np.testing.assert_array_equal(preview.mask, np.asarray(data.mask)[rows])
        assert preview.column_ids == data.column_ids and preview.setas == 'xy'
        assert preview.metadata['run'] == '001' and preview.metadata.type('run') == 'String'
        assert preview.metadata['parent'] is preview
        preview[0, 1] = -99.
        assert not (data.data.iloc[:, 1] == -99.).any()
    original = data.to_numpy()
    assert data.sort(0, reverse=True) is data
    positions = [3, 0, 2, 1]
    np.testing.assert_array_equal(data.to_numpy().data, original.data[positions])
    np.testing.assert_array_equal(data.mask, original.mask[positions])
    # Finite exclusions retain NumPy's masked group and optional result arrays.
    categories = Data(np.array([[3], [1], [3], [2]]))
    categories.mask[[1, 3], 0] = True
    values, indices, inverse = categories.unique(0, True, True)
    np.testing.assert_array_equal(values.compressed(), [3])
    np.testing.assert_array_equal(values.mask, [False, True])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(inverse, [0, 1, 0, 1])
    print('Reviewed head/tail, sorting and masked unique checks passed')


def check_data_and_resources(checkout):
    """Exercise package APIs and resources; callable for a source-level preflight too."""
    data = Data(np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]), setas='xy')
    if data.shape != (3, 2):
        raise RuntimeError(f'Unexpected constructed data shape: {data.shape}')
    frame = data.data
    if type(frame) is not pd.DataFrame or frame is not data.data:
        raise RuntimeError('Data.data does not expose the stored ordinary DataFrame')
    data.mask[1, 1] = True
    data.data.iloc[1, 1] = 17.0
    if data.to_numpy().data[1, 1] != 17.0 or not data.mask[1, 1]:
        raise RuntimeError('Native table edits lost raw values or persistent exclusions')
    loaded = Data(checkout / 'sample-data' / 'TDI_Format_RT.txt')
    if loaded.shape[0] <= 0 or loaded.shape[1] < 2:
        raise RuntimeError(f'Unexpected sample data shape: {loaded.shape}')

    with data.plot.style('stoner'):
        data.plot()
    plt.close('all')

    resources = files('Stoner.Image').joinpath('tessdata')
    for name in ('eng.traineddata', 'equ.traineddata', 'kerr-patterns.txt'):
        with resources.joinpath(name).open('rb') as stream:
            if not stream.read(1):
                raise RuntimeError(f'Empty OCR resource: {name}')

    image = ImageFile(np.arange(100, dtype=float).reshape(10, 10))
    dataset = image.image
    if type(dataset) is not xr.Dataset or dataset is not image.image:
        raise RuntimeError('ImageFile.image does not expose the stored Dataset')
    stack = ImageStack([image])
    if type(stack.stack) is not xr.Dataset or stack.stack is not stack.stack:
        raise RuntimeError('ImageStack.stack does not expose the stored Dataset')
    stack[0][0, 0] = 7.0
    if stack.stack.intensity.values[0, 0, 0] != 7.0:
        raise RuntimeError('Frame edits did not reach native stack storage')
    with tempfile.TemporaryDirectory(prefix='stoner-installed-smoke-') as directory:
        target = Path(directory) / 'roundtrip.tiff'
        image.save(target)
        restored = ImageFile(target)
        if not np.allclose(restored.to_numpy(), image.to_numpy()):
            raise RuntimeError('Image pixels changed during the TIFF round trip')


if __name__ == '__main__':
    main()
