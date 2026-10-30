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
    check_plotting()
    check_image_plotting()
    print(f'Installed-package checks passed: {installed}')


def check_image_plotting():
    """Exercise the image accessor from the installed package without replacing legacy display."""
    from Stoner.enums import CoordinateMode

    image = ImageFile(np.ma.array(np.arange(12.).reshape(3, 4), mask=False))
    image.mask[0, 0] = True
    before = image.export_storage()
    with image.plot.style([]):
        figure = image.plot.new_figure()
        with image.plot.MultiPanel((1, 2), figure=figure) as axes:
            ax = image.plot(ax=axes[0], coordinates=CoordinateMode.PIXEL, mask_color='red')
            assert ax is axes[0] and len(ax.images) == 2
            assert image.plot.hist(ax=axes[1], bins=4) is axes[1]
        image.plot.title = 'Installed image'
        assert image.plot.title() == 'Installed image'
        figure.canvas.draw()
    xr.testing.assert_identical(image.image, before.dataset)
    assert isinstance(image.imshow(), matplotlib.figure.Figure)
    plt.close('all')
    print('Image plotting accessor/property/context checks passed')


def check_plotting():
    """Exercise accessor, contexts, saving and folder targets from each archive."""
    from Stoner.folders import PlotFolder

    data = Data(np.arange(4.), np.arange(4.)**2, np.full(4, 0.1), setas='xye')
    data.mask[1, 1] = True
    before = data.export_storage()
    linewidth = plt.rcParams['lines.linewidth']
    with data.plot.style('stoner, aps', extra={'lines.linewidth': 1.75}):
        if plt.rcParams['lines.linewidth'] != 1.75:
            raise RuntimeError('Style overrides did not take precedence')
    if plt.rcParams['lines.linewidth'] != linewidth:
        raise RuntimeError('Style context did not restore caller settings')
    with data.plot.style('stoner'):
        fig = data.plot.new_figure(figsize=(8, 4))
        with data.plot.MultiPanel((1, 2), figure=fig, adjust_figsize=False) as panels:
            ax = data.plot(ax=panels[0])
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0., 2., 3.])
            data.plot.hist(y=1, ax=panels[1], bins=3)
        with data.plot.InsetPlot(ax=ax, loc='upper left', width=0.3, height=0.3) as inset:
            if data.plot.scatter(ax=inset, title=None, legend=False) is not inset:
                raise RuntimeError('Inset plotting lost its explicit target')
        with data.plot.DoubleYAxis(ax=ax) as right:
            data.plot.line(ax=right, title=None, legend=False)
        for context in (data.plot.DoubleXAxis, data.plot.DoubleXYAxis):
            parent = data.plot.new_figure().subplots()
            data.plot.line(ax=parent, label='Primary')
            with context(ax=parent) as secondary:
                if data.plot.line(label='Secondary') is not secondary:
                    raise RuntimeError('Double-axis plotting lost its current target')
            if plt.gca() is not parent or len(parent.figure.axes) != 2:
                raise RuntimeError('Double-axis context lost its target or added extra axes')
            parent.figure.canvas.draw()
        with tempfile.TemporaryDirectory(prefix='stoner-installed-plot-') as directory:
            target = Path(directory) / 'measurement.png'
            with data.plot.SavedFigure(str(target), formats='png', use=fig):
                pass
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError('SavedFigure did not write the selected figure')

    with data.plot.OffsetPlots(offset=2):
        offset_ax = data.plot.new_figure().subplots()
        data.plot.line(ax=offset_ax)
        data.plot.line(ax=offset_ax)
    np.testing.assert_array_equal(offset_ax.lines[1].get_ydata(), offset_ax.lines[0].get_ydata() + 2)

    after = data.export_storage()
    np.testing.assert_array_equal(after.to_numpy(masked=False), before.to_numpy(masked=False))
    np.testing.assert_array_equal(after.excluded, before.excluded)
    if after.metadata != before.metadata or after.schema != before.schema:
        raise RuntimeError('Drawing changed measurement metadata or column roles')
    if data.clone.plot.figure is not None:
        raise RuntimeError('Cloning retained a live rendering target')

    x, y = np.meshgrid(np.arange(3.), np.arange(3.))
    field = Data(x.ravel(), y.ravel(), (x+y).ravel(), setas='xyz')
    surface = field.plot.new_figure().add_subplot(projection='3d')
    if field.plot.surface(ax=surface) is not surface or not surface.collections:
        raise RuntimeError('Scientific field rendering lost its target or artists')

    folder = PlotFolder(readlist=False)
    for index in range(3):
        folder += Data(np.arange(3.), np.arange(3.)+index, setas='xy')
    axes = folder.plot(plots_per_page=2)
    if len(axes) != 3 or len(folder.figures) != 2:
        raise RuntimeError('Folder pagination lost members or created extra pages')
    overlay = plt.figure().subplots()
    if folder.each.plot.line(ax=overlay) != [overlay]*3:
        raise RuntimeError('Bulk plotting lost folder order or its explicit target')
    if any('func' in member.metadata for member in folder):
        raise RuntimeError('Bulk plotting added diagnostic metadata')
    plt.close('all')
    print(f'Plotting accessor/context/folder checks passed (stonerplots {version("stonerplots")})')


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
