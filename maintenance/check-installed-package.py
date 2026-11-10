"""Smoke-test an installed distribution while using checkout files only as fixtures."""

from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')
# The headless backend must be selected before pyplot or Stoner imports.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from Stoner import Data, ImageFile, ImageStack, ROI  # noqa: E402
import Stoner  # noqa: E402
from Stoner.enums import CoordinateMode, FitOutput  # noqa: E402
from Stoner.folders import PlotFolder  # noqa: E402
# pylint: enable=wrong-import-position


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
    check_image_roi()
    check_image_fitting()
    print(f'Installed-package checks passed: {installed}')


def check_image_plotting():
    """Exercise the image accessor from the installed package without replacing legacy display."""

    image = ImageFile(np.ma.array(np.arange(12.).reshape(3, 4), mask=False))
    image.mask[0, 0] = True
    before = image.export_storage()
    with image.plot.style([]):
        figure = image.plot.new_figure()
        with image.plot.MultiPanel((1, 2), figure=figure) as axes:
            ax = image.plot(ax=axes[0], coordinates=CoordinateMode.PIXEL, mask_color='red')
            if not (ax is axes[0] and len(ax.images) == 2):
                raise RuntimeError('Installed-package check failed: ax is axes[0] and len(ax.images) == 2')
            if not (image.plot.hist(ax=axes[1], bins=4) is axes[1]):
                raise RuntimeError('Installed-package check failed: image.plot.hist(ax=axes[1], bins=4) is axes[1]')
        image.plot.title = 'Installed image'
        # Assignment forwards to Matplotlib; reading still returns a callable property proxy.
        # pylint: disable-next=not-callable,comparison-with-callable
        if not (image.plot.title() == 'Installed image'):
            raise RuntimeError("Installed-package check failed: image.plot.title() == 'Installed image'")
        figure.canvas.draw()
    xr.testing.assert_identical(image.image, before.dataset)
    if not (isinstance(image.imshow(), matplotlib.figure.Figure)):
        raise RuntimeError('Installed-package check failed: isinstance(image.imshow(), matplotlib.figure.Figure)')
    plt.close('all')
    print('Image plotting accessor/property/context checks passed')


def check_image_fitting():
    """Exercise installed surface models, physical covariance and captured corrections."""

    x, y = np.meshgrid(np.arange(9.), np.arange(7.))
    background = 3 + .4 * x - .2 * y + .03 * x * x + .02 * x * y - .01 * y * y
    values = background.copy()
    values[2:5, 3:6] += 10
    image = ImageFile(values)
    image.roi.set(ROI.all() - ROI.box(3, 6, 2, 5))
    before = image.to_numpy()
    report = image.fit.quadratic_surface(output=FitOutput.REPORT)
    np.testing.assert_allclose(image.fit.evaluate(report).to_numpy(), background, atol=1e-9)
    corrected = image.fit.flatten(report)
    np.testing.assert_allclose(corrected.to_numpy(), values - background, atol=1e-9)
    if not (corrected.dtype == np.float64):
        raise RuntimeError('Installed-package check failed: corrected.dtype == np.float64')
    if not (np.asarray(report.image_fit.to_dict()['covariance']).shape == (6, 6)):
        raise RuntimeError("Surface covariance does not have six physical coefficients")
    np.testing.assert_array_equal(image.to_numpy(), before)
    print('Image fitting surface/covariance/replay/correction checks passed')


def check_image_roi():
    """Exercise installed ROI construction, replay, reduction and outline integration."""

    image = ImageFile(np.arange(20.).reshape(4, 5))
    image.mask[1, 1] = True
    before = image.export_storage()
    image.roi.box(1, 4, 1, 3)
    if not (image.roi.active and not image.roi.stale):
        raise RuntimeError('Installed-package check failed: image.roi.active and not image.roi.stale')
    if not (int(image.roi.resolve().sum()) == 6):
        raise RuntimeError('Installed-package check failed: int(image.roi.resolve().sum()) == 6')
    if not (image.mean() == np.mean([7., 8., 11., 12., 13.])):
        raise RuntimeError('Installed-package check failed: image.mean() == np.mean([7., 8., 11., 12., 13.])')
    region = ROI.from_dict(image.roi.current.to_dict())
    image.roi.set(region - ROI.circle((2, 2), 0.5))
    if not (int(image.roi.resolve().sum()) == 5):
        raise RuntimeError('Installed-package check failed: int(image.roi.resolve().sum()) == 5')
    ax = image.plot()
    if not (image.plot.roi(ax=ax) is ax):
        raise RuntimeError('Installed-package check failed: image.plot.roi(ax=ax) is ax')
    image.plot.hist(bins=3)
    if not (image.mean(where=ROI.all()) == np.mean(np.delete(np.arange(20.), 6))):
        raise RuntimeError('Explicit all-pixel selection did not bypass the current ROI')
    image.roi.clear()
    if not (not image.roi.active):
        raise RuntimeError('Installed-package check failed: not image.roi.active')
    xr.testing.assert_identical(image.image, before.dataset)
    plt.close('all')
    print('Image ROI geometry/replay/reduction/plot checks passed')


def check_plotting():
    """Exercise accessor, contexts, saving and folder targets from each archive."""

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

    _check_field_and_folder_plotting()
    plt.close('all')
    print(f'Plotting accessor/context/folder checks passed (stonerplots {version("stonerplots")})')


def _check_field_and_folder_plotting():
    """Check 3D rendering, folder pagination and explicit bulk overlays."""
    x, y = np.meshgrid(np.arange(3.), np.arange(3.))
    field = Data(x.ravel(), y.ravel(), (x + y).ravel(), setas='xyz')
    surface = field.plot.new_figure().add_subplot(projection='3d')
    if field.plot.surface(ax=surface) is not surface or not surface.collections:
        raise RuntimeError('Scientific field rendering lost its target or artists')

    folder = PlotFolder(readlist=False)
    for index in range(3):
        folder += Data(np.arange(3.), np.arange(3.) + index, setas='xy')
    axes = folder.plot(plots_per_page=2)
    if len(axes) != 3 or len(folder.figures) != 2:
        raise RuntimeError('Folder pagination lost members or created extra pages')
    overlay = plt.figure().subplots()
    if folder.each.plot.line(ax=overlay) != [overlay] * 3:
        raise RuntimeError('Bulk plotting lost folder order or its explicit target')
    if any('func' in member.metadata for member in folder):
        raise RuntimeError('Bulk plotting added diagnostic metadata')


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
        if not (preview.column_ids == data.column_ids and preview.setas == 'xy'):
            raise RuntimeError("Preview changed column identity or scientific roles")
        if not (preview.metadata['run'] == '001' and preview.metadata.type('run') == 'String'):
            raise RuntimeError("Preview did not preserve explicitly typed metadata")
        if not (preview.metadata['parent'] is preview):
            raise RuntimeError("Installed-package check failed: preview.metadata['parent'] is preview")
        preview[0, 1] = -99.
        if not (not (data.data.iloc[:, 1] == -99.).any()):
            raise RuntimeError('Installed-package check failed: not (data.data.iloc[:, 1] == -99.).any()')
    original = data.to_numpy()
    if not (data.sort(0, reverse=True) is data):
        raise RuntimeError('Installed-package check failed: data.sort(0, reverse=True) is data')
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
    # The native boundary promises the ordinary concrete backend type, excluding subclasses.
    # pylint: disable-next=unidiomatic-typecheck
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
    # The native boundary promises the ordinary concrete backend type, excluding subclasses.
    # pylint: disable-next=unidiomatic-typecheck
    if type(dataset) is not xr.Dataset or dataset is not image.image:
        raise RuntimeError('ImageFile.image does not expose the stored Dataset')
    stack = ImageStack([image])
    # The native boundary promises the ordinary concrete backend type, excluding subclasses.
    # pylint: disable-next=unidiomatic-typecheck
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
