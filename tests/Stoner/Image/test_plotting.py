"""Verify image plotting geometry, ownership and the shared Data plotting contract."""

import pickle
from copy import copy, deepcopy

import numpy as np
import pytest
import stonerplots
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from Stoner import Data, ImageFile, ImageStack
from Stoner.enums import CoordinateMode, FitOutput
from Stoner.Image import ImagePlotter
from Stoner.Image.storage import ImageStorage


@pytest.fixture(autouse=True)
def figures():
    """Isolate each test's Figure lifecycle."""
    plt.close('all')
    yield
    plt.close('all')


def calibrated(x=(10, 12, 14, 16), y=(30, 33, 36)):
    """Build an image with unequal calibrated spatial steps and a masked outlier."""
    values = np.ma.array(np.arange(len(x)*len(y), dtype=float).reshape(len(y), len(x)))
    values.mask = np.zeros(values.shape, dtype=bool)
    values.mask[0, 0] = True
    values.data[0, 0] = 999
    package = ImageStorage.from_numpy(values)
    package.dataset = package.dataset.assign_coords(x=list(x), y=list(y))
    package.dataset.x.attrs = {'units': 'um', 'long_name': 'Horizontal'}
    package.dataset.y.attrs = {'units': 'um', 'long_name': 'Vertical'}
    return ImageFile.from_storage(package)


def test_lookup_is_inert(monkeypatch):
    image = calibrated()

    def forbidden(*args, **kwargs):
        raise AssertionError('Lookup read scientific values')

    monkeypatch.setattr('Stoner.Image.plotting.selected_image_values', forbidden)
    assert isinstance(image.plot, ImagePlotter)
    assert image.plot.ax is None and image.plot.figure is None and image.plot.axes == []
    assert 'image' in dir(image.plot) and 'hist' in dir(image.plot)
    with pytest.raises(AttributeError):
        _ = image.plot.unknown_property
    assert not plt.get_fignums()
    assert '_plot_state' not in image.__dict__


@pytest.mark.parametrize('method', ['__call__', 'image', 'imshow', 'pcolormesh', 'contour', 'hist'])
def test_methods_return_explicit_axes_without_spare_figures(method):
    image = calibrated()
    fig, ax = plt.subplots()
    before = plt.get_fignums()
    assert getattr(image.plot, method)(ax=ax) is ax
    assert plt.get_fignums() == before
    assert image.plot.ax is ax and image.plot.figure is fig
    assert image.plot.axes == [ax]
    assert not {'ax', 'fig'}.intersection(image.metadata)


def test_uniform_native_grid_raw_units_and_owner_unchanged():
    image = calibrated()
    before = image.export_storage()
    ax = image.plot()
    artist = ax.images[0]
    assert artist.origin == 'upper'
    np.testing.assert_allclose(artist.get_extent(), [9, 17, 37.5, 28.5])
    np.testing.assert_array_equal(artist.get_array().mask, before.dataset.excluded.values)
    assert artist.get_clim() == (1, 11)
    assert ax.get_xlabel() == 'Horizontal / um'
    assert ax.get_ylabel() == 'Vertical / um'
    xr.testing.assert_identical(image.image, before.dataset)
    assert dict(image.metadata) == dict(before.metadata)


@pytest.mark.parametrize('method', ['image', 'pcolormesh', 'contour'])
@pytest.mark.parametrize('origin', ['upper', 'lower'])
def test_decreasing_axes_align_across_renderers(method, origin):
    image = calibrated(x=(16, 14, 12, 10), y=(36, 33, 30))
    ax = getattr(image.plot, method)(origin=origin)
    np.testing.assert_allclose(ax.get_xlim(), [17, 9])
    np.testing.assert_allclose(ax.get_ylim(), [28.5, 37.5] if origin == 'upper' else [37.5, 28.5])
    if method == 'image':
        assert ax.images[0].get_array()[2, 3] == 11
    elif method == 'pcolormesh':
        mesh = ax.collections[0]
        np.testing.assert_allclose(mesh.get_coordinates()[0, 0], [17, 37.5])
        assert mesh.get_array()[2, 3] == 11


def test_nonuniform_image_uses_mesh_and_explicit_imshow_rejects_before_figure():
    image = calibrated(x=(0, 1, 3, 8))
    with pytest.raises(ValueError, match='uniform'):
        image.plot.imshow()
    assert not plt.get_fignums()
    ax = image.plot()
    assert not ax.images
    np.testing.assert_allclose(ax.collections[0].get_coordinates()[0, :, 0], [-.5, .5, 2, 5.5, 10.5])
    with pytest.raises(ValueError, match='extent'):
        image.plot(extent=(0, 4, 0, 3))


@pytest.mark.parametrize('mode', [CoordinateMode.PIXEL, 'pixel'])
def test_explicit_pixel_mode_and_display_extent(mode):
    image = calibrated()
    ax = image.plot(coordinates=mode)
    np.testing.assert_allclose(ax.images[0].get_extent(), [-.5, 3.5, 2.5, -.5])
    assert ax.get_xlabel() == 'x / pixel'
    other = image.plot(extent=(100, 200, -30, 40))
    np.testing.assert_allclose(other.images[-1].get_extent(), [100, 200, -30, 40])
    np.testing.assert_array_equal(image.image.x, [10, 12, 14, 16])


@pytest.mark.parametrize('mode', [FitOutput.DATA, 1, 'bad'])
def test_invalid_coordinate_modes_fail_without_target(mode):
    image = calibrated()
    with pytest.raises((TypeError, ValueError)):
        image.plot(coordinates=mode)
    assert not plt.get_fignums()


def test_coordinate_names_maps_and_unordered_axes():
    image = calibrated()
    package = image.export_storage()
    package.dataset.coords['alternate'] = ('x', [100, 110, 120, 130])
    package.dataset.coords['map_x'] = (('y', 'x'), np.ones(image.shape))
    image = ImageFile.from_storage(package)
    assert image.plot(x='alternate').images[0].get_extent()[:2] == [95, 135]
    for kwargs in ({'x': 'map_x'}, {'x': 'missing'}, {'x': 0}, {'coordinates': 'pixel', 'x': 'x'}):
        with pytest.raises(ValueError):
            image.plot(**kwargs)
    package = image.export_storage()
    package.dataset.coords['x'] = [0, 1, 1, 2]
    image = ImageFile.from_storage(package)
    with pytest.raises(ValueError, match='monotonic'):
        image.plot()
    assert image.plot(coordinates='pixel').images


@pytest.mark.parametrize('shape', [(1, 1), (1, 4), (4, 1)])
def test_singleton_axes_have_unit_width(shape):
    image = ImageFile(np.ones(shape))
    ax = image.plot()
    np.testing.assert_allclose(ax.images[0].get_extent(), [-.5, shape[1]-.5, shape[0]-.5, -.5])
    with pytest.raises(ValueError, match='two rows'):
        image.plot.contour()


def test_selection_alignment_and_mask_overlay_same_geometry():
    image = calibrated()
    selected = np.ones(image.shape, dtype=bool)
    selected[2] = False
    where = xr.DataArray(selected, dims=('y', 'x'), coords={'y': image.image.y, 'x': image.image.x})
    norm = Normalize(0, 12)
    ax = image.plot(where=where, mask_color='red', mask_alpha=.25, norm=norm)
    assert ax.images[0].norm is norm
    assert ax.images[0].get_array().mask[2].all()
    assert ax.images[0].get_extent() == ax.images[1].get_extent()
    assert ax.images[0].origin == ax.images[1].origin
    assert image.mask[:].sum() == 1
    with pytest.raises(ValueError, match='coordinates'):
        image.plot(where=where.assign_coords(x=[0, 1, 2, 3]))
    with pytest.raises(ValueError, match='Boolean'):
        image.plot(where=np.ones(image.shape))


def test_mesh_overlay_and_finite_colour_limits():
    image = calibrated(x=(0, 1, 3, 8))
    image[1, 1] = np.nan
    ax = image.plot(mask_color='blue')
    assert len(ax.collections) == 2
    np.testing.assert_array_equal(ax.collections[0].get_coordinates(), ax.collections[1].get_coordinates())
    assert ax.collections[0].get_clim() == (1, 11)


def test_overlay_follows_display_transform_and_zorder():
    from matplotlib.transforms import Affine2D

    image = calibrated()
    _, ax = plt.subplots()
    transform = Affine2D().translate(3, 4) + ax.transData
    image.plot(ax=ax, transform=transform, zorder=5, mask_color='red')
    assert ax.images[0].get_transform() is ax.images[1].get_transform()
    assert ax.images[1].get_zorder() > ax.images[0].get_zorder()


@pytest.mark.parametrize('all_masked', [False, True])
def test_constant_and_empty_image_display(all_masked):
    image = ImageFile(np.ones((3, 4)))
    if all_masked:
        image.mask = True
    ax = image.plot()
    ax.figure.canvas.draw()
    if all_masked:
        assert ax.images[0].get_array().mask.all()
        with pytest.raises(ValueError, match='No finite'):
            image.plot.hist()


def test_histogram_matches_raw_eligible_samples_and_weights():
    image = calibrated()
    where = np.ones(image.shape, dtype=bool)
    where[2, :] = False
    weights = np.arange(12).reshape(image.shape)/10
    expected, _ = np.histogram(np.arange(1, 8), bins=[0, 4, 8, 12], weights=weights.ravel()[1:8])
    ax = image.plot.hist(where=where, bins=[0, 4, 8, 12], weights=weights)
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], expected)
    assert ax.get_xlabel() == 'Intensity' and ax.get_ylabel() == 'Count'
    labelled = xr.DataArray(weights, dims=('y', 'x'), coords={'y': image.image.y, 'x': image.image.x})
    image.plot.hist(weights=labelled, density=True)
    assert image.plot.ax.get_ylabel() == 'Density'
    with pytest.raises(ValueError, match='coordinates'):
        image.plot.hist(weights=labelled.assign_coords(x=[0, 1, 2, 3]))
    weights[1, 1] = np.inf
    with pytest.raises(ValueError, match='finite'):
        image.plot.hist(weights=weights)
    image.plot.hist(orientation='horizontal', density=True)
    assert image.plot.ax.get_xlabel() == 'Density' and image.plot.ax.get_ylabel() == 'Intensity'


def test_shared_properties_contexts_and_closed_targets():
    image = calibrated()
    ax = image.plot()
    image.plot.xlabel = 'Position'
    saved = image.plot.xlabel
    assert saved() == 'Position'
    image.plot.title('Corner', loc='left')
    assert image.plot.title.get(loc='left') == 'Corner'
    image.plot.dpi = 120
    assert ax.figure.get_dpi() == 120
    assert image.plot.MultiPanel is stonerplots.MultiPanel
    assert image.plot.SavedFigure is Data.plot.fget(Data()).SavedFigure
    with image.plot.style([], extra={'lines.linewidth': 3}):
        assert plt.rcParams['lines.linewidth'] == 3
    other = calibrated()
    other.plot.new_figure()
    assert image.plot.ax is ax
    plt.close(ax.figure)
    assert image.plot.ax is None and image.plot.figure is None
    with pytest.raises(AttributeError):
        saved()
    with pytest.raises(ValueError, match='live'):
        image.plot(ax=ax)


def test_plot_state_does_not_leak_into_copies_pickle_or_storage():
    image = calibrated()
    image.plot()
    before = plt.get_fignums()
    for cloned in (image.clone, deepcopy(image), pickle.loads(pickle.dumps(image))):
        assert cloned.plot.figure is None and cloned.plot.ax is None
        xr.testing.assert_identical(cloned.image, image.image)
    # Shallow copy follows the existing owner semantics, but must not create figures.
    copy(image)
    assert plt.get_fignums() == before
    assert not {'fig', 'ax', '_plot_state'}.intersection(image.export_storage().metadata)


def test_crop_keeps_native_offsets_and_legacy_display_return():
    image = calibrated()
    cropped = image[1:, 1:]
    ax = cropped.plot()
    np.testing.assert_allclose(ax.images[0].get_extent(), [11, 17, 37.5, 31.5])
    assert isinstance(image.imshow(), Figure)
    assert isinstance(image.hist(), Data)
    assert isinstance(ax, Axes)


def test_ragged_stack_frame_renders_only_its_valid_extent():
    stack = ImageStack([np.ones((2, 3)), np.ones((4, 5))])
    frame = stack[0]
    ax = frame.plot()
    assert ax.images[0].get_array().shape == (2, 3)
    np.testing.assert_allclose(ax.images[0].get_extent(), [-.5, 2.5, 1.5, -.5])
    assert frame.plot.hist().patches
    assert stack[1].plot.figure is None


def test_region_and_saved_accessor_follow_live_values():
    image = calibrated()
    region = image.crop(1, 4, 1, 3, _=None)
    saved = region.plot
    image[1, 1] = 42
    ax = saved()
    assert ax.images[0].get_array()[0, 0] == 42
    np.testing.assert_allclose(ax.images[0].get_extent(), [11, 17, 37.5, 31.5])


def test_contexts_restore_current_axes_on_exception():
    image = calibrated()
    parent = image.plot()
    with pytest.raises(RuntimeError), image.plot.InsetPlot(ax=parent) as inset:
        assert image.plot(ax=inset) is inset
        raise RuntimeError('test exception')
    assert plt.gca() is parent
    figure = image.plot.new_figure()
    with image.plot.MultiPanel((1, 2), figure=figure) as axes:
        assert image.plot(ax=axes[0]) is axes[0]
        assert image.plot.hist(ax=axes[1]) is axes[1]


def test_coordinate_precision_and_invalid_opacity_fail_without_figure():
    image = calibrated()
    for opacity in ('bad', None, np.nan, -1, 2):
        with pytest.raises(ValueError, match='mask_alpha'):
            image.plot(mask_alpha=opacity)
    package = image.export_storage()
    package.dataset.coords['x'] = np.arange(4, dtype=np.int64) + 2**54
    image = ImageFile.from_storage(package)
    with pytest.raises(ValueError, match='precision'):
        image.plot()
    assert not plt.get_fignums()


@pytest.mark.parametrize('kwargs', [{'xcol': 0}, {'figure': 'new'}, {'kind': 'image'}, {'origin': 'sideways'}])
def test_invalid_options_fail_before_figure(kwargs):
    image = calibrated()
    with pytest.raises((ValueError, TypeError)):
        image.plot(**kwargs)
    assert not plt.get_fignums()
