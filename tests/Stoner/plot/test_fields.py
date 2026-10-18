"""Verify field coordinates, exclusions, colour mapping and Axes contracts."""

import builtins
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import colormaps, colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from Stoner import Data


@pytest.fixture(autouse=True)
def isolated_figures():
    """Close owned figures and restore styles even after failed assertions."""
    plt.close("all")
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def scalar():
    """Return an unordered non-square grid with a known scalar plane."""
    x, y = np.meshgrid([0., 1., 2.], [0., 10.])
    order = [4, 0, 5, 2, 1, 3]
    return Data(x.ravel()[order], y.ravel()[order], (3*x+2*y).ravel()[order],
                np.full(6, 99.), setas="xyz.", column_headers=["X", "Y", "Value", "Unused"])


@pytest.fixture
def vectors():
    """Return cardinal vectors on a two-by-two field."""
    return Data(np.array([[0., 0., 1., 0.], [1., 0., 0., 1.], [0., 1., -1., 0.], [1., 1., 0., -1.]]),
                setas="xyuv", column_headers=["X", "Y", "U", "V"])


def spy_surface(monkeypatch):
    """Capture the actual surface coordinates without replacing native drawing."""
    calls = []
    original = Axes3D.plot_surface

    def capture(ax, x, y, z, **kwargs):
        calls.append((np.array(x), np.array(y), np.array(z)))
        return original(ax, x, y, z, **kwargs)

    monkeypatch.setattr(Axes3D, "plot_surface", capture)
    return calls


def test_surface_orientation_and_automatic_dispatch(scalar, monkeypatch):
    calls = spy_surface(monkeypatch)
    ax = scalar.plot.surface(colorbar=False)
    assert ax.name == "3d" and scalar.plot.ax is ax
    np.testing.assert_equal(calls[0][0], [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_equal(calls[0][1], [[0, 0, 0], [10, 10, 10]])
    np.testing.assert_equal(calls[0][2], [[0, 3, 6], [20, 23, 26]])
    scalar.plot(ax=ax, colorbar=False)
    scalar.plot(kind="surface", ax=ax, colorbar=False)
    for call in calls[1:]:
        np.testing.assert_equal(call, calls[0])


def test_explicit_surface_on_unassigned_table_does_not_change_three_column_auto_guess():
    data = Data(np.array([[0., 0., 1.], [1., 0., 2.], [0., 1., 3.], [1., 1., 4.]]))
    ax = data.plot()
    assert ax.name == "rectilinear" and ax.containers
    ax = data.plot.surface(colorbar=False)
    assert ax.name == "3d" and len(plt.get_fignums()) == 2


@pytest.mark.parametrize("excluded_column", [0, 1, 2])
def test_scalar_masks_leave_holes_and_do_not_use_unrelated_exclusions(scalar, excluded_column):
    scalar.mask[0, excluded_column] = True  # X=1, Y=10.
    scalar.mask[1, 3] = True  # An unrelated column must not affect the value at 0,0.
    ax = scalar.plot.image(colorbar=False)
    image = ax.images[0]
    assert image.get_array().shape == (2, 3)
    np.testing.assert_equal(np.ma.getmaskarray(image.get_array()), [[False]*3, [False, True, False]])
    assert image.get_array()[0, 0] == 0
    assert image.get_extent() == [-0.5, 2.5, -5, 15]
    assert image.origin == "lower"


def test_existing_numerical_griddata_remains_independent_of_rendering(scalar):
    x, y, value = scalar.griddata(xcol=0, ycol=1, zcol=2)
    np.testing.assert_allclose(value, 3*x+2*y)
    assert plt.get_fignums() == [] and scalar.plot.figure is None


def test_image_retains_physical_scalar_normalisation_and_colourbar(scalar):
    ax = scalar.plot.image(vmin=-10, vmax=30, cmap="plasma")
    image = ax.images[0]
    assert image.get_array().max() == 26
    assert image.norm.vmin == -10 and image.norm.vmax == 30
    np.testing.assert_allclose(image.to_rgba(10), colormaps["plasma"](0.5))
    assert image.colorbar.mappable is image
    assert image.colorbar.ax.get_ylabel() == "Value"
    assert scalar.plot.ax is ax and len(scalar.plot.axes) == 2


def test_colourbar_does_not_steal_current_or_explicit_axes(scalar):
    _, previous = plt.subplots()
    target = Figure().add_subplot(projection="3d")
    assert scalar.plot.surface(ax=target) is target
    assert plt.gca() is previous and len(plt.get_fignums()) == 1
    ax = scalar.plot.surface()
    assert plt.gca() is ax


def test_explicit_vector_entry_point_uses_matplotlib():
    data = Data(np.arange(12, dtype=float).reshape(2, 6), setas="xyzuvw")
    ax = data.plot.vector3d(color="blue")
    assert isinstance(ax, Axes3D)


def test_scattered_linear_field_and_explicit_non_square_shape():
    data = Data(np.array([[0., 0., 0.], [2., 0., 2.], [0., 2., 4.], [2., 2., 6.], [1., 0.5, 2.]]), setas="xyz")
    ax = data.plot.image(shape=(5, 3), colorbar=False)
    expected_x, expected_y = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 2, 3))
    np.testing.assert_allclose(ax.images[0].get_array(), expected_x+2*expected_y)


@pytest.mark.parametrize("kind", ["contour", "pcolormesh", "image"])
def test_scalar_methods_share_the_2d_target(scalar, kind):
    figure, target = plt.subplots()
    assert getattr(scalar.plot, kind)(ax=target, colorbar=False) is target
    assert scalar.plot(kind=kind, ax=target, colorbar=False) is target
    assert len(figure.axes) == 1 and len(plt.get_fignums()) == 1


def test_outside_interpolation_hull_is_missing_not_zero():
    data = Data(np.array([[0., 0., 10.], [1., 0., 11.], [0., 1., 12.]]), setas="xyz")
    image = data.plot.image(shape=(3, 3), colorbar=False).images[0]
    assert np.ma.is_masked(image.get_array()[2, 2])
    assert image.get_array().min() >= 10


def test_vector_colours_and_overlay_are_aligned(vectors):
    ax = vectors.plot.vector()
    expected = np.array([[[127, 0, 255], [0, 255, 255]], [[127, 255, 0], [255, 0, 0]]])/255
    np.testing.assert_allclose(ax.images[0].get_array(), expected)
    arrows = ax.collections[0]
    np.testing.assert_equal(arrows.U, [1, 0, -1, 0])
    np.testing.assert_equal(arrows.V, [0, 1, 0, -1])
    assert arrows.angles == "xy" and arrows.scale == 1
    assert len(ax.figure.axes) == 1


def test_vector_masks_apply_to_image_and_arrows(vectors):
    vectors.mask[2, 2] = True
    ax = vectors.plot.vector()
    assert ax.collections[0].U.size == 3
    assert np.ma.getmaskarray(ax.images[0].get_array())[1, 0].all()


def test_vector_retains_one_valid_pixel_without_interpolating_exclusions(vectors):
    vectors.mask[1:, 2] = True
    ax = vectors.plot.vector()
    assert ax.collections[0].U.size == 1
    assert np.ma.getmaskarray(ax.images[0].get_array()).all(axis=-1).sum() == 3


def test_vector_duplicates_use_the_same_first_valid_sample_for_colours_and_arrows(vectors):
    vectors += np.array([[0., 0., 3., 0.]])
    ax = vectors.plot.vector()
    assert ax.collections[0].U.size == 4
    np.testing.assert_equal(ax.collections[0].U, [1, 0, -1, 0])
    np.testing.assert_allclose(ax.images[0].get_array()[0, 0], [127/255, 0, 1])


@pytest.mark.parametrize("zero", [False, True])
def test_constant_third_component_and_zero_field_have_finite_colours(vectors, zero):
    vectors &= np.zeros(4)
    vectors.setas = "xyuvw"
    if zero:
        vectors[:, 2:4] = 0
    image = vectors.plot.vector(arrows=False).images[0].get_array()
    assert np.isfinite(image).all()
    if zero:
        np.testing.assert_allclose(image, np.full((2, 2, 3), 127/255))


def test_quiver_ignores_unused_z_w_masks_and_preserves_supplied_colour(vectors):
    vectors &= np.zeros(4)
    vectors.setas = "xyuvw"
    vectors.mask[:, 4] = True
    ax = vectors.plot.quiver(color="red")
    assert ax.collections[0].U.size == 4
    np.testing.assert_allclose(ax.collections[0].get_facecolors()[0], colors.to_rgba("red"))


def test_vector3d_is_matplotlib_and_applies_joint_masks(monkeypatch):
    data = Data(np.arange(18, dtype=float).reshape(3, 6), setas="xyzuvw")
    data.mask[1, 5] = True
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("mayavi"):
            pytest.fail("Implicit Mayavi import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ax = data.plot(color="red")
    assert isinstance(ax, Axes3D)
    assert len(ax.collections[0]._segments3d) == 6  # Shaft plus two head lines per valid sample.
    assert len(plt.get_fignums()) == 1


def test_scatter3d_filters_per_sample_colours_and_sizes():
    data = Data(np.arange(12, dtype=float).reshape(4, 3), setas="xyz")
    data.mask[1, 2] = True
    ax = data.plot.scatter3d(c=np.array([10., 20., 30., 40.]), s=np.array([1., 2., 3., 4.]))
    np.testing.assert_equal(ax.collections[0].get_array(), [10, 30, 40])
    np.testing.assert_equal(ax.collections[0].get_sizes(), [1, 3, 4])


def test_voxel_corners_visibility_masks_and_normalisation(monkeypatch):
    x, y, z = np.meshgrid([10., 12.], [20., 24.], [0., 1.], indexing="ij")
    data = Data(x.ravel(), y.ravel(), z.ravel(), (x+y+z).ravel(), setas="xyzu")
    data.mask[0, 3] = True
    calls = []
    original = Axes3D.voxels

    def capture(ax, x, y, z, filled, **kwargs):
        calls.append((x, y, z, filled.copy(), kwargs))
        return original(ax, x, y, z, filled, **kwargs)

    monkeypatch.setattr(Axes3D, "voxels", capture)
    ax = data.plot.voxels(visible=lambda x, y, z: x < 12, vmin=0, vmax=50, cmap="plasma")
    corners_x, corners_y, corners_z, occupied, options = calls[0]
    np.testing.assert_equal(corners_x[:, 0, 0], [9, 11, 13])
    np.testing.assert_equal(corners_y[0, :, 0], [18, 22, 26])
    np.testing.assert_equal(corners_z[0, 0, :], [-0.5, 0.5, 1.5])
    assert occupied.shape == (2, 2, 2) and occupied.sum() == 3 and not occupied[0, 0, 0]
    expected = list(colormaps["plasma"](31/50))
    expected[3] *= 0.5
    np.testing.assert_allclose(options["facecolors"][0, 0, 1], expected)
    assert len(ax.figure.axes) == 2


def test_voxel_visibility_shape_is_checked_before_axes():
    x, y, z = np.meshgrid([0., 1.], [0., 1.], [0., 1.], indexing="ij")
    data = Data(x.ravel(), y.ravel(), z.ravel(), np.ones(8), setas="xyzu")
    with pytest.raises(ValueError, match="Boolean"):
        data.plot.voxels(filled=np.ones((2, 2, 2)))
    assert plt.get_fignums() == []


def test_matrix_orientation_non_square_crop_and_masks(monkeypatch):
    data = Data(np.array([[0., 10., 20.], [2., 210., 220.], [4., 410., 420.]]), column_headers=["X", "10", "20"])
    data.mask[1, 2] = True
    calls = spy_surface(monkeypatch)
    data.plot.matrix(colorbar=False)
    np.testing.assert_equal(calls[0][0], [[0, 2, 4], [0, 2, 4]])
    np.testing.assert_equal(calls[0][1], [[10, 10, 10], [20, 20, 20]])
    np.testing.assert_equal(calls[0][2], [[10, 210, 410], [20, np.nan, 420]])
    data.plot(kind="matrix", rect=(1, 1, 2, 2), colorbar=False)
    np.testing.assert_equal(calls[1][0], [[2, 4], [2, 4]])


def test_matrix_coordinate_row_and_explicit_vectors(monkeypatch):
    data = Data(np.array([[0., 10., 20.], [1., 11., 21.], [2., 12., 22.]]))
    calls = spy_surface(monkeypatch)
    data.plot.matrix(x=0, y=0, colorbar=False)
    np.testing.assert_equal(calls[0][0], [[1, 2], [1, 2]])
    np.testing.assert_equal(calls[0][1], [[10, 10], [20, 20]])
    np.testing.assert_equal(calls[0][2], [[11, 12], [21, 22]])
    data.plot.matrix(x=[3, 4, 5], y=[6, 7, 8], colorbar=False)
    np.testing.assert_equal(calls[1][0], [[3, 4, 5]]*3)


def test_waterfall_uses_selected_axes_without_creating_hidden_figures(scalar):
    target = Figure().add_subplot()
    result = scalar.plot.waterfall(ax=target, y_shift=30)
    assert result is target and plt.get_fignums() == []
    assert len(target.lines) == 2
    assert target.get_ylabel() == "Value"


@pytest.mark.parametrize("kind", ["surface", "contour", "pcolormesh", "image", "scatter3d", "waterfall"])
def test_field_draws_preserve_measurements_styles_and_never_show_or_save(scalar, monkeypatch, kind):
    frame, mask, roles, metadata = scalar.data.copy(deep=True), scalar.mask.copy(), list(scalar.setas), deepcopy(scalar.metadata)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: pytest.fail("Automatic show"))
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: pytest.fail("Automatic save"))
    with scalar.plot.style({"axes.facecolor": "pink"}):
        ax = getattr(scalar.plot, kind)(colorbar=False)
        assert isinstance(ax, Axes)
        if kind != "waterfall":
            np.testing.assert_allclose(ax.get_facecolor(), colors.to_rgba("pink"))
    pd.testing.assert_frame_equal(scalar.data, frame)
    np.testing.assert_equal(scalar.mask, mask)
    assert list(scalar.setas) == roles and scalar.metadata == metadata


@pytest.mark.parametrize("options", [{"shape": (3, 0)}, {"shape": (3,)}, {"xlim": (1, 0)},
                                     {"projection": "3d"}, {"backend": "mayavi"}, {"xerr": 0.1}])
def test_invalid_field_requests_do_not_create_axes(scalar, options):
    with pytest.raises(ValueError):
        scalar.plot.surface(**options)
    assert plt.get_fignums() == []


def test_field_projection_and_label_contract(scalar):
    figure, target = plt.subplots()
    with pytest.raises(ValueError, match="projection"):
        scalar.plot.surface(ax=target)
    assert len(figure.axes) == 1
    ax = scalar.plot.surface(colorbar=False, xlabel="Custom", title=None)
    scalar.plot.surface(ax=ax, colorbar=False)
    assert ax.get_xlabel() == "Custom" and ax.get_zlabel() == "Value"
    assert scalar.plot.ax is ax
