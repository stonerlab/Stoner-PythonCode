"""Exercise ROI geometry, atomic state, scientific eligibility and handle lifetimes."""

import json
import pickle
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from Stoner import ROI, Data, ImageFile, ImageStack
from Stoner.core.selection import resolve_selection
from Stoner.enums import CoordinateMode, FitOutput
from Stoner.Image import ImageROI
from Stoner.Image.storage import ImageStorage


@pytest.fixture(autouse=True)
def figures():
    plt.close("all")
    yield
    plt.close("all")


def image():
    values = np.arange(20, dtype=float).reshape(4, 5)
    values[1, 1] = np.nan
    result = ImageFile(values)
    result.mask[1, 2] = True
    return result


def native():
    package = ImageStorage.from_numpy(np.arange(20, dtype=float).reshape(4, 5))
    package.dataset = package.dataset.assign_coords(x=[20, 17, 13, 11, 10], y=[2, 4, 7, 12])
    for axis in ("x", "y"):
        package.dataset[axis].attrs["units"] = "um"
    return ImageFile.from_storage(package)


def test_lookup_and_builders_are_inert():
    im = image()
    before = im.export_storage()
    assert isinstance(im.roi, ImageROI)
    assert im.roi.current is None and not im.roi.active and not im.roi.stale
    assert "_roi_state" not in im.__dict__
    assert ROI.box(1, 4, 1, 3) == ROI.box(1, 4, 1, 3)
    assert not plt.get_fignums()
    xr.testing.assert_identical(im.image, before.dataset)
    with pytest.raises(TypeError, match="active"):
        bool(im.roi)
    with pytest.raises(TypeError):
        ROI()


def test_half_open_box_preserves_values_masks_and_metadata():
    im = image()
    before = im.export_storage()
    accessor = im.roi
    assert accessor.box(1, 4, 1, 3) is im
    expected = np.zeros(im.shape, bool)
    expected[1:3, 1:4] = True
    np.testing.assert_array_equal(accessor.resolve(), expected)
    assert accessor.active and not accessor.stale
    assert accessor.resolve().dtype == bool
    assert accessor.resolve().name == "included"
    assert accessor.resolve().attrs == {}
    assert int(accessor.resolve().sum()) == 6  # Includes NaN and excluded pixel.
    xr.testing.assert_identical(im.image, before.dataset)
    assert dict(im.metadata) == dict(before.metadata)
    assert accessor.clear() is im
    assert accessor.resolve().values.all()
    assert not accessor.active
    xr.testing.assert_identical(im.image, before.dataset)


@pytest.mark.parametrize("bounds,count", [((-5, 2, -3, 2), 4), ((20, 30, 20, 30), 0),
                                         ((1, 1, 0, 4), 0), ((0, 5, 0, 4), 20)])
def test_bounds_no_wrapping(bounds, count):
    im = image()
    im.roi.box(*bounds)
    assert im.roi.active and int(im.roi.resolve().sum()) == count


@pytest.mark.parametrize("bounds", [(3, 1, 0, 1), (0, 1, 2, 1), (0, np.nan, 0, 1),
                                  (0, np.inf, 0, 1), (False, 1, 0, 1), ("0", 1, 0, 1)])
def test_invalid_box_atomic(bounds):
    im = image()
    im.roi.box(1, 3, 0, 2)
    previous = im.roi.current
    with pytest.raises((TypeError, ValueError)):
        im.roi.box(*bounds)
    assert im.roi.current is previous


def test_native_nonuniform_decreasing_grid_and_unit_compatibility():
    im = native()
    im.roi.box(11, 18, 4, 12, coordinates=CoordinateMode.NATIVE)
    expected = np.zeros(im.shape, bool)
    expected[1:3, 1:4] = True
    np.testing.assert_array_equal(im.roi.resolve(), expected)
    region = im.roi.current
    other = native()
    assert other.roi.set(region) is other
    wrong = other.export_storage()
    wrong.dataset.x.attrs["units"] = "nm"
    incompatible = ImageFile.from_storage(wrong)
    with pytest.raises(ValueError, match="units"):
        incompatible.roi.set(region)
    assert not incompatible.roi.active
    with pytest.raises(ValueError, match="reference"):
        ROI.box(0, 1, 0, 1, coordinates="native")
    with pytest.raises(TypeError):
        im.roi.box(0, 1, 0, 1, coordinates=FitOutput.DATA)


def test_mask_is_immutable_and_exactly_bound_to_grid():
    im = native()
    included = np.zeros(im.shape, bool)
    included[1:3, 2:4] = True
    assert im.roi.from_mask(included) is im
    region = im.roi.current
    included[:] = False
    assert int(im.roi.resolve().sum()) == 4
    detached = im.roi.resolve()
    detached.values[:] = False
    detached.x.attrs["units"] = "changed"
    assert int(im.roi.resolve().sum()) == 4
    assert not im.roi.stale
    with pytest.raises(FrozenInstanceError):
        region._kind = "all"
    for invalid in (np.ones(im.shape), np.ones((4, 1), bool), im.roi.resolve().isel(x=slice(None, None, -1)),
                    np.ma.array(np.ones(im.shape, bool), mask=True)):
        with pytest.raises(ValueError):
            im.roi.from_mask(invalid)
        assert im.roi.current == region
    other = image()
    with pytest.raises(ValueError, match="geometry"):
        other.roi.set(region)


def test_composition_holes_empty_full_and_mixed_coordinate_rejection():
    im = image()
    a, b = ROI.box(0, 3, 0, 4), ROI.box(2, 5, 0, 4)
    assert im.roi.set(a | b) is im
    assert im.roi.resolve().values.all()
    im.roi.set(a & b)
    assert int(im.roi.resolve().sum()) == 4
    im.roi.set(ROI.all() - (a & b))
    assert int(im.roi.resolve().sum()) == 16
    im.roi.set(~ROI.all())
    assert im.roi.active and not im.roi.resolve().values.any()
    assert im.roi.add(a) is im
    assert im.roi.subtract(b) is im
    assert int(im.roi.resolve().sum()) == 8
    im.roi.clear()
    assert im.roi.intersect(a) is im
    assert im.roi.current.resolve(im.image.intensity).values.sum() == 12
    with pytest.raises(ValueError, match="pixel and native"):
        _ = a | ROI.box(0, 1, 0, 1, coordinates="native", reference=im.image.intensity)


@pytest.mark.parametrize("builder,args,count", [("circle", ((2, 2), 1), 5),
                                               ("ellipse", ((2, 2), (2, 1)), 7),
                                               ("polygon", ([(1, 1), (3, 1), (3, 3), (1, 3)],), 9)])
def test_shape_centres_and_closed_boundaries(builder, args, count):
    im = image()
    assert getattr(im.roi, builder)(*args) is im
    assert int(im.roi.resolve().sum()) == count
    assert not plt.get_fignums()


def test_native_circle_respects_anisotropic_spacing():
    im = native()
    im.roi.circle((13, 4), 3, coordinates="native")
    x, y = np.meshgrid(im.image.x.values, im.image.y.values)
    np.testing.assert_array_equal(im.roi.resolve(), (x-13)**2 + (y-4)**2 <= 9)


@pytest.mark.parametrize("builder,args", [("circle", ((0, 0), -1)), ("ellipse", ((0, 0), (0, 1))),
                                        ("polygon", ([(0, 0), (1, 1)],)),
                                        ("polygon", ([(0, 0), (1, 1), (2, 2)],))])
def test_invalid_shapes(builder, args):
    with pytest.raises(ValueError):
        getattr(ROI, builder)(*args)


def test_reduction_current_explicit_all_precedence_and_empty():
    im = image()
    im.roi.box(1, 4, 1, 3)
    expected = np.nanmean([8, 11, 12, 13])
    assert im.mean() == expected
    assert im.mean(where=ROI.all()) == np.nanmean(np.delete(im.to_numpy(masked=False).ravel(), 7))
    assert im.mean(where=ROI.box(0, 1, 0, 1)) == 0
    assert int(im.roi.resolve().sum()) == 6
    im.roi.set(~ROI.all())
    assert np.isnan(im.mean())
    assert np.isnan(im.mean(axis=0)).all()
    im.roi.clear()
    assert np.isfinite(im.mean())


def test_histogram_uses_current_roi_but_image_display_is_whole_domain():
    im = image()
    im.roi.box(1, 4, 1, 3)
    _fig, (display, hist) = plt.subplots(1, 2)
    assert im.plot(ax=display) is display
    assert display.images[0].get_array().count() == 18
    assert im.plot.hist(ax=hist, bins=[-1, 100]) is hist
    assert hist.patches[0].get_height() == 4
    hist.clear()
    im.plot.hist(ax=hist, bins=[-1, 100], where=ROI.all())
    assert hist.patches[0].get_height() == 18
    display.clear()
    im.plot(ax=display, where=ROI.box(0, 1, 0, 1))
    assert display.images[0].get_array().count() == 1


def test_outline_exact_cell_edges_preserves_orientation_and_handles_holes():
    im = native()
    im.roi.box(1, 3, 1, 3)
    ax = im.plot()
    limits = ax.get_xlim(), ax.get_ylim()
    assert im.plot.roi(ax=ax) is ax
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    segments = ax.collections[-1].get_segments()
    assert len(segments) == 8
    points = np.concatenate(segments)
    assert set(points[:, 0]) == {18.5, 15, 12}
    assert set(points[:, 1]) == {3, 5.5, 9.5}
    im.roi.set(~ROI.all())
    im.plot.roi(ax=ax)
    assert not ax.collections[-1].get_segments()


@pytest.mark.parametrize("copier", [lambda im: im.clone, copy, deepcopy, lambda im: pickle.loads(pickle.dumps(im))])
def test_copy_pickle_and_value_edits_preserve_independent_state(copier):
    im = image()
    im.roi.box(1, 4, 1, 3)
    other = copier(im)
    np.testing.assert_array_equal(other.roi.resolve(), im.roi.resolve())
    other.roi.clear()
    assert im.roi.active
    with im.edit_numpy() as draft:
        draft[:] = 3
    assert not im.roi.stale
    im += 1
    assert not im.roi.stale


@pytest.mark.parametrize("operation", [lambda im: im.flip_h, lambda im: im.flip_v, lambda im: im.CW,
                                      lambda im: im.T, lambda im: im.rotate(10, _=None),
                                      lambda im: im.crop(1, 4, 1, 3, _=None), lambda im: im[1:3, 1:4]])
def test_geometry_changes_fail_explicitly_without_losing_intent(operation):
    im = image()
    im.roi.box(1, 4, 1, 3)
    other = operation(im)
    assert other.roi.active and other.roi.stale
    with pytest.raises(ValueError, match="stale"):
        other.roi.resolve()
    with pytest.raises(ValueError, match="stale"):
        other.mean()
    assert np.isfinite(other.mean(where=ROI.all()))
    other.roi.box(0, 1, 0, 1)
    assert not other.roi.stale
    assert not im.roi.stale


def test_frames_do_not_share_current_roi_and_storage_does_not_persist_it():
    stack = ImageStack([image(), image()])
    first, second = stack[0], stack[1]
    first.roi.box(0, 1, 0, 1)
    assert not second.roi.active
    assert not stack[0].roi.active  # New frame handle has separate operational state.
    restored = ImageFile.from_storage(first.export_storage())
    assert not restored.roi.active


def test_pickle_omits_an_active_selector_and_preserves_last_accepted_roi():
    im = image()
    im.roi.box(0, 2, 0, 2)
    im.__dict__["_roi_selector"] = lambda: None  # Deliberately unpicklable GUI stand-in.
    restored = pickle.loads(pickle.dumps(im))
    assert "_roi_selector" not in restored.__dict__
    assert restored.roi.current == im.roi.current


def test_shape_replacement_keeps_roi_stale_even_if_original_shape_is_restored():
    im = image()
    im.roi.box(0, 2, 0, 2)
    im.image = np.ones((2, 2))
    assert im.roi.stale
    im.image = np.ones((4, 5))
    assert im.roi.stale
    im.roi.clear()
    assert not im.roi.stale


@pytest.mark.parametrize("builder", [lambda im: ROI.box(0, 3, 0, 2),
                                     lambda im: ROI.box(11, 18, 3, 8, coordinates="native",
                                                       reference=im.image.intensity),
                                     lambda im: ROI.from_mask(np.ones(im.shape, bool), reference=im.image.intensity),
                                     lambda im: ~ROI.circle((2, 2), 1) | ROI.box(0, 1, 0, 4)])
def test_explicit_json_replay(builder):
    im = native()
    original = builder(im)
    record = original.to_dict()
    restored = ROI.from_dict(json.loads(json.dumps(record)))
    assert restored == original
    np.testing.assert_array_equal(restored.resolve(im.image.intensity), original.resolve(im.image.intensity))
    record["version"] = 2
    with pytest.raises(ValueError):
        ROI.from_dict(record)


def test_data_and_low_level_selection_remain_state_independent():
    im = image()
    im.roi.set(~ROI.all())
    assert resolve_selection(None, im.image.intensity).all()
    data = Data(np.arange(12).reshape(4, 3))
    assert not hasattr(data, "roi")


@pytest.mark.parametrize("kind", ["rectangle", "polygon"])
@pytest.mark.parametrize("outcome", ["accept", "escape", "close", "error"])
@pytest.mark.parametrize("borrowed", [False, True])
def test_interactive_lifecycle(monkeypatch, kind, outcome, borrowed):
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.backend_bases import FigureCanvasBase

    im = image()
    im.roi.box(0, 1, 0, 1)
    previous_roi = im.roi.current
    fig, ax = plt.subplots()
    line, = ax.plot([0, 3], [0, 2])
    previous_axes = im.plot(ax=ax)
    ax.set_title("Original")
    ax.set_xlabel("Original x")
    original_children = set(ax.get_children())
    original_limits = ax.get_xlim(), ax.get_ylim()
    before_callbacks = {name: set(items) for name, items in fig.canvas.callbacks.callbacks.items()}
    manager = Gcf.get_active()
    before_figures = plt.get_fignums()

    def event_loop(canvas, timeout):
        session = im.__dict__["_roi_selector"]
        if kind == "rectangle":
            session.rectangle(SimpleNamespace(xdata=1, ydata=1), SimpleNamespace(xdata=4, ydata=3))
        else:
            session.polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        if outcome == "error":
            raise RuntimeError("Injected backend failure")
        if outcome == "close":
            session.close(SimpleNamespace())
        else:
            session.key(SimpleNamespace(key="enter" if outcome == "accept" else "escape"))

    monkeypatch.setattr(FigureCanvasBase, "start_event_loop", event_loop)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    method = getattr(im.roi, f"select_{kind}")
    if outcome == "error":
        with pytest.raises(RuntimeError, match="backend"):
            method(ax=ax if borrowed else None)
    else:
        assert method(ax=ax if borrowed else None) is im
    if outcome == "accept":
        assert im.roi.current != previous_roi
        assert int(im.roi.resolve().sum()) == (6 if kind == "rectangle" else 9)
    else:
        assert im.roi.current == previous_roi
    assert im.plot.ax is previous_axes
    assert "_roi_selector" not in im.__dict__
    assert set(ax.get_children()) == original_children
    assert (ax.get_xlim(), ax.get_ylim()) == original_limits
    assert ax.get_title() == "Original"
    assert ax.get_xlabel() == "Original x"
    assert line in ax.lines
    assert plt.get_fignums() == before_figures
    assert Gcf.get_active() is manager
    after_callbacks = {name: set(items) for name, items in fig.canvas.callbacks.callbacks.items()}
    assert after_callbacks == before_callbacks


def test_invalid_interactive_grid_and_target_create_no_figure():
    im = image()
    with pytest.raises(ValueError):
        im.roi.select_rectangle(coordinates="invalid")
    with pytest.raises(ValueError):
        im.roi.select_polygon(ax=object())
    assert not plt.get_fignums()


def test_native_geometry_uses_crop_offsets_and_boolean_mask_rejects_cropped_grid():
    im = native()
    region = ROI.box(10, 18, 4, 12, coordinates="native", reference=im.image.intensity)
    child = im[1:3, 1:4]
    child.roi.set(region)
    assert child.roi.resolve().values.all()
    child.roi.box(0, 1, 0, 1)
    assert int(child.roi.resolve().sum()) == 1
    assert not im.roi.active
    mask = ROI.from_mask(np.ones(im.shape, bool), reference=im.image.intensity)
    with pytest.raises(ValueError):
        child.roi.set(mask)


@pytest.mark.parametrize("record", [None, {}, {"version": 2, "kind": "all", "values": [],
                                            "coordinates": None, "units": None},
                                    {"version": 1, "kind": "execute", "values": [],
                                     "coordinates": None, "units": None}])
def test_invalid_replay_records(record):
    with pytest.raises(ValueError):
        ROI.from_dict(record)
