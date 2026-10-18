"""Verify page targets, serial bulk plotting and unchanged measurement state."""

import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from Stoner import Data, DataFolder, Options
from Stoner.folders import PlotFolder


@pytest.fixture(autouse=True)
def isolated_figures():
    """Restore styles and close test-owned figures."""
    with plt.rc_context():
        yield
    plt.close("all")


def make_folder(count, cls=PlotFolder):
    """Build independent in-memory measurements with different signals."""
    folder = cls(readlist=False)
    for index in range(count):
        data = Data(np.arange(4.), np.arange(4.)+index, setas="xy", filename=f"trace-{index}")
        data["trace"] = index
        folder += data
    return folder


@pytest.mark.parametrize("count,per_page,size", [(1, 1, (8, 6)), (5, 2, (8, 6)),
                                                (3, 12, (1, 10)), (4, 12, (10, 1)), (0, 2, (8, 6))])
def test_pagination_targets_and_unused_slots(count, per_page, size):
    folder = make_folder(count)
    calls = []
    result = folder.plot(plots_per_page=per_page, figsize=size,
                         plot_extra=lambda i, j, data: calls.append((i, j, data.plot.ax)))
    assert len(result) == count and all(isinstance(ax, Axes) for ax in result)
    assert len(folder.figures) == (count+per_page-1)//per_page
    assert sum(len(fig.axes) for fig in folder.figures) == count
    assert len(plt.get_fignums()) == len(folder.figures)
    for index, (ax, data) in enumerate(zip(result, folder)):
        np.testing.assert_equal(ax.lines[0].get_ydata(), np.arange(4.)+index)
        assert data.plot.ax is ax and calls[index] == (index, index % per_page, ax)
    if folder.figures:
        plt.close(folder.figures[0])
        assert len(folder.figures) == (count+per_page-1)//per_page-1


def test_mixed_2d_3d_members_use_their_page_without_hidden_figures():
    folder = make_folder(1)
    x, y = np.meshgrid([0., 1.], [0., 1.])
    folder += Data(x.ravel(), y.ravel(), (x+y).ravel(), setas="xyz")
    result = folder.plot()
    assert [ax.name for ax in result] == ["rectilinear", "3d"]
    assert result[0].figure is result[1].figure and len(plt.get_fignums()) == 1
    assert len(result[1].collections) == 1 and len(folder.figures[0].axes) == 3


def test_figure_settings_do_not_assign_member_targets_or_share_defaults():
    folder, other = make_folder(2), make_folder(1)
    fig = folder.figure(figsize=(4, 5), facecolor="pink")
    assert all(data.plot.figure is None for data in folder)
    assert fig.axes == [] and other.fig_defaults == {"figsize": (8, 6)}
    result = folder.plot(plots_per_page=1)
    assert len(folder.figures) == 2 and all(ax.figure is not fig for ax in result)
    assert all(tuple(page.get_size_inches()) == (4, 5) for page in folder.figures)
    cloned = copy.deepcopy(folder)
    assert cloned.figures == [] and all(data.plot.ax is None for data in cloned)


@pytest.mark.parametrize("kwargs", [{"plots_per_page": 0}, {"plots_per_page": True},
                                    {"plots_per_page": 1.2}, {"ax": None}, {"figure": False},
                                    {"kind": "unknown"}, {"plot_extra": "bad"}])
def test_invalid_page_controls_do_not_create_figures(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_folder(2).plot(**kwargs)
    assert not plt.get_fignums()


@pytest.mark.parametrize("method", ["auto", "line", "scatter", "hist"])
def test_bulk_plotting_is_serial_and_preserves_metadata(method, monkeypatch):
    folder = make_folder(3, DataFolder)
    before = [data.export_storage() for data in folder]
    fig, ax = plt.subplots()
    calls = []

    def pool(serial):
        calls.append(serial)
        assert serial, "Bulk plotting requested a thread/process pool"
        return None, map

    monkeypatch.setattr("Stoner.folders.each.get_pool", pool)
    monkeypatch.setattr(folder, "fetch", lambda: pytest.fail("Plotting used parallel prefetch"))
    monkeypatch.setattr(Options, "multiprocessing", True)
    monkeypatch.setattr(Options, "threading", True)
    proxy = folder.each.plot
    assert callable(proxy) and not ax.lines
    results = proxy(ax=ax) if method == "auto" else getattr(proxy, method)(ax=ax)
    assert results == [ax]*3 and calls == [True] and len(plt.get_fignums()) == 1
    for original, data in zip(before, folder):
        after = data.export_storage()
        np.testing.assert_equal(after.to_numpy(masked=False), original.to_numpy(masked=False))
        np.testing.assert_equal(after.excluded, original.excluded)
        assert after.metadata == original.metadata and after.schema == original.schema
        assert data.plot.ax is ax
    assert ax.figure is fig


def test_bulk_dotted_strings_iteration_and_repeated_live_selection():
    folder = make_folder(2)
    ax = plt.figure().subplots()
    assert folder.each("plot.line", ax=ax) == [ax, ax]
    folder[0].data.iloc[:, 1] += 10
    assert list(folder.each.iter("plot.scatter", ax=ax)) == [ax, ax]
    np.testing.assert_equal(ax.collections[0].get_offsets()[:, 1], np.arange(4.)+10)
    assert not any("func" in data for data in folder)


@pytest.mark.parametrize("attribute", ["ax", "figure", "labels", "style", "SavedFigure", "new_figure", "missing"])
def test_bulk_accessors_do_not_collect_configuration_attributes(attribute):
    folder = make_folder(2)
    before = [data.metadata.copy() for data in folder]
    with pytest.raises(AttributeError):
        getattr(folder.each.plot, attribute)
    assert not plt.get_fignums() and all(data.metadata == old for data, old in zip(folder, before))


def test_empty_bulk_folder_and_explicit_return_storage():
    folder = make_folder(0)
    assert folder.each.plot.line() == [] and not plt.get_fignums()
    folder = make_folder(1)
    result = folder.each.plot.line(_return="rendered")
    assert folder[0]["rendered"] is result[0] and isinstance(result[0], Axes)


def test_generic_partial_callable_name_and_result_storage_remain_supported():
    folder = make_folder(2)

    def size(data, offset=0):
        return len(data)+offset

    result = folder.each(partial(size, offset=3), _serial=True, _return=True)
    assert result == [7, 7] and folder[0]["partial"] == 7
