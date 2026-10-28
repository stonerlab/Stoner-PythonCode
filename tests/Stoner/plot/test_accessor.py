"""Exercise the scientific contracts of the callable ordinary plotting API."""

import copy
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure

from Stoner import Data, Options
from Stoner.plot import DataPlotter
from Stoner.plot.selection import resolve_columns
from Stoner.tools.classes import copy_into


@pytest.fixture(autouse=True)
def isolate_figures():
    """Keep test-owned figures and rcParams isolated even after an assertion fails."""
    plt.close("all")
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def measurements():
    """Provide two Y/error pairs with deliberately different values."""
    x = np.arange(5, dtype=float)
    return Data(x, 2*x, np.full(5, 0.1), 3*x, np.full(5, 0.3),
                column_headers=["Field", "Signal A", "Error A", "Signal B", "Error B"], setas="xyeye")


def test_accessor_getter_has_no_rendering_or_storage_side_effects(measurements, monkeypatch):
    figure = plt.figure()
    owner = measurements.__dict__["_storage_owner"]
    monkeypatch.setattr(measurements, "column", lambda *args: pytest.fail("Getter read measurements"))
    plot = measurements.plot
    assert isinstance(plot, DataPlotter) and callable(plot)
    assert plot.ax is None and plot.figure is None and plot.axes == []
    assert figure.axes == []
    assert measurements.__dict__["_storage_owner"] is owner
    assert "_plot_state" not in measurements.__dict__


@pytest.mark.parametrize("width,kind", [(2, "xy"), (3, "xy"), (4, "xy"), (5, "vector"), (6, "vector3d")])
def test_count_defaults(width, kind):
    data = Data(np.ones((4, width)))
    columns = resolve_columns(data)
    assert columns.kind == kind
    if width <= 4:
        ax = data.plot()
        assert isinstance(ax, Axes)
        assert bool(ax.containers) == (width > 2)
        if width == 4:
            assert columns.x == 0 and columns.y == (2,) and columns.xerr == 1 and columns.yerr == (3,)


@pytest.mark.parametrize("roles,kind", [("xyz", "surface"), ("xyuv", "vector"), ("xyuvw", "vector"),
                                        ("xyzuvw", "vector3d"), ("xyzu", "voxels")])
def test_scientific_dispatch_description(roles, kind):
    assert resolve_columns(Data(np.ones((4, len(roles))), setas=roles)).kind == kind


def test_explicit_columns_and_keyword_format_repair_old_dispatch_defects():
    data = Data(np.column_stack([np.arange(4), np.arange(4)**2, np.full(4, 0.2)]))
    ax = data.plot(x=0, y=1, fmt="r-")
    assert isinstance(ax.containers[0], ErrorbarContainer)
    np.testing.assert_equal(ax.lines[0].get_ydata(), np.arange(4)**2)
    assert ax.lines[0].get_color() == "r"
    with pytest.raises(TypeError):
        data.plot(0, 1, "r-")


def test_explicit_kind_and_method_draw_same_plain_line(measurements):
    first = measurements.plot.line(y="Signal B")
    second = measurements.plot(kind="line", y="Signal B")
    assert second is first and not first.containers
    np.testing.assert_equal(first.lines[0].get_ydata(), first.lines[1].get_ydata())


def test_role_errors_follow_reordered_y_and_multi_x_group(measurements):
    ax = measurements.plot.errorbar(y=[3, 1])
    segments = [container.lines[2][0].get_segments()[1] for container in ax.containers]
    np.testing.assert_allclose(segments[0][:, 1], [2.7, 3.3])
    np.testing.assert_allclose(segments[1][:, 1], [1.9, 2.1])
    grouped = Data(np.ones((4, 7)), setas="xyexdye")
    columns = resolve_columns(grouped, startx=3)
    assert columns.x == 3 and columns.y == (5,) and columns.xerr == 4 and columns.yerr == (6,)


def test_exclusions_finite_values_and_errors_align_per_trace(measurements):
    measurements.mask[1, 1] = True
    measurements.mask[2, 4] = True
    measurements.data.iloc[3, 3] = np.nan
    snapshot = measurements.export_storage()
    ax = measurements.plot()
    np.testing.assert_equal(ax.containers[0].lines[0].get_xdata(), [0, 2, 3, 4])
    np.testing.assert_equal(ax.containers[1].lines[0].get_xdata(), [0, 1, 4])
    after = measurements.export_storage()
    np.testing.assert_equal(after.to_numpy(masked=False), snapshot.to_numpy(masked=False))
    np.testing.assert_equal(after.excluded, snapshot.excluded)
    assert after.schema == snapshot.schema and after.metadata == snapshot.metadata


def test_asymmetric_and_constant_uncertainties(measurements):
    asymmetric = np.array([[0.1]*5, [0.4]*5])
    ax = measurements.plot.errorbar(y=1, xerr=0.2, yerr=asymmetric)
    horizontal, vertical = ax.containers[0].lines[2]
    np.testing.assert_allclose(horizontal.get_segments()[2][:, 0], [1.8, 2.2])
    np.testing.assert_allclose(vertical.get_segments()[2][:, 1], [3.9, 4.4])


@pytest.mark.parametrize("arguments", [{"yerr": np.ones(3)}, {"yerr": -0.1}, {"y": [1, 3], "fmt": ["r-"]},
                                      {"kind": "unknown"}, {"kind": "scatter", "fmt": "r-"},
                                      {"kind": "errorfill", "xerr": 0.2}, {"figure": False},
                                      {"kind": "line", "yerr": 0.2}])
def test_invalid_requests_fail_before_creating_axes(measurements, arguments):
    with pytest.raises(ValueError):
        measurements.plot(**arguments)
    assert plt.get_fignums() == []


def test_empty_and_all_excluded_fail_without_drawing():
    for data in (Data(np.empty((0, 2)), setas="xy"), Data(np.ones((3, 2)), setas="xy")):
        data.mask = True
        with pytest.raises(ValueError, match="No finite"):
            data.plot()
        assert plt.get_fignums() == []


def test_raw_values_in_excluded_errors_do_not_cause_negative_error_failure(measurements):
    measurements.mask[2, 2] = True
    measurements.data.iloc[2, 2] = -100
    ax = measurements.plot.errorbar(y=1)
    np.testing.assert_equal(ax.containers[0].lines[0].get_xdata(), [0, 1, 3, 4])


def test_scatter_filters_per_row_sizes_and_colours(measurements):
    measurements.mask[1, 1] = True
    ax = measurements.plot.scatter(y=1, c=np.arange(5), s=np.arange(5)+10)
    np.testing.assert_equal(ax.collections[0].get_offsets()[:, 0], [0, 2, 3, 4])
    np.testing.assert_equal(ax.collections[0].get_array(), [0, 2, 3, 4])
    np.testing.assert_equal(ax.collections[0].get_sizes(), [10, 12, 13, 14])


def test_errorfill_uses_current_colour_cycle_without_private_matplotlib_api(measurements):
    ax = measurements.plot.errorfill(y=1, fmt="r-")
    assert ax.lines[0].get_color() == "r" and ax.collections
    measurements.plot.errorfill(y=3, ax=ax)
    assert len(ax.lines) == 2


def test_current_explicit_and_independent_targets(measurements):
    first = measurements.plot.line(y=1)
    other = Data(np.column_stack([np.arange(5), np.arange(5)]), setas="xy")
    assert other.plot() is first
    figure = measurements.plot.new_figure(figsize=(6, 4))
    assert isinstance(figure, Figure) and not figure.axes
    second = measurements.plot.line(y=3)
    assert second.figure is figure
    assert measurements.plot.line(y=1, ax=first) is first
    assert plt.gca() is second  # Explicit OO drawing does not change the current target.
    assert measurements.plot.ax is first and measurements.plot.figure is first.figure


def test_unmanaged_explicit_axes_and_incompatible_projection(measurements):
    standalone = Figure().add_subplot()
    assert measurements.plot.line(y=1, ax=standalone) is standalone
    three_d = plt.figure().add_subplot(projection="3d")
    with pytest.raises(ValueError, match="projection"):
        measurements.plot.line(y=1, ax=three_d)
    assert measurements.plot.line(y=1).name == "rectilinear"


def test_closed_removed_and_reused_targets_are_not_reported(measurements):
    ax = measurements.plot.line(y=1)
    number = ax.figure.number
    plt.close(ax.figure)
    replacement = plt.figure(number)
    assert measurements.plot.ax is None and measurements.plot.figure is None
    assert not replacement.axes
    ax = measurements.plot.line(y=1)
    ax.remove()
    assert measurements.plot.ax is None and measurements.plot.axes == []


def test_labels_metadata_and_existing_axis_configuration(measurements):
    measurements["temperature"] = 12
    measurements.plot.labels = ["Field / T", "Signal {temperature}", "err", "B", "err"]
    ax = measurements.plot.line(y=1, title="T = {temperature}")
    assert ax.get_xlabel() == "Field / T" and ax.get_ylabel() == "Signal 12"
    assert ax.lines[0].get_label() == "Signal 12" and ax.get_title() == "T = 12"
    ax.set(xlabel="User X", ylabel="User Y", title="User title")
    measurements.plot.line(y=3)
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("User X", "User Y", "User title")
    measurements.plot.line(y=1, label=None, title=None, xlabel="Override", legend=False)
    assert ax.get_title() == "" and ax.get_xlabel() == "Override"


def test_duplicate_headers_and_compiled_selectors(measurements):
    measurements.column_headers = ["X", "Y", "error", "Y", "error"]
    ax = measurements.plot.line(y=re.compile("^Y$"))
    np.testing.assert_equal(ax.lines[0].get_ydata(), 2*np.arange(5))
    np.testing.assert_equal(ax.lines[1].get_ydata(), 3*np.arange(5))
    assert resolve_columns(measurements, y="Y").y == (1, 3)


def test_retained_accessor_reads_current_values_and_roles(measurements):
    plot = measurements.plot
    plot.line(y=1)
    measurements.data.iloc[:, 1] = np.arange(5)+20
    measurements.setas = "x..y."
    ax = plot()
    np.testing.assert_equal(ax.lines[-1].get_ydata(), 3*np.arange(5))
    ax = plot.line(y=1)
    np.testing.assert_equal(ax.lines[-1].get_ydata(), np.arange(5)+20)


@pytest.mark.parametrize("copy_operation", [copy.copy, copy.deepcopy, lambda d: d.clone, lambda d: Data(d),
                                           lambda d: pickle.loads(pickle.dumps(d))])
def test_copies_detach_rendering_state_and_preserve_independent_labels(measurements, copy_operation):
    measurements.plot.labels[1] = "Custom"
    measurements.plot.line(y=1)
    figures = plt.get_fignums()
    cloned = copy_operation(measurements)
    assert cloned.plot.ax is None and cloned.plot.figure is None
    assert "_figure" not in cloned.__dict__ and plt.get_fignums() == figures
    assert cloned.plot.labels[1] == "Custom"
    cloned.plot.labels[1] = "Independent"
    assert measurements.plot.labels[1] == "Custom"


def test_copy_into_detaches_existing_destination_target(measurements):
    destination = measurements.clone
    destination.plot.line(y=1)
    measurements.plot.line(y=3)
    copy_into(measurements, destination)
    assert destination.plot.ax is None and "_figure" not in destination.__dict__


def test_no_automatic_show_save_or_legacy_return_policy(measurements, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: pytest.fail("Automatic show"))
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: pytest.fail("Automatic save"))
    monkeypatch.setattr(Options, "no_figs", True)
    assert not hasattr(measurements, "showfig")
    assert isinstance(measurements.plot(), Axes)
    assert isinstance(Data(np.ones((3, 2)), setas="xy").plot(), Axes)


def test_caller_style_is_not_reset(measurements):
    with plt.rc_context({"axes.facecolor": "pink", "lines.linewidth": 3.5}):
        ax = measurements.plot.line(y=1)
        assert ax.get_facecolor()[:3] == (1.0, 0.7529411764705882, 0.796078431372549)
        assert ax.lines[0].get_linewidth() == 3.5


def test_explicit_show_and_fit_annotation(measurements, monkeypatch):
    from Stoner.analysis.fitting.models.generic import Linear

    calls = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: calls.append(kwargs))
    measurements.plot.show(block=False)
    assert calls == [{"block": False}]
    measurements.lmfit(Linear, result=False)
    ax = measurements.plot.line(y=1)
    assert measurements.annotate_fit(Linear) is measurements
    assert ax.texts


def test_annotation_explicit_target_and_missing_plot(measurements):
    from Stoner.analysis.fitting.models.generic import Linear

    measurements.lmfit(Linear, result=False)
    with pytest.raises(ValueError, match="Plot data first"):
        measurements.annotate_fit(Linear)
    _, target = plt.subplots()
    assert measurements.annotate_fit(Linear, ax=target) is measurements
    assert target.texts and measurements.plot.ax is None


def test_raw_double_y_context_uses_current_target(measurements):
    stonerplots = pytest.importorskip("stonerplots")
    measurements.plot.line(y=1)
    figure, primary = plt.subplots()
    with stonerplots.DoubleYAxis(ax=primary, legend=False) as secondary:
        assert measurements.plot.line(y=3) is secondary
    assert measurements.plot.figure is figure
    assert len(secondary.lines) == 1 and plt.gca() is primary


def test_implicit_labels_follow_current_headers(measurements):
    measurements.plot.line(y=1)
    measurements.column_headers[1] = "Renamed signal"
    measurements.plot.new_figure()
    ax = measurements.plot.line(y=1)
    assert ax.get_ylabel() == "Renamed signal" and ax.lines[0].get_label() == "Renamed signal"


@pytest.mark.filterwarnings("error:.*:RuntimeWarning")
@pytest.mark.parametrize("errors", [np.zeros(2), np.array([[0.1, 0.2], [0.3, 0.4]])])
def test_errorfill_zero_and_two_sample_asymmetric_errors(errors):
    data = Data(np.arange(2), np.arange(2), setas="xy")
    ax = data.plot.errorfill(yerr=errors)
    assert ax.collections
    for collection in ax.collections:
        assert all(np.isfinite(path.vertices).all() for path in collection.get_paths())


def test_range_selector_restores_recent_plot_and_closes_only_its_figure(measurements, monkeypatch):
    from Stoner.tools.widgets import RangeSelect

    previous = measurements.plot.line(y=1)
    other = plt.figure()
    selector = RangeSelect()
    monkeypatch.setattr(plt, "pause", lambda delay: setattr(selector, "finished", True))
    selector(measurements, 0, 0.0)
    assert measurements.plot.ax is previous and plt.gcf() is other
    assert set(plt.get_fignums()) == {previous.figure.number, other.number}
    assert "_select" not in measurements.__dict__
