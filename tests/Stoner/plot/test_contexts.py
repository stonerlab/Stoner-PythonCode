"""Exercise Stoner plotting inside the public stonerplots contexts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import stonerplots
from matplotlib.ticker import MultipleLocator

from Stoner import Data
from Stoner import plot as plotting


@pytest.fixture(autouse=True)
def isolate_plotting():
    """Restore caller styles and close test-owned figures after every test."""
    plt.close("all")
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def data():
    """Provide two traces with different scales and no uncertainty roles."""
    x = np.arange(5, dtype=float)
    return Data(x, 2*x, 100*x, setas="xyy", column_headers=["X", "A", "B"])


@pytest.mark.parametrize("name", [
    "SavedFigure", "MultiPanel", "StackVertical", "DoubleYAxis", "InsetPlot",
    "CentredAxes", "PlotLabeller", "TexFormatter", "TexEngFormatter",
])
def test_direct_exports_preserve_upstream_api(data, name):
    assert getattr(data.plot, name) is getattr(stonerplots, name)
    assert getattr(plotting, name) is getattr(stonerplots, name)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("fail", [False, True])
def test_nested_contexts_restore_targets_and_style(data, fail):
    previous, previous_ax = plt.subplots()
    plt.rcParams["lines.linewidth"] = 3.25
    figure = data.plot.new_figure(figsize=(8, 4))
    plt.sca(previous_ax)
    try:
        with (
            data.plot.SavedFigure(filename=False, style="stoner", extra={"lines.linewidth": 1.75}),
            data.plot.PlotLabeller(),
        ):
            # Trackers enumerate open figures on entry as well as exit.
            plt.sca(previous_ax)
            with data.plot.MultiPanel((1, 2), figure=figure, adjust_figsize=False) as panels:
                left = panels[0]
                assert data.plot.line(y=1) is left
                assert left.lines[0].get_linewidth() == 1.75
                with data.plot.InsetPlot(ax=left, loc="upper right") as inset:
                    assert data.plot.line(y=2) is inset
                assert plt.gca() is left
                assert data.plot.ax is inset and inset in figure.axes
                assert len(inset.lines) == 1
                right = panels[1]
                data.plot.line(y=1, ax=right)
                with data.plot.DoubleYAxis(ax=right) as twin:
                    assert data.plot.line(y=2) is twin
                    if fail:
                        raise RuntimeError("body failed")
                assert plt.gca() is right
            assert plt.gcf() is previous and plt.gca() is previous_ax
    except RuntimeError as error:
        assert fail and str(error) == "body failed"
    else:
        assert not fail
    # The tracking contexts scan figures on exit; they do not promise to select
    # the previous axes. Layout, inset and twin restoration is checked above.
    assert plt.rcParams["lines.linewidth"] == 3.25
    assert len(figure.axes) == 4
    assert len(plt.get_fignums()) == 2
    assert all(isinstance(ax.xaxis.get_major_formatter(), data.plot.TexEngFormatter) for ax in figure.axes)


@pytest.mark.parametrize("switch", [False, True])
def test_inset_exception_restores_unrelated_current_axes(data, switch):
    figure, parent = plt.subplots()
    _, previous = plt.subplots()
    with (
        pytest.raises(RuntimeError, match="inset body"),
        data.plot.InsetPlot(ax=parent, loc="upper right", switch_to_inset=switch) as inset,
    ):
        assert plt.gca() is (inset if switch else previous)
        data.plot.line(y=1, ax=inset)
        raise RuntimeError("inset body")
    assert plt.gca() is previous
    assert inset in figure.axes and len(inset.lines) == 1
    assert data.plot.ax is inset


@pytest.mark.parametrize("context_name", ["MultiPanel", "StackVertical", "DoubleYAxis"])
def test_layout_exception_restores_entry_target(data, context_name):
    _, previous = plt.subplots()
    figure, parent = plt.subplots() if context_name == "DoubleYAxis" else (data.plot.new_figure(), None)
    plt.sca(previous)
    context = (data.plot.DoubleYAxis(ax=parent, legend=False) if parent is not None else
               getattr(data.plot, context_name)(2, figure=figure, adjust_figsize=False))
    with pytest.raises(RuntimeError, match="layout body"), context as targets:
        ax = targets if parent is not None else targets[0]
        data.plot.line(y=1, ax=ax)
        raise RuntimeError("layout body")
    assert plt.gca() is previous
    assert len(figure.axes) == 2


def test_mixed_native_pandas_and_stoner_stacked_targets(data):
    figure = data.plot.new_figure()
    with data.plot.StackVertical(3, figure=figure, adjust_figsize=False) as panels:
        first, second, third = list(panels)
        data.plot.line(y=1, ax=first)
        pd.DataFrame({"x": [0, 1], "y": [2, 3]}).plot(x="x", y="y", ax=second, backend="matplotlib")
        third.plot([0, 1], [4, 5])
        assert first.get_shared_x_axes().joined(first, third)
    assert len(figure.axes) == 3 and len(plt.get_fignums()) == 1
    assert [len(ax.lines) for ax in figure.axes] == [1, 1, 1]


@pytest.mark.parametrize("existing", [False, True])
def test_saved_figure_writes_selected_figure_and_restores_style(data, tmp_path, existing):
    unrelated, _ = plt.subplots()
    figure = data.plot.new_figure() if existing else None
    filename = tmp_path / "measurement.png"
    before = plt.rcParams["axes.facecolor"]
    with data.plot.SavedFigure(filename, formats="png", style="stoner", use=figure,
                              extra={"axes.facecolor": "lightgrey"}):
        if not existing:
            data.plot.new_figure()
        ax = data.plot.line(y=1)
        assert ax.get_facecolor()[:3] == pytest.approx((211/255,)*3)
    assert filename.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert list(tmp_path.iterdir()) == [filename]
    assert plt.rcParams["axes.facecolor"] == before
    assert unrelated.number in plt.get_fignums()


def test_style_context_restores_on_error_without_creating_figures(data):
    before = plt.rcParams["lines.linewidth"]
    with pytest.raises(RuntimeError), data.plot.style({"lines.linewidth": before + 2}):
        assert plt.rcParams["lines.linewidth"] == before + 2
        raise RuntimeError
    assert plt.rcParams["lines.linewidth"] == before
    assert plt.get_fignums() == []


def test_labeller_instances_are_independent_and_existing_ticks_survive_draws(data):
    locator = MultipleLocator(0.5)
    formatter = data.plot.TexEngFormatter(unit="V")
    with data.plot.PlotLabeller(y=(locator, formatter)), data.plot.MultiPanel((1, 2)) as panels:
        first, second = list(panels)
        data.plot.line(y=1, ax=first)
        data.plot.line(y=2, ax=second)
    assert first.yaxis.get_major_locator() is not second.yaxis.get_major_locator()
    assert first.yaxis.get_major_formatter() is not second.yaxis.get_major_formatter()
    assert first.yaxis.get_major_locator().axis is first.yaxis
    assert second.yaxis.get_major_formatter().axis is second.yaxis
    assert locator.axis is None and formatter.axis is None
    existing = first.yaxis.get_major_formatter()
    data.plot.line(y=2, ax=first)
    assert first.yaxis.get_major_formatter() is existing


def test_repeated_context_instance_selects_new_targets(data):
    context = data.plot.InsetPlot(loc="upper right")
    for _ in range(2):
        data.plot.new_figure()
        parent = data.plot.line(y=1)
        with context as inset:
            data.plot.line(y=2)
        assert plt.gca() is parent
        assert len(parent.figure.axes) == 2
        assert data.plot.ax is inset


def test_centred_axes_context_does_not_add_axes(data):
    with data.plot.CentredAxes():
        ax = data.plot.line(y=1)
    assert len(ax.figure.axes) == 1
    assert ax.spines["left"].get_position() == ("data", 0.0)
