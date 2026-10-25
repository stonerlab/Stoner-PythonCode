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
    "SavedFigure", "MultiPanel", "StackVertical", "DoubleYAxis", "DoubleXAxis", "DoubleXYAxis", "InsetPlot",
    "CentredAxes", "OffsetPlots", "PlotLabeller", "TexFormatter", "TexEngFormatter",
])
def test_direct_exports_preserve_upstream_api(data, name):
    assert getattr(data.plot, name) is getattr(stonerplots, name)
    assert getattr(plotting, name) is getattr(stonerplots, name)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("logarithmic", [False, True])
def test_offsets_pair_lines_and_scatter_without_changing_data(data, logarithmic):
    data.mask[2, 2] = True
    before = data.export_storage()
    _, previous = plt.subplots()
    with data.plot.OffsetPlots(offset=7, factor=3):
        ax = data.plot.line(y=1, yscale="log" if logarithmic else "linear")
        # The existing axes above is deliberately ignored by the default tracker.
        ax = data.plot.new_figure().subplots()
        for y in (1, 2):
            data.plot.line(y=y, ax=ax, yscale="log" if logarithmic else "linear")
            data.plot.scatter(y=y, ax=ax)
        original = ax.lines[1].get_ydata().copy()
        np.testing.assert_array_equal(original, data.column(2).compressed())
    expected = original * 3 if logarithmic else original + 7
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), expected)
    np.testing.assert_array_equal(ax.collections[1].get_offsets()[:, 1], expected)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), data.column(1).compressed())
    np.testing.assert_array_equal(previous.lines[0].get_ydata(), data.column(1).compressed())
    assert plt.gca() is previous
    assert data.plot.ax is ax
    after = data.export_storage()
    np.testing.assert_array_equal(after.to_numpy(masked=False), before.to_numpy(masked=False))
    np.testing.assert_array_equal(after.excluded, before.excluded)
    assert after.schema == before.schema and after.metadata == before.metadata


@pytest.mark.parametrize("fmt", ["o-", "none"])
@pytest.mark.parametrize("logarithmic", [False, True])
def test_offsets_move_complete_masked_errorbars(fmt, logarithmic):
    x = np.arange(1., 5.)
    source = Data(x, 2*x, 3*x, np.full(4, 0.2), setas="xyye")
    source.mask[1, 2] = True
    before = source.to_numpy().copy()
    with source.plot.OffsetPlots(offset=5, factor=4, pair=False):
        ax = source.plot.new_figure().subplots()
        for y in (1, 2):
            source.plot.errorbar(y=y, yerr=3, ax=ax, capsize=3, fmt=fmt,
                                 yscale="log" if logarithmic else "linear")
        containers = list(ax.containers)
        originals = [[(artist, np.array(artist.get_ydata(), dtype=float))
                      for artist in container.get_children() if hasattr(artist, "get_ydata")]
                     for container in containers]
        segments = [[collection.get_segments() for collection in container.lines[2]]
                    for container in containers]
    for index, container in enumerate(containers):
        def shifted(values, index=index):
            return values * 4**index if logarithmic else values + 5*index
        for artist, values in originals[index]:
            np.testing.assert_allclose(artist.get_ydata(), shifted(values))
        for collection, original in zip(container.lines[2], segments[index]):
            for actual, old in zip(collection.get_segments(), original):
                np.testing.assert_allclose(actual[:, 0], old[:, 0])
                np.testing.assert_allclose(actual[:, 1], shifted(old[:, 1]))
        assert len(container.lines[2][0].get_segments()) == (4 if index == 0 else 3)
    np.testing.assert_array_equal(source.to_numpy().data, before.data)
    np.testing.assert_array_equal(source.mask, before.mask)


@pytest.mark.parametrize("include_open", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_offsets_existing_axes_and_exception_restoration(data, include_open, fail):
    _, previous = plt.subplots()
    data.plot.line(y=[1, 2], ax=previous)
    original = previous.lines[1].get_ydata().copy()
    try:
        with data.plot.OffsetPlots(offset=9, include_open=include_open):
            new = data.plot.new_figure().subplots()
            data.plot.line(y=[1, 2], ax=new)
            if fail:
                raise RuntimeError("offset body")
    except RuntimeError as error:
        assert fail and str(error) == "offset body"
    assert plt.gca() is previous
    np.testing.assert_array_equal(previous.lines[1].get_ydata(),
                                  original + (9 if include_open and not fail else 0))
    np.testing.assert_array_equal(new.lines[1].get_ydata(), original + (0 if fail else 9))


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


@pytest.mark.parametrize("context_name", ["MultiPanel", "StackVertical", "DoubleYAxis", "DoubleXAxis", "DoubleXYAxis"])
def test_layout_exception_restores_entry_target(data, context_name):
    _, previous = plt.subplots()
    figure, parent = plt.subplots() if context_name.startswith("Double") else (data.plot.new_figure(), None)
    plt.sca(previous)
    context = (getattr(data.plot, context_name)(ax=parent, legend=False) if parent is not None else
               getattr(data.plot, context_name)(2, figure=figure, adjust_figsize=False))
    with pytest.raises(RuntimeError, match="layout body"), context as targets:
        ax = targets if parent is not None else targets[0]
        data.plot.line(y=1, ax=ax)
        raise RuntimeError("layout body")
    assert plt.gca() is previous
    assert len(figure.axes) == 2


@pytest.mark.parametrize("context_name, switch_name, share_x, share_y", [
    ("DoubleYAxis", "switch_to_y2", True, False),
    ("DoubleXAxis", "switch_to_x2", False, True),
    ("DoubleXYAxis", "switch_to_xy2", False, False),
])
@pytest.mark.parametrize("switch", [False, True])
def test_double_axes_accessors_preserve_scales_targets_and_data(data, context_name, switch_name, share_x, share_y, switch):
    figure, parent = plt.subplots()
    data.plot.line(y=1, ax=parent, label="Primary")
    other = Data(np.arange(5.)*10, 100 + np.arange(5.), setas="xy")
    before = data.to_numpy().copy()
    other_before = other.to_numpy().copy()
    _, previous = plt.subplots()
    with getattr(data.plot, context_name)(ax=parent, colours=("black", "red"),
                                          loc="upper left", **{switch_name: switch}) as secondary:
        assert plt.gca() is (secondary if switch else previous)
        if switch:
            assert other.plot.line(label="Secondary", color="red") is secondary
        else:
            assert other.plot.line(ax=secondary, label="Secondary", color="red") is secondary
            assert plt.gca() is previous
        assert parent.get_shared_x_axes().joined(parent, secondary) == share_x
        assert parent.get_shared_y_axes().joined(parent, secondary) == share_y
        if not share_x:
            parent.set_xlim(-1, 5)
            secondary.set_xlim(0, 50)
            assert parent.get_xlim() == (-1, 5)
        if not share_y:
            parent.set_ylim(-1, 10)
            secondary.set_ylim(90, 110)
            assert parent.get_ylim() == (-1, 10)
    assert plt.gca() is previous
    assert other.plot.ax is secondary
    assert figure.axes == [parent, secondary]
    assert [text.get_text() for text in parent.get_legend().get_texts()] == ["Primary", "Secondary"]
    for coordinate, shared in (("x", share_x), ("y", share_y)):
        if not shared:
            axis = getattr(secondary, f"{coordinate}axis")
            assert axis.get_label_position() == ("top" if coordinate == "x" else "right")
            assert axis.label.get_color() == "red"
            assert getattr(parent, f"{coordinate}axis").label.get_color() == "black"
    figure.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    figure.set_size_inches(7, 5)
    figure.canvas.draw()
    np.testing.assert_allclose(parent.get_position().bounds, secondary.get_position().bounds)
    np.testing.assert_array_equal(secondary.lines[0].get_xdata(), other_before[:, 0])
    np.testing.assert_array_equal(secondary.lines[0].get_ydata(), other_before[:, 1])
    np.testing.assert_array_equal(data.to_numpy(), before)
    np.testing.assert_array_equal(other.to_numpy(), other_before)


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


@pytest.mark.parametrize("styles", ["stoner, aps", " , stoner, , aps, ", ["stoner", "aps"], ("stoner", "aps")])
def test_style_names_and_overrides_match_saved_figure(data, styles):
    before = plt.rcParams.copy()
    overrides = {"lines.linewidth": 3.75, "axes.facecolor": "ivory"}
    with data.plot.SavedFigure(filename=False, style=styles, extra=overrides):
        expected = plt.rcParams.copy()
    with data.plot.style(styles, extra=overrides):
        assert plt.rcParams == expected
        assert plt.get_fignums() == []
    assert plt.rcParams == before


@pytest.mark.parametrize("styles", [None, [], ""])
def test_style_defaults_and_empty_styles_match_saved_figure(data, monkeypatch, styles):
    monkeypatch.setattr(stonerplots.default, "_style", ["ggplot"])
    plt.rcParams["axes.facecolor"] = "pink"
    with data.plot.SavedFigure(filename=False, style=styles):
        expected = plt.rcParams.copy()
    with data.plot.style(styles):
        assert plt.rcParams == expected
    with data.plot.style():
        assert plt.rcParams["axes.facecolor"] == plt.style.library["ggplot"]["axes.facecolor"]
    assert plt.rcParams["axes.facecolor"] == "pink"


def test_style_preserves_native_inputs_reset_and_nested_targets(data, tmp_path):
    stylesheet = tmp_path / "local.mplstyle"
    stylesheet.write_text("lines.linewidth: 4.25\n", encoding="utf-8")
    figure, ax = plt.subplots()
    plt.rcParams["axes.facecolor"] = "pink"
    with data.plot.style(stylesheet):
        assert plt.rcParams["lines.linewidth"] == 4.25
        with data.plot.style(["ggplot", {"lines.linewidth": 6}], extra={"lines.linewidth": 7}):
            assert plt.rcParams["lines.linewidth"] == 7
        assert plt.rcParams["lines.linewidth"] == 4.25
        with data.plot.style([], True, extra={"lines.linewidth": 2}):
            assert plt.rcParams["axes.facecolor"] == plt.rcParamsDefault["axes.facecolor"]
            assert plt.rcParams["lines.linewidth"] == 2
        assert plt.rcParams["axes.facecolor"] == "pink"
    assert plt.gca() is ax and plt.get_fignums() == [figure.number]


@pytest.mark.parametrize("extra, error", [
    ({"not.a.parameter": 1}, KeyError),
    ({"lines.linewidth": "invalid"}, ValueError),
    (["lines.linewidth"], TypeError),
])
def test_style_invalid_overrides_restore_caller_state(data, extra, error):
    before = plt.rcParams.copy()
    with pytest.raises(error), data.plot.style("ggplot", extra=extra):
        pytest.fail("Invalid overrides entered the body")
    assert plt.rcParams == before
    assert plt.get_fignums() == []


def test_style_extra_restores_on_body_exception(data):
    before = plt.rcParams.copy()
    with pytest.raises(RuntimeError), data.plot.style("stoner, aps", extra={"lines.linewidth": 9}):
        assert plt.rcParams["lines.linewidth"] == 9
        raise RuntimeError("style body")
    assert plt.rcParams == before
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
