"""Exercise callable property forwarding on real Matplotlib rendering targets."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Stoner import Data


@pytest.fixture
def data():
    """Provide an independent measurement and clean up its figures."""
    with plt.rc_context():
        yield Data(np.arange(5.), np.arange(5.), setas="xy")
    plt.close("all")


def test_property_assignment_and_value_operations(data):
    ax = data.plot()
    data.plot.xlabel = "Field / T"
    data.plot.xlim = (0, 10)
    assert ax.get_xlabel() == "Field / T"
    assert data.plot.xlabel == "Field / T"
    assert data.plot.xlabel != "Other"
    assert str(data.plot.xlabel) == "Field / T"
    assert repr(data.plot.xlabel) == repr("Field / T")
    assert data.plot.xlabel.upper() == "FIELD / T"
    assert len(data.plot.xlabel) == 9 and bool(data.plot.xlabel)
    assert tuple(data.plot.xlim) == (0, 10) and data.plot.xlim[1] == 10
    np.testing.assert_array_equal(np.asarray(data.plot.xlim), [0, 10])
    assert "xlabel" in dir(data.plot) and "dpi" in dir(data.plot)
    assert "xlabel" not in data.__dict__ and "xlabel" not in data.plot.__dict__


def test_calls_preserve_native_setter_arguments_and_results(data):
    ax = data.plot()
    result = data.plot.title("Left title", loc="left", color="red")
    assert result is ax._left_title
    assert result.get_color() == "red"
    assert data.plot.title.get(loc="left") == "Left title"
    assert data.plot.title() == ax.get_title()
    data.plot.xticks.set([0.25, 0.75], labels=["Quarter", "Three quarters"], minor=True)
    np.testing.assert_array_equal(ax.get_xticks(minor=True), [0.25, 0.75])
    data.plot.xlim(left=-1, right=8)
    assert ax.get_xlim() == (-1, 8)


def test_getter_accepts_arguments_and_returns_native_values(data):
    ax = data.plot()
    renderer = ax.figure.canvas.get_renderer()
    bbox = data.plot.tightbbox.get(renderer)
    np.testing.assert_allclose(bbox.bounds, ax.get_tightbbox(renderer).bounds)
    assert isinstance(data.plot.xlabel.get(), str)
    with pytest.raises(AttributeError, match="set_tightbbox"):
        data.plot.tightbbox = bbox


def test_axes_precede_figure_and_figure_only_properties_work(data):
    ax = data.plot()
    ax.figure.set_label("Figure label")
    data.plot.label = "Axes label"
    assert data.plot.label() == ax.get_label() == "Axes label"
    assert ax.figure.get_label() == "Figure label"
    data.plot.dpi = 120
    assert data.plot.dpi() == ax.figure.get_dpi() == 120
    assert float(data.plot.dpi) == 120 and int(data.plot.dpi) == 120
    assert f"{data.plot.dpi:.1f}" == "120.0"
    data.plot.size_inches.set(7, 4, forward=False)
    np.testing.assert_array_equal(ax.figure.get_size_inches(), [7, 4])


def test_figure_properties_work_without_axes(data):
    figure = data.plot.new_figure()
    data.plot.dpi = 110
    assert data.plot.dpi() == figure.get_dpi() == 110
    assert data.plot.ax is None and not figure.axes
    with pytest.raises(AttributeError, match="xlabel"):
        data.plot.xlabel = "No axes"
    assert not figure.axes


def test_getter_and_setter_fall_back_independently(data, monkeypatch):
    ax = data.plot()
    monkeypatch.setattr(ax, "get_dpi", lambda: 42, raising=False)
    data.plot.dpi = 130
    assert data.plot.dpi() == 42
    assert ax.figure.get_dpi() == 130
    monkeypatch.setattr(ax, "set_custom", lambda *args, **kwargs: (args, kwargs), raising=False)
    assert data.plot.custom.set(1, 2, option=True) == ((1, 2), {"option": True})
    assert data.plot.custom.set() == ((), {})
    with pytest.raises(AttributeError, match="get_custom"):
        data.plot.custom()


def test_unknown_and_unrendered_properties_do_not_create_figures(data):
    before = plt.get_fignums()
    for name in ("xlabel", "dpi", "xlable", "_unknown"):
        assert not hasattr(data.plot, name)
    with pytest.raises(AttributeError):
        data.plot.xlabel = "Unrendered"
    assert plt.get_fignums() == before and "_plot_state" not in data.__dict__
    data.plot()
    with pytest.raises(AttributeError, match="xlable"):
        data.plot.xlable = "Typo"
    assert not hasattr(data.plot, "xlable")


def test_existing_accessor_attributes_keep_precedence(data):
    ax = data.plot()
    data.plot.labels = ["X", "Y"]
    assert data.plot.labels == ["X", "Y"]
    assert callable(data.plot.line) and data.plot.figure is ax.figure
    with pytest.raises(AttributeError):
        data.plot.axes = []
    assert data.plot.axes == [ax]


def test_saved_proxy_follows_owner_target_and_preserves_pyplot_selection(data):
    first = data.plot()
    title = data.plot.title
    other = Data(np.arange(5.), np.arange(5.), setas="xy")
    second = other.plot(ax=plt.subplots()[1])
    title("First")
    assert first.get_title() == "First" and second.get_title() != "First"
    assert plt.gca() is second
    data.plot(ax=second)
    title("Second")
    assert second.get_title() == "Second" and first.get_title() == "First"
    assert "title" not in type(data.plot).__dict__


def test_closed_removed_and_cloned_targets_are_not_reused(data):
    ax = data.plot()
    xlabel = data.plot.xlabel
    clone = data.clone
    assert not hasattr(clone.plot, "xlabel")
    ax.figure.delaxes(ax)
    with pytest.raises(AttributeError, match="set_xlabel"):
        xlabel("Detached")
    assert hasattr(data.plot, "dpi")
    plt.close(ax.figure)
    assert not hasattr(data.plot, "dpi")
    with pytest.raises(AttributeError, match="get_xlabel"):
        xlabel()


def test_native_errors_are_not_swallowed_or_redirected(data, monkeypatch):
    ax = data.plot()

    def reject(value):
        raise ValueError("Native setter rejected value")

    monkeypatch.setattr(ax, "set_label", reject)
    with pytest.raises(ValueError, match="Native setter rejected"):
        data.plot.label = "Rejected"
    assert ax.figure.get_label() == ""
    with pytest.raises(ValueError):
        data.plot.xscale = "invalid-scale"


def test_assignment_can_copy_another_property_value(data):
    data.plot()
    data.plot.xlabel = "Field"
    data.plot.ylabel = data.plot.xlabel
    assert data.plot.ylabel == data.plot.xlabel


def test_projection_specific_properties_do_not_leak(data):
    figure = plt.figure()
    three_d = figure.add_subplot(projection="3d")
    data.plot.scatter3d(x=0, y=1, z=1, ax=three_d)
    data.plot.zlabel = "Height"
    assert data.plot.zlabel() == "Height"
    two_d = plt.subplots()[1]
    data.plot(ax=two_d)
    assert not hasattr(data.plot, "zlabel") and "zlabel" not in dir(data.plot)
    assert three_d.get_zlabel() == "Height"
