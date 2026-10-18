"""Check statistical sample selection, exclusions and native artist values."""

import re
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from Stoner import Data


@pytest.fixture(autouse=True)
def isolated_figures():
    """Restore styles and close test-owned figures even on failure."""
    plt.close("all")
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def samples():
    """Provide independent Y series, weights, errors and unused excluded values."""
    data = Data(np.arange(6.), np.array([0., 1., 2., 3., 4., 100.]),
                np.arange(10., 16.), np.ones(6), np.arange(1., 7.), np.zeros(6),
                setas="xyye..", column_headers=["X", "A", "B", "Error", "Weight", "Unused"])
    data.mask[:, 0] = True
    data.mask[5, 1] = True
    data.mask[0, 2] = True
    data.mask[:, 3] = True
    data.mask[:, 5] = True
    data.data.iloc[1, 2] = np.inf
    return data


def test_hist_independent_series_ignore_x_errors_and_unrelated_masks(samples):
    ax = samples.plot.hist(bins=[0, 5, 10, 15, 20])
    np.testing.assert_equal([patch.get_height() for patch in ax.containers[0]], [5, 0, 0, 0])
    np.testing.assert_equal([patch.get_height() for patch in ax.containers[1]], [0, 0, 3, 1])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B"]


def test_hist_weights_align_independently_with_each_series(samples):
    samples.mask[2, 4] = True
    ax = samples.plot.hist(y=[1, 2], weights="Weight", bins=[0, 5, 10, 20])
    assert sum(patch.get_height() for patch in ax.containers[0]) == 12
    assert sum(patch.get_height() for patch in ax.containers[1]) == 15
    assert ax.get_ylabel() == "Weighted count"


def test_single_column_and_unassigned_numeric_defaults():
    data = Data(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))
    ax = data.plot.hist(bins=3)
    assert len(ax.containers) == 2
    plt.close("all")
    ax = Data(np.arange(4.)[:, None]).plot.hist()
    assert sum(patch.get_height() for patch in ax.patches) == 4


@pytest.mark.parametrize("kind", ["hist", "box", "kde"])
def test_named_and_kind_calls_use_identical_samples(samples, kind):
    figure, axes = plt.subplots(1, 2)
    first = getattr(samples.plot, kind)(y=1, ax=axes[0])
    second = samples.plot(kind=kind, y=1, ax=axes[1])
    assert first is axes[0] and second is axes[1] and samples.plot.ax is second
    if kind == "hist":
        np.testing.assert_equal([p.get_height() for p in first.patches], [p.get_height() for p in second.patches])
    else:
        for left, right in zip(first.lines, second.lines):
            np.testing.assert_equal(left.get_data(), right.get_data())
    assert samples.plot.figure is figure


def test_duplicate_headers_and_regex_keep_stoner_selector_order(monkeypatch):
    data = Data(np.arange(5.), np.arange(5.)+10, column_headers=["Signal", "Signal"])
    seen = []
    original = Axes.hist

    def capture(ax, values, **kwargs):
        seen.extend(values)
        return original(ax, values, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture)
    data.plot.hist(y=[1, 0], bins=2)
    np.testing.assert_equal(seen, [np.arange(5.)+10, np.arange(5.)])
    seen.clear()
    data.plot.hist(y=re.compile("Signal"), bins=2)
    # Follow Stoner's existing duplicate-pattern selection rather than pandas labels.
    expected = data.find_col(re.compile("Signal"), force_list=True)
    np.testing.assert_equal(seen, [data.column(position) for position in expected])


def test_only_required_columns_are_read_and_changed_data_is_visible(samples, monkeypatch):
    seen = []
    original = samples.column

    def read(position, *args, **kwargs):
        seen.append(position)
        return original(position, *args, **kwargs)

    monkeypatch.setattr(samples, "column", read)
    plot = samples.plot
    plot.hist(y=1, weights=4, bins=[0, 10, 200])
    assert seen == [4, 1]
    samples.data.iloc[0, 1] = 50
    seen.clear()
    figure = plot.new_figure()
    ax = plot.hist(y=1, bins=[0, 10, 200])
    np.testing.assert_equal([patch.get_height() for patch in ax.patches], [4, 1])
    assert seen == [1] and ax.figure is figure


def test_box_statistics_and_categories_ignore_other_series_masks(samples):
    ax = samples.plot.box(y=[2, 1], showfliers=False, medianprops={"color": "purple"})
    medians = [line.get_ydata()[0] for line in ax.lines if line.get_color() == "purple"]
    np.testing.assert_equal(medians, [13.5, 2.])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["B", "A"]


def test_box_horizontal_orientation_labels_and_positions(samples):
    ax = samples.plot.box(y=1, orientation="horizontal", positions=[2], label="Category")
    assert ax.get_xlabel() == "A" and ax.get_ylabel() == ""
    assert [label.get_text() for label in ax.get_yticklabels()] == ["Category"]
    assert ax.get_yticks()[0] == 2


def test_kde_matches_scipy_with_independent_masks_and_bandwidth(samples):
    points = np.linspace(-1, 18, 31)
    ax = samples.plot.kde(ind=points, bw_method=0.4)
    for line, values in zip(ax.lines, (np.arange(5.), np.arange(12., 16.))):
        np.testing.assert_equal(line.get_xdata(), points)
        np.testing.assert_allclose(line.get_ydata(), gaussian_kde(values, bw_method=0.4)(points))
    assert ax.get_ylabel() == "Density"


def test_kde_weights_and_default_range_ignore_zero_weight_outlier():
    data = Data(np.array([0., 1., 2., 1000.]), np.array([1., 2., 1., 0.]))
    ax = data.plot.kde(y=0, weights=1, ind=9)
    points = np.linspace(-1, 3, 9)
    np.testing.assert_equal(ax.lines[0].get_xdata(), points)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), gaussian_kde([0., 1., 2.], weights=[1, 2, 1])(points))


def test_hexbin_counts_and_c_reduction_match_direct_matplotlib():
    data = Data(np.arange(8.), np.arange(8.) % 3, np.arange(8.)*10, np.ones(8), setas="xyze")
    data.mask[1, 0] = True
    data.mask[2, 1] = True
    data.mask[3, 2] = True
    data.mask[:, 3] = True
    figure, axes = plt.subplots(1, 3)
    count = data.plot.hexbin(ax=axes[0], gridsize=3, mincnt=1, colorbar=False)
    assert count.collections[0].get_array().sum() == 6  # C's mask is irrelevant to counts.
    actual = data.plot(kind="hexbin", C=2, ax=axes[1], gridsize=3, mincnt=1,
                       reduce_C_function=np.sum, colorbar=False)
    valid = np.array([0, 4, 5, 6, 7])
    expected = axes[2].hexbin(np.arange(8.)[valid], (np.arange(8.) % 3)[valid], C=np.arange(8.)[valid]*10,
                              gridsize=3, mincnt=1, reduce_C_function=np.sum)
    np.testing.assert_equal(actual.collections[0].get_offsets(), expected.get_offsets())
    np.testing.assert_equal(actual.collections[0].get_array(), expected.get_array())
    assert actual.figure is figure


def test_hexbin_masked_array_c_and_primary_target_with_oo_figure():
    data = Data(np.arange(4.), np.arange(4.))
    figure = Figure()
    ax = figure.subplots()
    values = np.ma.array([1., 2., 3., np.nan], mask=[0, 1, 0, 0])
    assert data.plot.hexbin(C=values, ax=ax, gridsize=2, mincnt=1, reduce_C_function=np.sum) is ax
    assert len(figure.axes) == 2 and data.plot.ax is ax and not plt.get_fignums()
    assert ax.collections[0].get_array().sum() == 4


def test_hexbin_logarithmic_bins_match_native_renderer():
    data = Data(np.geomspace(1., 100., 15), np.geomspace(2., 50., 15))
    figure, axes = plt.subplots(1, 2)
    result = data.plot.hexbin(ax=axes[0], xscale="log", yscale="log", colorbar=False)
    expected = axes[1].hexbin(data.column(0), data.column(1), xscale="log", yscale="log")
    np.testing.assert_equal(result.collections[0].get_array(), expected.get_array())
    assert result.get_xscale() == result.get_yscale() == "log" and result.figure is figure


@pytest.mark.parametrize("kind", ["hist", "box", "kde", "hexbin"])
def test_rendering_preserves_storage_styles_and_ignores_pandas_backend(samples, monkeypatch, kind):
    samples.mask[:, 0] = False
    backend = ModuleType("stoner_test_backend")
    backend.plot = lambda *args, **kwargs: pytest.fail("Native statistical rendering called a pandas backend")
    monkeypatch.setitem(__import__("sys").modules, backend.__name__, backend)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: pytest.fail("Implicit show"))
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: pytest.fail("Implicit save"))
    snapshot = samples.export_storage()
    owner = samples.__dict__["_storage_owner"]
    with pd.option_context("plotting.backend", backend.__name__), plt.rc_context({"axes.facecolor": "pink"}):
        ax = getattr(samples.plot, kind)(y=1)
        assert ax.get_facecolor() == (1., 0.7529411764705882, 0.796078431372549, 1.)
    after = samples.export_storage()
    np.testing.assert_equal(after.to_numpy(masked=False), snapshot.to_numpy(masked=False))
    np.testing.assert_equal(after.excluded, snapshot.excluded)
    assert after.schema == snapshot.schema and after.metadata == snapshot.metadata
    assert samples.__dict__["_storage_owner"] is owner and samples.plot.ax is ax


def test_metadata_labels_explicit_suppression_and_caller_annotations(samples):
    samples["run"] = "one"
    samples.plot.labels = ["X", "A {run}", "B", "Error", "Weight", "Unused"]
    ax = samples.plot.hist(y=1, label=None, title="Run {run}")
    assert ax.get_legend() is None and ax.get_xlabel() == "A one" and ax.get_title() == "Run one"
    ax.set(xlabel="Existing", ylabel="Existing Y", title="Existing title")
    samples.plot.kde(y=1, ax=ax)
    assert ax.get_xlabel() == "Existing" and ax.get_ylabel() == "Existing Y" and ax.get_title() == "Existing title"
    samples.plot.box(y=1, ax=ax, xlabel=None, ylabel=None, title=None, label=None)
    assert ax.get_xlabel() == ax.get_ylabel() == ax.get_title() == ""
    assert all(not label.get_text() for label in ax.get_xticklabels())


@pytest.mark.parametrize("kind,kwargs", [
    ("hist", {"y": 0, "no_guess": True}),  # Explicit samples override no_guess.
    ("hist", {"y": slice(1, 3)}),
])
def test_explicit_selectors_do_not_require_x_roles(kind, kwargs):
    data = Data(np.arange(15.).reshape(5, 3))
    assert isinstance(data.plot(kind=kind, **kwargs), Axes)


@pytest.mark.parametrize("kind,kwargs,match", [
    ("hist", {"no_guess": True}, "Select samples"),
    ("hist", {"x": 0}, "y="),
    ("hist", {"y": 0, "weights": [1]}, "shape"),
    ("box", {"y": 0, "weights": 1}, "weights"),
    ("kde", {"y": 0, "ind": 1}, "ind"),
    ("kde", {"y": 0, "ind": [0., np.inf]}, "ind"),
    ("kde", {"y": 0, "weights": np.zeros(4)}, "weights"),
    ("kde", {"y": 0, "weights": [1., -1., 1., 1.]}, "weights"),
    ("hexbin", {"y": [0, 1]}, "exactly one"),
    ("hexbin", {"C": [1]}, "shape"),
    ("hist", {"yerr": 1}, "uncertainty"),
    ("hist", {"y": [0, 1], "label": ["one"]}, "one entry"),
    ("hist", {"backend": "matplotlib"}, "Unsupported"),
    ("box", {"subplots": True}, "Unsupported"),
    ("kde", {"secondary_y": True}, "Unsupported"),
    ("hexbin", {"layout": (1, 2)}, "Unsupported"),
])
def test_invalid_scientific_inputs_fail_before_creating_axes(kind, kwargs, match):
    data = Data(np.arange(4.), np.arange(4.)+1)
    with pytest.raises((ValueError, TypeError), match=match):
        data.plot(kind=kind, **kwargs)
    assert not plt.get_fignums() and data.plot.ax is None


@pytest.mark.parametrize("kind", ["hist", "box", "kde", "hexbin"])
def test_empty_or_all_excluded_inputs_fail_without_axes(kind):
    data = Data(np.arange(4.), np.arange(4.))
    data.mask[:, 1] = True
    with pytest.raises(ValueError, match="No finite"):
        getattr(data.plot, kind)(y=1)
    assert not plt.get_fignums()


def test_kde_constant_or_single_effective_samples_fail_without_axes():
    for values in ([2., 2., 2.], [1.]):
        data = Data(np.asarray(values)[:, None])
        with pytest.raises(ValueError, match="two distinct"):
            data.plot.kde(y=0)
    assert not plt.get_fignums()


def test_startx_selects_the_assigned_y_group_without_reading_x():
    data = Data(np.arange(20.).reshape(4, 5), setas="xyxye")
    data.mask[:, 0] = data.mask[:, 2] = True
    ax = data.plot.hist(startx=2)
    assert ax.get_xlabel() == data.column_headers[3] and len(ax.containers) == 1


def test_y_roles_work_without_any_x_and_no_guess():
    data = Data(np.arange(8.).reshape(4, 2), setas="yy")
    assert len(data.plot.hist(no_guess=True).containers) == 2


def test_hist_density_and_cumulative_probability_match_numpy():
    data = Data(np.arange(5.)[:, None])
    ax = data.plot.hist(y=0, bins=np.arange(6.), density=True, cumulative=True)
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], np.arange(1, 6)/5)
    assert ax.get_ylabel() == "Cumulative probability"


def test_signed_histogram_weights_retain_native_meaning():
    data = Data(np.arange(3.), np.array([1., -2., 3.]))
    ax = data.plot.hist(y=0, weights=1, bins=[-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_equal([patch.get_height() for patch in ax.patches], [1., -2., 3.])


def test_invalid_second_kde_does_not_leave_first_curve_or_axes():
    data = Data(np.arange(4.), np.ones(4))
    with pytest.raises(ValueError, match="two distinct"):
        data.plot.kde(y=[0, 1])
    assert not plt.get_fignums()


@pytest.mark.parametrize("kind", ["hist", "box", "kde", "hexbin"])
def test_empty_tables_and_complex_samples_are_rejected(kind):
    with pytest.raises(ValueError, match="No finite"):
        Data(np.empty((0, 2))).plot(kind=kind, y=1)
    with pytest.raises(TypeError, match="real numerical"):
        Data(np.ones((3, 2), dtype=complex)).plot(kind=kind, y=1)
    assert not plt.get_fignums()
