"""Exercise fitting delegation, shortcut discovery and live owner boundaries."""

import copy
import inspect
import pickle
from enum import StrEnum

import matplotlib.pyplot as plt
from lmfit import Model
import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.fitting import DataFitter, functions
from Stoner.analysis.fitting.models import make_model, model_registry, register_model
from Stoner.analysis.fitting.models.generic import Linear, Quadratic
from Stoner.enums import FitOutput


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch):
    monkeypatch.setattr(model_registry, "_entries", model_registry._entries.copy())
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def data():
    x = np.linspace(0, 3, 25)
    return Data(x, 2*x + 1 + 0.01*np.sin(5*x), column_headers=["Field", "Signal"], setas="xy")


def test_lookup_completion_and_models_do_not_read_or_fit(data, monkeypatch):
    before = data.__dict__.copy()
    monkeypatch.setattr(data, "column", lambda *a, **kw: pytest.fail("Lookup read data"))
    monkeypatch.setattr(functions, "lmfit", lambda *a, **kw: pytest.fail("Lookup fitted data"))
    monkeypatch.setattr(Linear, "__init__", lambda *a, **kw: pytest.fail("Lookup constructed a model"))
    fitter = data.fit
    assert isinstance(fitter, DataFitter) and fitter is not data.fit
    shortcut = fitter.linear
    assert callable(shortcut) and shortcut.__name__ == "linear"
    assert "Linear" in inspect.getdoc(shortcut)
    assert {"linear", "bloch_grueneisen", "lmfit", "models"} <= set(dir(fitter))
    assert fitter.models["linear"] is Linear
    with pytest.raises(TypeError):
        fitter.models["linear"] = Quadratic
    assert not hasattr(fitter.models, "register")
    assert plt.get_fignums() == []
    assert set(data.__dict__) == set(before) | {"column"}


def test_lookup_errors_suggestions_and_live_registration(data):
    with pytest.raises(AttributeError, match="linear"):
        data.fit.linera
    with pytest.raises(AttributeError):
        data.fit._missing
    with pytest.raises(KeyError):
        data.fit.models["missing"]
    view = data.fit.models
    register_model(Linear, name="another_line")
    assert view["another_line"] is Linear and "another_line" in dir(data.fit)


def test_explicit_and_shortcut_signatures(data):
    assert list(inspect.signature(data.fit).parameters) == ["model", "x", "y", "p0", "sigma", "kwargs"]
    assert list(inspect.signature(data.fit.linear).parameters) == ["x", "y", "p0", "sigma", "kwargs"]
    assert list(inspect.signature(data.fit.curve_fit).parameters) == ["func", "x", "y", "sigma", "kwargs"]
    assert "xcol" not in inspect.signature(data.fit.lmfit).parameters


@pytest.mark.parametrize("algorithm", ["lmfit", "curve_fit", "odr", "differential_evolution"])
def test_explicit_methods_translate_coordinates_without_converting_outputs(data, monkeypatch, algorithm):
    sentinel = object()
    calls = []

    def capture(owner, model, **kwargs):
        calls.append((owner, model, kwargs))
        return sentinel

    monkeypatch.setattr(functions, algorithm, capture)
    options = {"p0": [1, 0], "sigma": None, "output": FitOutput.REPORT, "prefix": "test"}
    assert getattr(data.fit, algorithm)(Linear, "Field", "Signal", **options) is sentinel
    owner, model, forwarded = calls[0]
    assert owner is data and model is Linear
    assert forwarded == dict(options, xcol="Field", ycol="Signal")


def test_call_and_shortcuts_use_overridden_accessor_lmfit(data):
    class CustomFitter(DataFitter):
        def lmfit(self, model, *args, **kwargs):
            return self._data, model, args, kwargs

    fitter = CustomFitter(data)
    assert fitter(Linear)[0:2] == (data, Linear)
    assert fitter.linear()[0:2] == (data, Linear)


@pytest.mark.parametrize("method", ["lmfit", "curve_fit", "odr", "differential_evolution", "linear", "__call__"])
@pytest.mark.parametrize("legacy", [{"xcol": 0}, {"ycol": 1}, {"x": 0, "xcol": 0}])
def test_old_coordinate_keywords_fail_before_fit(data, monkeypatch, method, legacy):
    monkeypatch.setattr(functions, "lmfit", lambda *a, **kw: pytest.fail("Invalid keyword reached fitting"))
    args = () if method == "linear" else (Linear,)
    with pytest.raises(TypeError, match="Data.fit uses x and y"):
        getattr(data.fit, method)(*args, **legacy)


def test_polyfit_keeps_positional_contract_and_rejects_unsupported_options(data):
    expected = data.clone.polyfit(0, 1, polynomial_order=1)
    actual = data.fit.polyfit(0, 1, 1)
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(TypeError):
        data.fit.polyfit(xcol=0)
    with pytest.raises(TypeError):
        data.fit.polyfit(output=FitOutput.REPORT)


@pytest.mark.parametrize("algorithm", ["lmfit", "curve_fit", "odr", "differential_evolution"])
def test_numerical_algorithms_match_direct_methods(data, algorithm):
    expected_data = data.clone
    state = np.random.get_state()
    try:
        np.random.seed(893)
        expected = getattr(expected_data, algorithm)(
            Linear, xcol="Field", ycol="Signal", p0=[1.5, 0.8], result=True, output=FitOutput.FIT,
        )
        np.random.seed(893)
        actual = getattr(data.fit, algorithm)(
            Linear, x="Field", y="Signal", p0=[1.5, 0.8], result=True, output=FitOutput.FIT,
        )
    finally:
        np.random.set_state(state)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value)
    np.testing.assert_allclose(data.to_numpy(), expected_data.to_numpy())
    assert data.column_headers == expected_data.column_headers
    assert list(data.setas) == list(expected_data.setas)
    assert dict(data.metadata) == dict(expected_data.metadata)


@pytest.mark.parametrize("output", list(FitOutput))
def test_shortcut_and_callable_keep_output_contract(data, output):
    first, second = data.clone, data.clone
    expected = data.fit.lmfit(Linear, output=output)
    actual = first.fit.linear(output=output)
    called = second.fit(Linear, output=output)
    assert type(actual) is type(called) is type(expected)
    if output is FitOutput.DATA:
        assert expected is data and actual is first and called is second
    elif output is FitOutput.FIT:
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(called[0], expected[0])
    elif output is FitOutput.ROW:
        np.testing.assert_allclose(actual, expected)
    elif output is FitOutput.FULL:
        np.testing.assert_allclose(actual[1], expected[1])
    elif output is FitOutput.DICT:
        assert actual == expected == called


def test_saved_shortcut_reads_current_owner_and_registration(data, monkeypatch):
    shortcut = data.fit.linear
    data[:, 1] = 4 * data.column(0) + 3
    report = shortcut(output="report")
    assert report.params["slope"].value == pytest.approx(4)
    register_model(Quadratic, name="linear", replace=True)
    sentinel = object()
    monkeypatch.setattr(functions, "lmfit", lambda owner, model, **kw: sentinel if model is Quadratic else None)
    assert shortcut() is sentinel
    del model_registry._entries["linear"]
    with pytest.raises(KeyError):
        shortcut()


def test_factory_shortcut_is_lazy_and_creates_independent_models(data, monkeypatch):
    created = []

    @register_model(name="factory_line")
    def factory():
        model = Linear()
        created.append(model)
        return model

    shortcut = data.fit.factory_line
    assert not created
    shortcut(output="report")
    shortcut(output="report")
    assert len(created) == 2 and created[0] is not created[1]
    monkeypatch.setattr(functions, "lmfit", lambda *a, **kw: pytest.fail("Bad factory reached fitting"))
    register_model(lambda: 3, name="bad_factory")
    with pytest.raises(TypeError, match="expected lmfit.Model"):
        data.fit.bad_factory()


def test_make_model_shortcut_and_configured_composite_model(data):
    @make_model
    def accessor_response(x, slope, intercept):
        return slope*x + intercept

    result = data.fit.accessor_response(p0=[2, 1], output="report")
    assert result.params["slope"].value == pytest.approx(2, abs=0.01)
    model = Linear(prefix="base_") + Model(lambda x, curvature: curvature*x*x, prefix="curve_")
    parameters = model.make_params(base_slope=2, base_intercept=1, curve_curvature=0)
    parameters["base_slope"].set(min=0, max=5)
    report = data.fit(model, p0=parameters, output=FitOutput.REPORT)
    assert report.params["base_slope"].min == 0 and report.params["base_slope"].max == 5


@pytest.mark.parametrize("copy_operation", [copy.copy, copy.deepcopy, lambda d: d.clone,
                                           lambda d: pickle.loads(pickle.dumps(d))])
def test_data_copies_do_not_capture_accessor_or_custom_shortcut(data, copy_operation):
    @make_model
    def local_response(x, slope):
        return slope*x

    saved = data.fit.local_response
    saved(p0=[2], output="report")
    copied = copy_operation(data)
    assert not any(isinstance(value, DataFitter) for value in copied.__dict__.values())
    assert copied.fit._data is copied
    copied[:, 1] = 5*copied.column(0)
    assert copied.fit.local_response(p0=[2], output="report").params["slope"].value == pytest.approx(5)
    assert not np.array_equal(data.column(1), copied.column(1))


def test_fit_column_remains_explicitly_accessible_and_old_methods_still_work(data):
    data.rename("Signal", "fit")
    assert isinstance(data.fit, DataFitter)
    np.testing.assert_array_equal(data.column("fit"), data.column(1))
    assert data.fit.linear(x="Field", y="fit", output="data") is data
    assert Data.lmfit(data, Linear, xcol="Field", ycol="fit", output="data") is data


def test_masked_duplicate_channels_match_direct_curve_fit():
    x = np.linspace(0, 3, 25)
    data = Data(x, 2*x + 1, 3*x - 1, column_headers=["Field", "Signal", "Signal"], setas="xyy")
    data.mask[2, 1] = True
    data.mask[3, 2] = True
    before = data.to_numpy().copy()
    actual = data.fit.curve_fit(Linear, x="Field", y="Signal", output="report")
    expected = data.curve_fit(Linear, xcol="Field", ycol="Signal", output="report")
    assert len(actual) == len(expected) == 2
    for first, second in zip(actual, expected):
        np.testing.assert_allclose(first.popt, second.popt)
        assert first.n_observations == second.n_observations == 24
    np.testing.assert_array_equal(data.to_numpy().mask, before.mask)


def test_shortcut_rejects_wrong_domain_output(data):
    class OtherOutput(StrEnum):
        REPORT = "report"

    with pytest.raises(TypeError, match="FitOutput"):
        data.fit.linear(output=OtherOutput.REPORT)
