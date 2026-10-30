"""Exercise member-bound fitting, worker transport and bulk return contracts."""

import inspect
import os

import multiprocess
import numpy as np
import pytest

from Stoner import Data, DataFolder, Options
from Stoner.analysis.fitting.models import make_model, model_registry, register_model
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.enums import FitOutput
from Stoner.folders.each import FitProxy, _worker
from Stoner.folders import utils


@pytest.fixture(autouse=True)
def isolate_options_and_registry(monkeypatch):
    monkeypatch.setattr(Options, "multiprocessing", False)
    monkeypatch.setattr(Options, "threading", False)
    monkeypatch.setattr(model_registry, "_entries", model_registry._entries.copy())


def make_folder(count=3):
    folder = DataFolder(readlist=False)
    x = np.linspace(0, 3, 21)
    for index in range(count):
        data = Data(x, (index+2)*x + index-1, column_headers=["Field", "Signal"],
                    setas="xy", filename=f"trace-{index}")
        data["member"] = index
        folder += data
    return folder


def configure_pool(monkeypatch, mode):
    monkeypatch.setattr(Options, "multiprocessing", True)
    monkeypatch.setattr(Options, "threading", mode == "thread")
    monkeypatch.setattr(utils.multiprocessing, "cpu_count", lambda: 4)
    if mode == "spawn":
        monkeypatch.setattr(utils.multiprocessing, "Pool", multiprocess.get_context("spawn").Pool)


def test_lookup_completion_and_signatures_do_not_load_members(monkeypatch):
    folder = make_folder()
    monkeypatch.setattr(folder, "fetch", lambda: pytest.fail("Lookup fetched members"))
    proxy = folder.each.fit
    assert isinstance(proxy, FitProxy)
    assert {"linear", "lmfit", "curve_fit", "models"} <= set(dir(proxy))
    assert proxy.models["linear"] is Linear
    assert "Fit" in inspect.getdoc(proxy.linear)
    assert list(inspect.signature(proxy.linear).parameters) == ["x", "y", "p0", "sigma", "kwargs"]
    assert list(inspect.signature(proxy.lmfit).parameters) == ["model", "x", "y", "p0", "sigma", "kwargs"]
    with pytest.raises(TypeError):
        proxy.models["linear"] = object
    for name in ("missing", "_hidden", "models.create"):
        with pytest.raises(AttributeError):
            getattr(proxy, name)


@pytest.mark.parametrize("style", ["default", "explicit", "shortcut", "string", "iterator"])
def test_calls_preserve_order_member_updates_and_controls(style):
    folder = make_folder()
    options = dict(x="Field", y="Signal", result=True, header="fit", output=FitOutput.ROW, _return=True)
    if style == "default":
        results = folder.each.fit(Linear, **options)
    elif style == "explicit":
        results = folder.each.fit.lmfit(Linear, **options)
    elif style == "shortcut":
        results = folder.each.fit.linear(**options)
    elif style == "string":
        results = folder.each("fit.linear", **options)
    else:
        results = list(folder.each.iter("fit.linear", **options))
    name = "fit" if style == "default" else "lmfit" if style == "explicit" else "linear"
    assert len(results) == 3
    for index, (result, member) in enumerate(zip(results, folder)):
        np.testing.assert_allclose(result[:3:2], [index+2, index-1], atol=1e-6)
        assert member["member"] == index and member.shape[1] == 3
        assert member.column_headers[-1] == "fit"
        np.testing.assert_array_equal(member[name], result)
        assert member["func"] == name
        assert member["setas"] == ["x", "y"]
        if style == "default":
            assert member["args"][1:3] == ("Field", "Signal")
        else:
            assert member["kwargs"]["x"] == "Field"


@pytest.mark.parametrize("algorithm", ["lmfit", "curve_fit", "odr", "differential_evolution", "polyfit"])
def test_explicit_algorithm_results(algorithm):
    folder = make_folder(1)
    if algorithm == "polyfit":
        results = folder.each.fit.polyfit(x=0, y=1, polynomial_order=1)
        np.testing.assert_allclose(results[0], [2, -1], atol=1e-6)
    else:
        results = getattr(folder.each.fit, algorithm)(Linear, x=0, y=1, p0=[1.8, -0.8], output="fit")
        assert isinstance(results[0], tuple) and not isinstance(results[0][0], Exception)
        # Existing non-curve_fit algorithms retain a trailing fit statistic here.
        np.testing.assert_allclose(results[0][0][:2], [2, -1], atol=1e-5)


@pytest.mark.parametrize("output", list(FitOutput))
def test_output_modes_and_named_return_metadata(output):
    folder = make_folder(1)
    results = folder.each.fit.linear(output=output, _return="saved")
    assert len(results) == 1
    assert "saved" in folder[0].metadata
    if output is FitOutput.DATA:
        assert results[0] is folder[0]
    elif output is FitOutput.REPORT:
        assert results[0].params["slope"].value == pytest.approx(2)


def test_data_return_replaces_members_and_empty_calls_work():
    folder = make_folder(2)
    results = folder.each.fit.linear(output=FitOutput.DATA)
    assert all(member is result for member, result in zip(folder, results))
    assert make_folder(0).each.fit.linear() == []


def test_worker_exceptions_are_collected_and_later_members_continue():
    folder = make_folder(3)
    folder[1].data = np.zeros((1, 2))
    results = folder.each.fit.curve_fit(Linear, output=FitOutput.FIT, _return="saved")
    assert len(results) == 3 and isinstance(results[1][0], Exception)
    assert "Traceback" in results[1][1]
    assert not isinstance(results[0][0], Exception) and not isinstance(results[2][0], Exception)
    assert isinstance(folder[1]["saved"][0], Exception)


def test_pool_policy_and_explicit_serial_override(monkeypatch):
    calls = []
    monkeypatch.setattr("Stoner.folders.each.get_pool", lambda serial: (calls.append(serial) or None, map))
    make_folder(1).each.fit.linear()
    make_folder(1).each.fit.linear(_serial=True)
    assert calls == [False, True]


@pytest.mark.parametrize("mode", ["thread", "spawn"])
def test_real_workers_support_custom_factory_without_global_registration(monkeypatch, mode):
    configure_pool(monkeypatch, mode)
    name = "bulk_factory_line"

    @register_model(name=name)
    def factory():
        from Stoner.analysis.fitting.models import model_registry as worker_registry

        model = Linear()
        model.creator_pid = os.getpid()
        model.registered_globally = name in worker_registry
        return model

    folder = make_folder(3)
    results = folder.each.fit.bulk_factory_line(output=FitOutput.REPORT, result=True)
    for index, (result, member) in enumerate(zip(results, folder)):
        assert not isinstance(result, tuple), result
        assert result.params["slope"].value == pytest.approx(index+2)
        assert member.shape[1] == 3 and member["member"] == index
        assert result.model.registered_globally is (mode == "thread")
        assert (result.model.creator_pid == os.getpid()) is (mode == "thread")
    assert model_registry[name] is factory


def test_spawn_workers_receive_decorated_replacement_of_builtin_name(monkeypatch):
    configure_pool(monkeypatch, "spawn")

    @make_model(name="linear", replace=True)
    def replacement(x, gradient, offset):
        return gradient*x + offset

    folder = make_folder(2)
    results = folder.each.fit.linear(p0=[2, 0], output=FitOutput.ROW)
    for index, (result, member) in enumerate(zip(results, folder)):
        assert isinstance(result, np.ndarray), result
        assert member["_ModelDecorator:gradient"] == pytest.approx(index+2)
    assert model_registry["linear"] is replacement


def test_spawn_explicit_algorithm_data_results_and_named_metadata(monkeypatch):
    configure_pool(monkeypatch, "spawn")
    folder = make_folder(2)
    results = folder.each.fit.lmfit(Linear, x="Field", y="Signal", output=FitOutput.DATA,
                                   result=True, _return=None)
    assert all(result is member for result, member in zip(results, folder))
    assert all(member.shape[1] == 3 for member in folder)


def test_worker_uses_payload_without_installing_it_in_registry():
    @register_model(name="temporary_line")
    class TemporaryLine(Linear):
        pass

    registration = model_registry.registration("temporary_line")
    del model_registry._entries["temporary_line"]
    owner, result = _worker(make_folder(1)[0], byname=True, func="fit.temporary_line",
                            fit_registration=registration, kwargs={"output": FitOutput.ROW})
    assert isinstance(result, np.ndarray), result
    assert owner["func"] == "temporary_line" and "temporary_line" not in model_registry


def test_saved_bulk_shortcut_resolves_new_registration():
    folder = make_folder(1)
    saved = folder.each.fit.linear

    @make_model(name="linear", replace=True)
    def replacement(x, gradient, offset):
        return gradient*x + offset

    assert isinstance(saved(p0=[2, 0], output=FitOutput.ROW)[0], np.ndarray)
    assert "_ModelDecorator:gradient" in folder[0].metadata


def test_failed_serialisation_terminates_and_joins_owned_pool(monkeypatch):
    class UnserialisableFactory:
        def __call__(self):
            return Linear()

        def __reduce__(self):
            raise RuntimeError("Cannot serialise this factory")

    class FakeProcessPool:
        def __init__(self):
            self.events = []

        def terminate(self):
            self.events.append("terminate")

        def join(self):
            self.events.append("join")

    pool = FakeProcessPool()
    register_model(UnserialisableFactory(), name="unserialisable_factory")
    monkeypatch.setattr("Stoner.folders.each.ProcessPool", FakeProcessPool)
    monkeypatch.setattr("Stoner.folders.each.get_pool", lambda serial: (pool, map))
    with pytest.raises(TypeError, match="Cannot transfer.*_serial=True") as caught:
        make_folder(1).each.fit.unserialisable_factory()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert pool.events == ["terminate", "join"]


def test_closing_bulk_iterator_terminates_and_joins_owned_pool(monkeypatch):
    class FakePool:
        def __init__(self):
            self.events = []

        def terminate(self):
            self.events.append("terminate")

        def join(self):
            self.events.append("join")

    pool = FakePool()
    monkeypatch.setattr("Stoner.folders.each.get_pool", lambda serial: (pool, map))
    iterator = make_folder(3).each.iter("fit.linear", output=FitOutput.ROW)
    assert isinstance(next(iterator), np.ndarray)
    iterator.close()
    assert pool.events == ["terminate", "join"]
