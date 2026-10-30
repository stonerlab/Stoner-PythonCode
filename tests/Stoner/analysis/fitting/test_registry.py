"""Exercise registration ownership, naming and decorator compatibility."""

import importlib
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

from lmfit import Model
import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.fitting import models
from Stoner.analysis.fitting.models import make_model, model_registry, register_model
from Stoner.analysis.fitting.registry import ModelRegistry


@pytest.fixture(autouse=True)
def isolate_registrations(monkeypatch):
    """Keep custom registrations local to each test without losing built-ins."""
    monkeypatch.setattr(model_registry, "_entries", model_registry._entries.copy())


def test_builtin_catalogue_covers_defined_models_only():
    expected = {
        obj for module in (models.generic, models.thermal, models.magnetism,
                           models.tunnelling, models.e_transport, models.superconductivity)
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, Model) and obj.__module__ == module.__name__
    }
    assert set(model_registry.values()) == expected
    assert len(model_registry) == 28
    assert model_registry["bloch_grueneisen"] is models.e_transport.BlochGrueneisen
    assert model_registry["fuchs_sondheimer"] is models.e_transport.FluchsSondheimer
    assert model_registry["wl_fit"] is models.e_transport.WLfit
    assert "bloch_gruneisen" not in model_registry and "fluchs_sondheimer" not in model_registry
    assert Model not in model_registry.values()
    importlib.reload(models)  # Catalogue initialisation is idempotent.
    assert len(model_registry) == 28


@pytest.mark.parametrize("class_name, key", [
    ("PowerLaw", "power_law"), ("RSJ_Noiseless", "rsj_noiseless"),
    ("VFTEquation", "vft_equation"), ("Ic_B_Airy", "ic_b_airy"),
    ("NDimArrhenius", "n_dim_arrhenius"),
])
def test_class_names_are_normalised_without_construction(class_name, key):
    def forbidden_init(self):
        pytest.fail("Registration constructed a model")

    cls = type(class_name, (Model,), {"__init__": forbidden_init})
    registry = ModelRegistry()
    assert registry.register(cls) is cls
    assert list(registry) == [key] and registry[key] is cls


@pytest.mark.parametrize("name", ["", "_hidden", "Linear", "two__words", "not-valid", "café", "class",
                                  "lmfit", "curve_fit", "odr", "differential_evolution", "polyfit", "models"])
def test_invalid_and_reserved_names_do_not_mutate_registry(name):
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.register(models.generic.Linear, name=name)
    assert not registry


def test_read_only_live_mapping_collisions_and_provenance():
    registry = ModelRegistry()
    view = MappingProxyType(registry)
    registry.register(models.generic.Linear)
    assert view["linear"] is models.generic.Linear
    info = registry.registration("linear")
    assert info.module == models.generic.__name__ and info.display_name == "Linear"
    assert registry.register(models.generic.Linear) is models.generic.Linear
    with pytest.raises(ValueError, match="already registered.*Linear.*Quadratic"):
        registry.register(models.generic.Quadratic, name="linear")
    assert registry["linear"] is models.generic.Linear
    registry.register(models.generic.Quadratic, name="linear", replace=True)
    assert view["linear"] is models.generic.Quadratic
    with pytest.raises(TypeError):
        view["linear"] = Model
    with pytest.raises(TypeError):
        registry["linear"] = Model
    with pytest.raises(KeyError):
        registry["missing"]


@pytest.mark.parametrize("obj", [object(), 3, models.generic.Linear(), str])
def test_instances_and_unrelated_objects_are_rejected(obj):
    with pytest.raises(TypeError):
        ModelRegistry().register(obj, name="bad_model")


def test_factory_is_lazy_validated_and_called_afresh():
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return models.generic.Linear(**kwargs)

    registry = ModelRegistry()
    with pytest.raises(ValueError, match="explicit"):
        registry.register(factory)
    registry.register(factory, name="custom_linear")
    assert calls == []
    first = registry.create("custom_linear", prefix="first_")
    second = registry.create("custom_linear")
    assert first is not second and first.prefix == "first_" and second.prefix == ""
    assert calls == [{"prefix": "first_"}, {}]
    registry.register(lambda: 3, name="bad_factory")
    with pytest.raises(TypeError, match="expected lmfit.Model"):
        registry.create("bad_factory")


def test_shared_registration_decorator_forms():
    @register_model
    class TestRegisteredModel(models.generic.Linear):
        pass

    @register_model(name="custom_factory")
    def factory():
        return models.generic.Linear()

    assert model_registry["test_registered_model"] is TestRegisteredModel
    assert model_registry["custom_factory"] is factory
    assert register_model(TestRegisteredModel) is TestRegisteredModel


def test_make_model_registers_function_name_and_keeps_guesser_hinter_and_prefix():
    @make_model
    def TestResponse(x, slope, intercept):
        return slope * x + intercept

    assert model_registry["test_response"] is TestResponse
    info = model_registry.registration("test_response")
    assert info.display_name == "TestResponse" and info.module == __name__
    assert "TestResponse" in info.qualname and "_ModelDecorator" not in info.qualname

    @TestResponse.guesser
    def guess(y, x=None):
        return [2.0, 1.0]

    @TestResponse.hinter
    def hints():
        return {"slope": {"min": 0, "max": 5}, "intercept": {"min": -5, "max": 5}}

    first = model_registry.create("test_response")
    second = model_registry.create("test_response")
    first.set_param_hint("slope", max=4)
    assert first is not second and second.param_hints["slope"]["max"] == 5
    x = np.linspace(0, 2, 21)
    parameters = second.guess(2*x + 1, x=x)
    assert parameters["slope"].value == 2 and parameters["slope"].max == 5
    data = Data(x, 2*x + 1, setas="xy")
    report = data.lmfit(TestResponse, output="report")
    assert report.params["slope"].value == pytest.approx(2)
    assert data["_ModelDecorator:slope"] == pytest.approx(2)


def test_make_model_named_form_and_explicit_replacement():
    def response(x, slope):
        return x * slope

    first = make_model(response, name="named_response")
    assert model_registry["named_response"] is first
    with pytest.raises(ValueError, match="already registered"):
        make_model(response, name="named_response")
    second = make_model(name="named_response", replace=True)(response)
    assert model_registry["named_response"] is second and second is not first
    with pytest.raises(TypeError, match="callable"):
        make_model(42)


@pytest.mark.parametrize("first_import", [
    "from Stoner.enums import FitOutput",
    "from Stoner.analysis.fitting.models import make_model",
])
def test_cold_import_orders(first_import):
    code = first_import + "\n" + "\n".join([
        "from Stoner.analysis.fitting.registry import model_registry",
        "from Stoner.analysis.fitting.models import generic",
        "assert len(model_registry) == 28",
        "assert model_registry['linear'] is generic.Linear",
        "assert model_registry.create('linear') is not model_registry.create('linear')",
    ])
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[4],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
