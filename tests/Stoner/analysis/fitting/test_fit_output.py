"""Verify enum/string parity without changing the existing fit outputs."""

from enum import StrEnum
import json
import pickle

import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.fitting import functions
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.enums import FitOutput


def measurements():
    """Return deterministic slightly noisy data with explicit column roles."""
    x = np.linspace(0, 3, 25)
    return Data(x, 2*x + 1 + 0.01*np.sin(5*x), setas="xy")


@pytest.mark.parametrize("algorithm", ["curve_fit", "lmfit", "odr", "differential_evolution"])
@pytest.mark.parametrize("output", list(FitOutput))
def test_all_standard_outputs_match_strings(algorithm, output):
    enum_data, string_data = measurements(), measurements()
    state = np.random.get_state()
    try:
        np.random.seed(712)
        enum_result = getattr(functions, algorithm)(
            enum_data, Linear, p0=[1.5, 0.8], prefix="parity", result=True, output=output,
        )
        np.random.seed(712)
        string_result = getattr(string_data, algorithm)(
            Linear, p0=[1.5, 0.8], prefix="parity", result=True, output=output.value,
        )
    finally:
        np.random.set_state(state)
    assert type(enum_result) is type(string_result)
    assert enum_data["parity:slope"] == pytest.approx(string_data["parity:slope"])
    np.testing.assert_array_equal(enum_data.to_numpy(), string_data.to_numpy())
    if output is FitOutput.DATA:
        assert enum_result is enum_data and string_result is string_data
    elif output is FitOutput.FIT:
        for enum_value, string_value in zip(enum_result, string_result):
            np.testing.assert_allclose(enum_value, string_value)
    elif output is FitOutput.ROW:
        np.testing.assert_allclose(enum_result, string_result)
    elif output is FitOutput.FULL:
        np.testing.assert_allclose(enum_result[1], string_result[1])
    elif output is FitOutput.DICT:
        assert enum_result.keys() == string_result.keys()
        for key in enum_result:
            np.testing.assert_allclose(enum_result[key], string_result[key])


def test_enum_values_and_serialisation_remain_legacy_strings():
    assert {member.value for member in FitOutput} == {"fit", "report", "row", "full", "data", "dict"}
    assert str(FitOutput.REPORT) == "report" and FitOutput.REPORT == "report"
    assert json.loads(json.dumps({"output": FitOutput.REPORT})) == {"output": "report"}
    assert pickle.loads(pickle.dumps(FitOutput.REPORT)) is FitOutput.REPORT


@pytest.mark.parametrize("algorithm", ["curve_fit", "lmfit", "odr", "differential_evolution"])
def test_wrong_domain_enum_fails_before_mutating_measurements(algorithm):
    class OtherOutput(StrEnum):
        REPORT = "report"

    data = measurements()
    before = data.clone
    with pytest.raises(TypeError, match="FitOutput"):
        getattr(data, algorithm)(Linear, output=OtherOutput.REPORT)
    np.testing.assert_array_equal(data.to_numpy(), before.to_numpy())
    assert dict(data.metadata) == dict(before.metadata)


@pytest.mark.parametrize("output", [FitOutput.ROW, FitOutput.DATA, FitOutput.REPORT])
def test_lmfit_repeated_initial_guesses_keep_existing_outputs(output):
    data, other = measurements(), measurements()
    # The existing chi-squared table path uses metadata supplied by INI setup.
    for measurement in (data, other):
        measurement["lmfit.prefix"] = ["Linear:"]
        measurement["Linear:slope vary"] = True
        measurement["Linear:intercept vary"] = True
    initial = np.array([[1.0, 0.5], [2.5, 1.5]])
    enum_result = data.lmfit(Linear, p0=initial, output=output)
    string_result = other.lmfit(Linear, p0=initial, output=output.value)
    assert type(enum_result) is type(string_result)
    if isinstance(enum_result, Data):
        enum_result, string_result = enum_result.to_numpy(), string_result.to_numpy()
    np.testing.assert_allclose(enum_result, string_result)


def test_curve_fit_legacy_attribute_output_and_defaults_remain_available():
    data = measurements()
    parameters = data.curve_fit(Linear, output="popt")
    np.testing.assert_allclose(parameters, measurements().curve_fit(Linear)[0])
    assert isinstance(measurements().lmfit(Linear, asrow=True), np.ndarray)
