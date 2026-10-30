"""Keep internal scientific fitting calls on the accessor migration boundary."""

from pathlib import Path

import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.utils import hysteresis_correct
from Stoner.analysis.fitting.models.generic import Linear


def reject_legacy(*args, **kwargs):
    """Detect an internal call that bypasses the fitting accessor."""
    pytest.fail("Internal scientific consumer used a legacy direct fitting method")


def test_extrapolation_uses_accessor_and_preserves_linear_prediction(monkeypatch):
    monkeypatch.setattr(Data, "curve_fit", reject_legacy)
    x = np.linspace(0, 3, 31)
    data = Data(x, 2*x + 1, setas="xy")
    result = data.extrapolate([-0.5, 3.5], kind="linear", overlap=10)
    np.testing.assert_allclose(result[:, 0], [0, 8], atol=1e-8)
    assert np.all(result[:, 1] >= 0)


def test_hysteresis_helpers_use_accessor_with_real_instrument_fixture(monkeypatch):
    monkeypatch.setattr(Data, "curve_fit", reject_legacy)
    monkeypatch.setattr(Data, "lmfit", reject_legacy)
    fixture = Path(__file__).resolve().parents[3] / "sample-data" / "QD-SQUID-VSM.dat"
    result = hysteresis_correct(str(fixture), setas="3.xy", h_sat_method="linear_intercept",
                                saturated_fraction=0.25)
    assert result["Hc"][0] == pytest.approx(-578, abs=1)
    assert -0.0137 < result["Area"] < -0.0136


@pytest.mark.parametrize("selector", [1, np.int64(1)])
def test_fitting_preparation_accepts_scalar_response_selectors(selector):
    x = np.linspace(0, 3, 21)
    data = Data(x, 2*x + 1, setas="xy")
    report = data.fit.lmfit(Linear, x=0, y=selector, p0=[1.5, 0.8], output="report")
    assert report.params["slope"].value == pytest.approx(2)
