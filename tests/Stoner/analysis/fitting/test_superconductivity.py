"""Regression checks for superconducting fitting models."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import j1

from Stoner.analysis.fitting.models.superconductivity import Ic_B_Airy, Phi_0, ic_B_airy


@pytest.mark.parametrize("argument", [0.0, 5e-6])
def test_airy_central_scalar(argument):
    """Return the limiting peak current for scalar fields at and near the offset."""
    field = 0.03 + argument * Phi_0 / (np.pi * 1e-12)
    with np.errstate(divide="raise", invalid="raise"):
        result = ic_B_airy(field, 2.5, 0.03, 1e-12)
    assert np.ndim(result) == 0
    assert result == 2.5


def test_airy_small_arguments_and_boundary():
    """Match the analytic limit across both signs and the approximation boundary."""
    argument = np.array([[-2e-5, -1.00001e-5, -5e-6], [0.0, 1e-5, 2e-5]])
    field = argument * Phi_0 / (np.pi * 1e-12)
    with np.errstate(divide="raise", invalid="raise"):
        result = ic_B_airy(field, 2.5, 0.0, 1e-12)
    expected = 2.5 * (1 - argument**2 / 8 + argument**4 / 192)
    assert result.shape == field.shape
    assert_allclose(result, expected, rtol=2e-11, atol=0)


@pytest.mark.parametrize("area", [1e-12, 3e-12])
def test_airy_model_nonzero_field(area):
    """Preserve the Bessel expression away from the central approximation."""
    argument = np.array([-5.0, -2.0, 0.2, 2.0, 5.0])
    field = argument * Phi_0 / (np.pi * area)
    expected = 2.5 * np.abs(2 * j1(argument) / argument)
    model = Ic_B_Airy()
    with np.errstate(divide="raise", invalid="raise"):
        result = model.eval(B=field, Ic0=2.5, B_offset=0.0, A=area)
    assert_allclose(result, expected, rtol=1e-13, atol=0)
