#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""":py:class:`lmfit.Model` model classes and functions for various generic models."""
# pylint: disable=invalid-name
__all__ = [
    "Linear",
    "Lorentzian_diff",
    "Model",
    "PowerLaw",
    "Quadratic",
    "StretchedExp",
    "linear",
    "lorentzian_diff",
    "powerLaw",
    "quadratic",
    "stretchedExp",
]

import numpy as np

from lmfit import Model
from lmfit.models import LinearModel as _Linear  # NOQA pylint: disable=unused-import
from lmfit.models import PowerLawModel as _PowerLaw  # NOQA pylint: disable=unused-import
from lmfit.models import QuadraticModel as _Quadratic  # NOQA pylint: disable=unused-import
from lmfit.models import update_param_vals


def linear(x, intercept, slope):
    """Calculate a linear function."""
    return slope * x + intercept


def quadratic(x, a, b, c):
    r"""Calculate a simple quadratic fitting function.

    Args:
        x (array): Input data
        a (float): Quadratic term co-efficient
        b (float): Linear term co-efficient
        c (float): Constant offset term

    Returns:
        Array of data.

    :math:`y=ax^2+bx+c`

    Example:
        .. plot:: samples/Fitting/Quadratic.py
            :include-source:
            :outname: quadratic
    """
    return a * x**2 + b * x + c


def powerLaw(x, A, k):
    r"""Power Law Fitting Equation.

    Args:
        x (array): Input data
        A (float): Prefactor
        k (float): Power

    Return:
        Power law.

    :math:`p=Ax^k`

    Example:
        .. plot:: samples/Fitting/Powerlaw.py
            :include-source:
            :outname: powerlaw
    """
    return A * x**k


def stretchedExp(x, A, beta, x_0):
    r"""Calculate a stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\exp\left[\left(\frac{-x}{x_0}\right)^\beta\right]`.
    """
    return A * np.exp(-((x / x_0) ** beta))


def lorentzian_diff(x, A, sigma, mu):
    r"""Implement a differential form of a Lorentzian peak.

    Args:
        x (array): x data
        A (float): Peak amplitude
        sigma (float): peak wideth
        mu (float): peak location in x

    Returns
        :math:`\frac{A \sigma \left(2 \mu - 2 x\right)}{\pi \left(\sigma^{2} +
                                                                      \left(- \mu + x\right)^{2}\right)^{2}}`

    Example:
        .. plot:: samples/Fitting/lorentzian.py
            :include-source:
            :outname: lorentzian_diff_func
    """
    return A * sigma * (2 * mu - 2 * x) / (np.pi * (sigma**2 + (-mu + x) ** 2) ** 2)


class Linear(_Linear):
    """Simple linear fit class."""


class Quadratic(_Quadratic):
    r"""A Simple quadratic fitting function.

    Args:
        x (array): Input data
        a (float): Quadratic term co-efficient
        b (float): Linear term co-efficient
        c (float): Constant offset term

    Returns:
        Array of data.

    :math:`y=ax^2+bx+c`

    Example:
        .. plot:: samples/Fitting/Quadratic.py
            :include-source:
            :outname: quadratic-class
    """


class PowerLaw(_PowerLaw):
    r"""Power Law Fitting Equation.

    Args:
        x (array): Input data
        A (float): Prefactor
        k (float): Power

    Return:
        Power law.

    :math:`p=Ax^k`

    Example:
        .. plot:: samples/Fitting/Powerlaw.py
            :include-source:
            :outname: powerlaw-class
    """


class StretchedExp(Model):
    r"""A stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\exp\left[\left(\frac{-x}{x_0}\right)^\beta\right]`.
    """

    display_names = ["A", r"\beta", "x_0"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(stretchedExp, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters for the stretched exponential from data."""
        A, beta, x0 = 1.0, 1.0, 1.0
        if x is not None:
            A = data[np.argmin(np.abs(x))]
            x = np.log(x)
            y = np.log(-np.log(data / A))
            d = np.column_stack((x, y))
            d = d[~np.isnan(d).any(axis=1)]
            d = d[~np.isinf(d).any(axis=1)]
            d1, d2 = np.polyfit(d[:, 0], d[:, 1], 1)
            beta = d1
            x0 = np.exp(d2 / beta)
        pars = self.make_params(A=A, beta=beta, x_0=x0)
        return update_param_vals(pars, self.prefix, **kwargs)


class Lorentzian_diff(Model):
    r"""Provides a lmfit Model rerprenting the differential form of a Lorentzian Peak.

    Args:
        x (array): x data
        A (float): Peak amplitude
        sigma (float): peak wideth
        mu (float): peak location in x

    Returns
        :math:`\frac{A \sigma \left(2 \mu - 2 x\right)}{\pi \left(\sigma^{2} +
                                                                      \left(- \mu + x\right)^{2}\right)^{2}}`

    Example:
        .. plot:: samples/Fitting/lorentzian.py
            :include-source:
            :outname: lorentzian_diff_class
    """

    display_names = ["A", r"\sigma", r"\mu"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(lorentzian_diff, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H."""
        if x is None:
            x = np.linspace(1, len(data), len(data) + 1)

        x1 = x[np.argmax(data)]
        x2 = x[np.argmin(data)]
        sigma = abs(x1 - x2)
        mu = (x1 + x2) / 2.0
        y1 = np.max(data)
        y2 = np.min(data)
        dy = y1 - y2
        A = dy * (4 * np.pi * sigma**2) / (3 * np.sqrt(3))

        pars = self.make_params(A=A, sigma=sigma, mu=mu)
        pars["A"].min = 0
        pars["sigma"].min = 0
        pars["mu"].min = np.min(x)
        pars["mu"].max = np.max(x)
        return update_param_vals(pars, self.prefix, **kwargs)
