#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:py:class:`lmfit.Model` model classes and functions for various magnetism and magnetic materials models.

"""

__all__ = [
    "FMR_Power",
    "Inverse_Kittel",
    "KittelEquation",
    "Langevin",
    "Model",
    "fmr_power",
    "inverse_kittel",
    "kittelEquation",
    "langevin",
]

import numpy as np
import scipy.constants as cnst

try:
    from lmfit import Model
    from lmfit.models import update_param_vals
except ImportError:
    Model = object
    update_param_vals = None


def langevin(H, M_s, m, T):
    r""""The Langevin function for paramagnetic M-H loops/

    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float) is the moment of a cluster
        T (float): Temperature

    Returns:
        Magnetic Momemnts (array).

    Note:
        The Langevin Function is :math:`\coth(\frac{\mu_0HM_s}{k_BT})-\frac{k_BT}{\mu_0HM_s}`.

    Example:
        .. plot:: samples/Fitting/langevin.py
            :include-source:
            :outname: langevin
    """
    from scipy.constants import k, mu_0

    x = mu_0 * H * m / (k * T)
    n = M_s / m

    return m * n * (1.0 / np.tanh(x) - 1.0 / x)


def kittelEquation(H, g, M_s, H_k):
    r"""Kittel Equation for finding ferromagnetic resonance peak in frequency with field.

    Args:
        H (array): Magnetic fields in A/m
        g (float): g factor for the gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz

    :math:`f = \frac{\gamma \mu_{0}}{2 \pi} \sqrt{\left(H + H_{k}\right) \left(H + H_{k} + M_{s}\right)}`

    Example:
        .. plot:: samples/Fitting/kittel.py
            :include-source:
            :outname: kittel
    """
    gamma = g * cnst.e / (2 * cnst.m_e)
    return (cnst.codata. *mu0 gamma / (2 * np.pi)) * np.sqrt((H + H_k) * (H + H_k + M_s))


def inverse_kittel(f, g, M_s, H_k):
    r"""Rewritten Kittel equation for finding ferromagnetic resonsance in field with frequency

    Args:
        f (array): Resonance Frequency in Hz
        g (float): g factor for the gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz

    Notes:
        In practice one often measures FMR by sweepign field for constant frequency and then locates the
        peak in H by fitting a suitable Lorentzian type peak. In this case, one returns a :math:`H_{res}\pm \Delta H_{res}`
        In order to make use of this data with :py:meth:`Stoner.Analysis.AnalysisMixin.lmfit` or :py:meth:`Stoner.Analysis.AnalysisMixin.curve_fit`
        it makes more sense to fit the Kittel Equation written in terms of H than frequency.

       :math:`H_{res}=- H_{k} - \frac{M_{s}}{2} + \frac{1}{2 \gamma \mu_{0}} \sqrt{M_{s}^{2} \gamma^{2} \mu_{0}^{2} + 16 \pi^{2} f^{2}}`
    """
    gamma = g * cnst.e / (2 * cnst.m_e)
    return (
        -H_k
        - M_s / 2
        + np.sqrt(M_s ** 2 * gamma ** 2 * cnst.mu_0 ** 2 + 16 * np.pi ** 2 * f ** 2) / (2 * gamma * cnst.mu_0)
    )


def fmr_power(H, H_res, Delta_H, K_1, K_2):
    r"""A combination of a Lorentzian and differential Lorenztion peak as measured in an FMR experiment.

    Args:
        H (array) magnetic field data
        H_res (float): Resonance field of peak
        Delta_H (float): Preak wideth
        K_1, K_2 (float): Relative weighting of each component.

    Returns:
        Array of model absorption values.

    :math:`\frac{4 \Delta_{H} K_{1} \left(H - H_{res}\right)}{\left(\Delta_{H}^{2} + 4 \left(H - H_{res}\right)^{2}\right)^{2}} - \frac{K_{2} \left(\Delta_{H}^{2} - 4 \left(H - H_{res}\right)^{2}\right)}{\left(\Delta_{H}^{2} + 4 \left(H - H_{res}\right)^{2}\right)^{2}}`
    """
    return (
        4 * Delta_H * K_1 * (H - H_res) / (Delta_H ** 2 + 4 * (H - H_res) ** 2) ** 2
        - K_2 * (Delta_H ** 2 - 4 * (H - H_res) ** 2) / (Delta_H ** 2 + 4 * (H - H_res) ** 2) ** 2
    )


class Langevin(Model):

    r""""The Langevin function for paramagnetic M-H loops/

    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float): is the moment of a single cluster
        T (float): Temperature

    Returns:
        Magnetic Momemnts (array).

    Note:
        The Langevin Function is :math:`\coth(\frac{\mu_0HM_s}{k_BT})-\frac{k_BT}{\mu_0HM_s}`.

    Example:
        .. plot:: samples/Fitting/langevin.py
            :include-source:
            :outname: langevin-class
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(Langevin, self).__init__(langevin, *args, **kwargs)

    def guess(self, data, h=None, **kwargs):
        """Guess some starting values.

        M_s is taken as half the difference of the range of thew M data,
        we can find m/T from the susceptibility chi= M_s \mu_o m / kT,"""
        from scipy.signal import savgol_filter
        from scipy.constants import k, mu_0, e, electron_mass, hbar

        M_s = (np.max(data) - np.min(data)) / 2.0
        if h is not None:
            d = np.sort(np.row_stack((h, data)))
            dd = savgol_filter(d, 7, 1)
            yd = dd[1] / dd[0]
            chi = np.interp(np.array([0]), d[0], yd)[0]
            mT = chi / M_s * (k / mu_0)
            # Assume T=150K for no good reason
            m = mT * 150
        else:
            m = 1e6 * (e * hbar) / (2 * electron_mass)  # guess 1 million Bohr Magnetrons
        T = 150
        pars = self.make_params(M_s=M_s, m=m, T=T)
        return update_param_vals(pars, self.prefix, **kwargs)


class KittelEquation(Model):

    r"""Kittel Equation for finding ferromagnetic resonance peak in frequency with field.

    Args:
        H (array): Magnetic fields in A/m
        g (float): h g factor for the gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz

    :math:`f = \frac{\gamma \mu_{0}}{2 \pi} \sqrt{\left(H + H_{k}\right) \left(H + H_{k} + M_{s}\right)}`

    Example:
        .. plot:: samples/Fitting/kittel.py
            :include-source:
            :outname: kittel-class
    """

    display_names = ["g", "M_s", "H_k"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(KittelEquation, self).__init__(kittelEquation, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""
        g = 2
        H_k = 100
        gamma = g * cnst.e / (2 * cnst.m_e)
        M_s = (4 * np.pi ** 2 * data ** 2 - gamma ** 2 * cnst.mu_0 ** 2 * (x ** 2 + 2 * x * H_k + H_k ** 2)) / (
            gamma ** 2 * cnst.mu_0 ** 2 * (x + H_k)
        )
        M_s = np.mean(M_s)

        pars = self.make_params(g=g, M_s=M_s, H_k=H_k)
        pars["M_s"].min = 0
        pars["g"].min = g / 100
        pars["H_k"].min = 0
        pars["H_k"].max = M_s.max()
        return update_param_vals(pars, self.prefix, **kwargs)


class Inverse_Kittel(Model):

    r"""Kittel Equation for finding ferromagnetic resonance peak in frequency with field.

    Args:
        H (array): Magnetic fields in A/m
        g (float): h g factor for the gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz

    :math:`f = \frac{\gamma \mu_{0}}{2 \pi} \sqrt{\left(H + H_{k}\right) \left(H + H_{k} + M_{s}\right)}`

    """

    display_names = ["g", "M_s", "H_k"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(Inverse_Kittel, self).__init__(inverse_kittel, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""
        g = 2
        H_k = 100
        gamma = g * cnst.e / (2 * cnst.m_e)
        M_s = (4 * np.pi ** 2 * x ** 2 - gamma ** 2 * cnst.mu_0 ** 2 * (data ** 2 + 2 * data * H_k + H_k ** 2)) / (
            gamma ** 2 * cnst.mu_0 ** 2 * (data + H_k)
        )
        M_s = np.mean(M_s)

        pars = self.make_params(g=g, M_s=M_s, H_k=H_k)
        pars["M_s"].min = 0
        pars["g"].min = g / 100
        pars["H_k"].min = 0
        pars["H_k"].max = M_s.max()
        return update_param_vals(pars, self.prefix, **kwargs)


class FMR_Power(Model):
    r"""A combination of a Lorentzian and differential Lorenztion peak as measured in an FMR experiment.

    Args:
        H (array) magnetic field data
        H_res (float): Resonance field of peak
        Delta_H (float): Preak wideth
        K_1, K_2 (float): Relative weighting of each component.

    Returns:
        Array of model absorption values.

    :math:`\frac{4 \Delta_{H} K_{1} \left(H - H_{res}\right)}{\left(\Delta_{H}^{2} + 4 \left(H - H_{res}\right)^{2}\right)^{2}} - \frac{K_{2} \left(\Delta_{H}^{2} - 4 \left(H - H_{res}\right)^{2}\right)}{\left(\Delta_{H}^{2} + 4 \left(H - H_{res}\right)^{2}\right)^{2}}`
    """
    display_names = ["H_{res}", r"\Delta_H", "K_1", "K_2"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(FMR_Power, self).__init__(fmr_power, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""

        if x is None:
            x = np.linspace(1, len(data), len(data) + 1)

        x1 = x[np.argmax(data)]
        x2 = x[np.argmin(data)]
        Delta_H = abs(x1 - x2)
        H_res = (x1 + x2) / 2.0
        y1 = np.max(data)
        y2 = np.min(data)
        dy = y1 - y2
        K_2 = dy * (4 * np.pi * Delta_H ** 2) / (3 * np.sqrt(3))
        ay = (y1 + y2) / 2
        K_1 = ay * np.pi / Delta_H

        pars = self.make_params(Delta_H=Delta_H, H_res=H_res, K_1=K_1, K_2=K_2)
        pars["K_1"].min = 0
        pars["K_2"].min = 0
        pars["Delta_H"].min = 0
        pars["H_res"].min = np.min(x)
        pars["H_res"].max = np.max(x)
        return update_param_vals(pars, self.prefix, **kwargs)
