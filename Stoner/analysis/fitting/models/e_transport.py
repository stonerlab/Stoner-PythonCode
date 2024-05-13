#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model classes and functions for various models of electron transport (other than tunnelling processes)."""
# pylint: disable=invalid-name
__all__ = ["BlochGrueneisen", "FluchsSondheimer", "WLfit", "blochGrueneisen", "fluchsSondheimer", "wlfit"]

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

from lmfit import Model
from lmfit.models import update_param_vals

try:  # numba is an optional dependency
    from numba import jit, float64, int64
except ImportError:
    from ....compat import _dummy, _jit as jit

    float64 = _dummy()
    int64 = _dummy()


@jit(float64(float64, int64), nopython=True)
def _bgintegrand(x, n):
    """Calculate the integrand for the Bloch Grueneisen model."""
    return x**n / ((np.exp(x) - 1) * (1 - np.exp(-x)))


def wlfit(B, s0, DS, B1, B2):
    """Implement the Weak localisation fitting function.

    Args:
        B (array): mag. field
        s0 (float): zero field conductance
        DS (float): scaling parameter
        B1 (float): elastic characteristic field (B1)
        B2 (float): inelastic characteristic field (B2)

    Returns:
        Conductance vs Field for a weak localisation system

    Notes:
       2D WL model as per Wu et al  PRL 98, 136801 (2007), Porter et al PRB 86, 064423 (2012)

    Example:
        .. plot:: samples/Fitting/weak_localisation.py
            :include-source:
            :outname: wlfit
    """
    e = 1.6e-19  # C
    h = 6.62e-34  # Js
    # Sets up conductivity fit array
    cond = np.zeros(len(B))
    if B2 == B1:
        B2 = B1 * 1.00001  # prevent dividing by zero

    # performs calculation for all parts
    for tt, Bi in enumerate(B):
        if Bi != 0:  # prevent dividing by zero
            WLpt1 = digamma(0.5 + B2 / np.abs(Bi))
            WLpt2 = digamma(0.5 + B1 / np.abs(Bi))
        else:
            WLpt1 = (digamma(0.5 + B2 / np.abs(B[tt - 1])) + digamma(0.5 + B2 / np.abs(B[tt + 1]))) / 2
            WLpt2 = (digamma(0.5 + B1 / np.abs(B[tt - 1])) + digamma(0.5 + B1 / np.abs(B[tt + 1]))) / 2

        WLpt3 = np.log(B2 / B1)

        # Calculates fermi level smearing
        cond[tt] = (e**2 / (h * np.pi)) * (WLpt1 - WLpt2 - WLpt3)
    # cond = s0*cond / min(cond)
    cond = s0 + DS * cond
    return cond


def fluchsSondheimer(t, l, p, sigma_0):
    """Evaluate a Fluchs-Sondheumer model function for conductivity.

    Args:
        t (array): Thickness values
        l (float): mean-free-path
        p (float): reflection co-efficient
        sigma_0 (float): intrinsic conductivity

    Return:
        Reduced Resistivity

    Note:
        Expression used from: G.N.Gould and L.A. Moraga, Thin Solid Films 10 (2), 1972 pp 327-330

    Example:
        .. plot:: samples/Fitting/f_s.py
            :include-source:
            :outname: fluchssondheimer
    """
    k = t / l

    def kernel(x, k):
        return (x - x**3) * np.exp(-k * x) / (1 - np.exp(-k * x))

    result = np.zeros(k.shape)
    for i, v in enumerate(k):
        ret1 = 1 - (3 * (1 - p) / (8 * v)) + (3 * (1 - p) / (2 * v))
        ret2 = quad(kernel, 0, 1, (v,))[0]
        result[i] = ret1 * ret2
    return result / sigma_0


def blochGrueneisen(T, thetaD, rho0, A, n):
    """Calculate the BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Return:
        Evaluation of the BlochGrueneisen function for R(T)

    Example:
        .. plot:: samples/Fitting/b_g.py
            :include-source:
            :outname: blochgruneisen
    """
    ret = np.zeros(T.shape)
    for i, t in enumerate(T):
        intg = quad(_bgintegrand, 0, thetaD / (t), (n,))[0]
        ret[i] = rho0 + A * (t / thetaD) ** n * intg
    return ret


class WLfit(Model):
    """Weak localisation model class.

    Args:
        B (array): mag. field
        s0 (float): zero field conductance
        DS (float): scaling parameter
        B1 (float): elastic characteristic field (B1)
        B2 (float): inelastic characteristic field (B2)

    Returns:
        Conductance vs Field for a weak localisation system

    Notes:
       2D WL model as per Wu et al  PRL 98, 136801 (2007), Porter et al PRB 86, 064423 (2012)

    Example:
        .. plot:: samples/Fitting/weak_localisation.py
            :include-source:
            :outname: wlfit
    """

    display_names = [r"\sigma_0", "D_S", "B_1", "B_2"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(wlfit, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters for weak localisation fit."""
        s0, DS, B1, B2 = 1.0, 1.0, 1.0, 1.0
        if x is not None:
            zpos = np.argmin(np.abs(x))
            s0 = data[zpos]
            B1 = np.max(x) / 20.0
            B2 = B1 * 10
            DS = 1.0
        pars = self.make_params(s0=s0, DS=DS, B1=B1, B2=B2)
        for p in pars:
            pars[p].min = 0.0
        return update_param_vals(pars, self.prefix, **kwargs)


class FluchsSondheimer(Model):
    """Evaluate a Fluchs-Sondheumer model function for conductivity.

    Args:
        t (array): Thickness values
        l (float): mean-free-path
        p (float): reflection co-efficient
        sigma_0 (float): intrinsic conductivity

    Return:
        Reduced Resistivity

    Note:
        Expression used from: G.N.Gould and L.A. Moraga, Thin Solid Films 10 (2), 1972 pp 327-330

    Example:
        .. plot:: samples/Fitting/f_s.py
            :include-source:
            :outname: fluchsdondheimer-class
    """

    display_names = [r"\lambda_{mfp}", "p_{refl}", r"\sigma_0"]
    units = ["nm", "", r"\Omega^{-1}m^{-1}"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(fluchsSondheimer, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):  # pylint: disable=unused-argument
        """Guess some starting values - not very clever."""
        pars = self.make_params(l=10.0, p=0.5, sigma_0=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class BlochGrueneisen(Model):
    """BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Return:
        Evaluation of the BlochGrueneisen function for R(T)

    Example:
        .. plot:: samples/Fitting/b_g.py
            :include-source:
            :outname: blochgruneisen-class
    """

    display_names = [r"\theta_D", r"\rho_0", "A", "n"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(blochGrueneisen, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
        """Guess some starting values - not very clever."""
        rho0 = data.min()

        if x is None:
            t = np.linspace(0, 1, len(data))
        else:
            t = x / x.max()
        y = data - data.min()
        t = t[y > 0.05 * y.max()]
        y = y[y > 0.05 * y.max()]
        A = np.polyfit(t, y, 1)[0]

        pars = self.make_params(thetaD=500, rho0=rho0, A=A, n=5.0)
        pars["A"].min = 0
        pars["n"].vary = False
        return update_param_vals(pars, self.prefix, **kwargs)
