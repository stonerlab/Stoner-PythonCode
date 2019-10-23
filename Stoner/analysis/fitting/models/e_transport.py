#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:py:class:`lmfit.Model` model classes and functions for various models of electron transport (other than tunnelling processes).
"""

__all__ = ["BlochGrueneisen", "FluchsSondheimer", "WLfit", "blochGrueneisen", "fluchsSondheimer", "wlfit"]

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

try:
    from lmfit import Model
    from lmfit.models import update_param_vals
except ImportError:
    Model = object
    update_param_vals = None


def _bgintegrand(x, n):
    """The integrand for the Bloch Grueneisen model."""
    return x ** n / ((np.exp(x) - 1) * (1 - np.exp(-x)))


def wlfit(B, s0, DS, B1, B2):
    """
    Weak localisation

    Args:
        B (array): mag. field
        s0 (float): zero field conductance
        DS (float): scaling parameter
        B1 (float): elastic characteristic field (B1)
        B2 (float): inelastic characteristic field (B2)

    Return:
        Conductance vs Field for a weak localisation system

    .. note::

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
        cond[tt] = (e ** 2 / (h * np.pi)) * (WLpt1 - WLpt2 - WLpt3)
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

    kernel = lambda x, k: (x - x ** 3) * np.exp(-k * x) / (1 - np.exp(-k * x))

    result = np.zeros(k.shape)
    for i, v in enumerate(k):
        ret1 = 1 - (3 * (1 - p) / (8 * v)) + (3 * (1 - p) / (2 * v))
        ret2 = quad(kernel, 0, 1, (v,))[0]
        result[i] = ret1 * ret2
    return result / sigma_0


def blochGrueneisen(T, thetaD, rho0, A, n):
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
            :outname: blochgruneisen
    """
    ret = np.zeros(T.shape)
    for i, t in enumerate(T):
        intg = quad(_bgintegrand, 0, thetaD / (t), (n,))[0]
        ret[i] = rho0 + A * (t / thetaD) ** n * intg
    return ret


class WLfit(Model):

    """
    Weak localisation

    Args:
        B (array): mag. field
        s0 (float): zero field conductance
        DS (float): scaling parameter
        B1 (float): elastic characteristic field (B1)
        B2 (float): inelastic characteristic field (B2)

    Return:
        Conductance vs Field for a weak localisation system

    .. note::

       2D WL model as per Wu et al  PRL 98, 136801 (2007), Porter et al PRB 86, 064423 (2012)

    Example:
        .. plot:: samples/Fitting/weak_localisation.py
            :include-source:
            :outname: wlfit-class
    """

    display_names = [r"\sigma_0", "D_S", "B_1", "B_2"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(WLfit, self).__init__(wlfit, *args, **kwargs)

    def guess(self, data, B=None, **kwargs):
        """Guess parameters for weak localisation fit."""
        s0, DS, B1, B2 = 1.0, 1.0, 1.0, 1.0
        if B is not None:
            zpos = np.argmin(np.abs(B))
            s0 = data[zpos]
            B1 = np.max(B) / 2.0
            B2 = B1
            DS = 1.0
        pars = self.make_params(s0=s0, DS=DS, B1=B1, B2=B2)
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

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(FluchsSondheimer, self).__init__(fluchsSondheimer, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):  # pylint: disable=unused-argument
        """Guess some starting values - not very clever"""
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
        super(BlochGrueneisen, self).__init__(blochGrueneisen, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):  # pylint: disable=unused-argument
        """Guess some starting values - not very clever"""
        pars = self.make_params(thetaD=900, rho0=0.01, A=0.2, n=5.0)
        return update_param_vals(pars, self.prefix, **kwargs)
