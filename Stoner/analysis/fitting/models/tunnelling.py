#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""":py:class:`lmfit.Model` model classes and functions for various models of electron tunnelling."""
# pylint: disable=invalid-name
__all__ = [
    "BDR",
    "FowlerNordheim",
    "Model",
    "Simmons",
    "TersoffHammann",
    "bdr",
    "fowlerNordheim",
    "simmons",
    "tersoffHammann",
]

import numpy as np

from lmfit import Model
from lmfit.models import update_param_vals


def simmons(V, A, phi, d):
    """Simmons model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in micron^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Data for tunneling rate according to the Sommons model.

    .. note::

        Simmons model from Simmons J. App. Phys. 34 6 1963

    Example:
        .. plot:: samples/Fitting/Simmons.py
            :include-source:
            :outname: simmons
    """
    current = (
        6.2e2
        * A
        / d**2
        * (
            (phi - V / 2) * np.exp(-1.025 * d * np.sqrt(phi - V / 2))
            - (phi + V / 2) * np.exp(-1.025 * d * np.sqrt(phi + V / 2))
        )
    )
    return current


def bdr(V, A, phi, dphi, d, mass):
    """BDR model tunnelling.

    Args:
        V (array): ias voltage
        A (float): barrier area in m^2
        phi (float): average barrier height in eV
        dphi (float): change in barrier height in eV
        d (float): barrier width in angstrom
        mass (float): effective electron mass as a fraction of electron rest mass

    Return:
        Data for tunneling rate  according to the BDR model.
    .. note::

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)

    Example:
        .. plot:: samples/Fitting/BDR.py
            :include-source:
            :outname: bdr
    """
    mass = abs(mass)
    phi = abs(phi)
    d = abs(d)
    current = (
        3.16e10
        * A**2
        * np.sqrt(phi)
        / d
        * np.exp(-1.028 * np.sqrt(phi) * d)
        * (V - 0.0214 * np.sqrt(mass) * d * dphi / phi**1.5 * V**2 + 0.0110 * mass * d**2 / phi * V**3)
    )
    return current


def fowlerNordheim(V, A, phi, d):
    """Fowler Nordhiem Model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in m^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Tunneling rate according to Fowler Nordheim model.

    Example:
        .. plot:: samples/Fitting/FowlerNordheim.py
            :include-source:
            :outname: fowlernordheim
    """
    current = V / np.abs(V) * 3.38e6 * A * V**2 / (d**2 * phi) * np.exp(-0.689 * phi**1.5 * d / np.abs(V))
    return current


def tersoffHammann(V, A):
    """Tersoff-Hamman model for tunnelling through STM tip.

    Args:
        V (array): bias voltage
        A (float): Tip conductance

    Return:
        A linear fit.
    """
    current = A * V
    return current


class Simmons(Model):
    """Simmons model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in m^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Data for tunneling rate according to the Sommons model.

    .. note::

       Simmons model from Simmons J. App. Phys. 34 6 1963

    Example:
        .. plot:: samples/Fitting/Simmons.py
            :include-source:
            :outname: simmons-class
    """

    display_names = ["A", r"\phi", "d"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(simmons, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Set the A, phi and d values to typical answers for a small tunnel junction."""
        pars = self.make_params(A=1e3, phi=3.0, d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class BDR(Model):
    """BDR model tunnelling.

    Args:
        V (array): ias voltage
        A (float): barrier area in m^2
        phi (float): average barrier height in eV
        dphi (float): change in barrier height in eV
        d (float): barrier width in angstrom
        mass (float): effective electron mass as a fraction of electron rest mass

    Return:
        Data for tunneling rate  according to the BDR model.

    .. note::

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)

    Example:
        .. plot:: samples/Fitting/BDR.py
            :include-source:
            :outname: bdr-class
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(bdr, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Set the A, phi,dphi,d and mass values to typical answers for a small tunnel junction."""
        pars = self.make_params(A=1e-12, phi=3.0, d=10.0, dphi=1.0, mass=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class FowlerNordheim(Model):
    """Fowler Nordhiem Model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in m^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Tunneling rate according to Fowler Nordheim model.

    Example:
        .. plot:: samples/Fitting/FowlerNordheim.py
            :include-source:
            :outname: fowlernordheim-class
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(fowlerNordheim, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Set the A, phi and d values to typical answers for a small tunnel junction."""
        pars = self.make_params(A=1e-12, phi=3.0, d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class TersoffHammann(Model):
    """Tersoff-Hamman model for tunnelling through STM tip.

    Args:
        V (array): bias voltage
        A (float): Tip conductance

    Return:
        A linear fit.
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(tersoffHammann, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Set the parameter values from an apporximate line."""
        pars = self.make_params(A=np.mean(data / V))
        return update_param_vals(pars, self.prefix, **kwargs)
