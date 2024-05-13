#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""":py:class:`lmfit.Model` model classes and functions for various superconductivity related models."""
# pylint: disable=invalid-name
# This module can be used with Stoner v.0.9.0 asa standalone module
__all__ = [
    "RSJ_Noiseless",
    "RSJ_Simple",
    "Strijkers",
    "Ic_B_Airy",
    "rsj_noiseless",
    "rsj_simple",
    "strijkers",
    "ic_B_airy",
]

from functools import partial

import numpy as np
from scipy.special import jv
from scipy.constants import physical_constants
from scipy.integrate import quad

from lmfit import Model
from lmfit.models import update_param_vals

hbar = physical_constants["Planck constant over 2 pi"]
kb = physical_constants["Boltzmann constant"]
Phi_0 = physical_constants["mag. flux quantum"][0]

J1 = partial(jv, 1)


try:  # numba is an optional dependency
    from numba import jit, float64
except ImportError:
    from ....compat import _dummy, _jit as jit

    float64 = _dummy()


@jit(float64[:](float64[:], float64, float64, float64, float64), nopython=True, parallel=True, nogil=True)
def _strijkers_core(V, omega, delta, P, Z):
    """Implement strijkers Model for point-contact Andreev Reflection Spectroscopy.

    Args:
        V = bias voltages, params=list of parameter values, imega, delta,P and Z
        omega (float): Broadening
        delta (float): SC energy Gap
        P (float): Interface parameter
        Z (float): Current spin polarization through contact

    Return:
        Conductance vs bias data.

    Note:
           PCAR fitting Strijkers modified BTK model TK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    This version only uses 1 delta, not modified for proximity
    """
    #   Parameters

    mv = np.max(np.abs(V))  # Limit for evaluating the integrals
    E = np.linspace(-2 * mv, 2 * mv, V.size * 20)  # Energy range in meV - we use a mesh 20x denser than data points
    gauss = np.exp(-(E**2 / (2 * omega**2)))
    gauss /= gauss.sum()  # Normalised gaussian for the convolution

    # Conductance calculation
    #    For ease of calculation, epsilon = E/(sqrt(E^2 - delta^2))
    #    Calculates reflection probabilities when E < or > delta
    #    A denotes Andreev Reflection probability
    #    B denotes normal reflection probability
    #    subscript p for polarised, u for unpolarised
    #    Ap is always zero as the polarised current has 0 prob for an Andreev
    #    event

    Au1 = (delta**2) / ((E**2) + (((delta**2) - (E**2)) * (1 + 2 * (Z**2)) ** 2))
    Au2 = (((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) ** 2) - 1) / (
        ((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) + (1 + 2 * (Z**2))) ** 2
    )
    Bu2 = (4 * (Z**2) * (1 + (Z**2))) / (((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) + (1 + 2 * (Z**2))) ** 2)
    Bp2 = Bu2 / (1 - Au2)

    unpolarised_prefactor = (1 - P) * (1 + (Z**2))
    polarised_prefactor = 1 * (P) * (1 + (Z**2))
    # Optimised for a single use of np.where
    G = (
        unpolarised_prefactor
        + polarised_prefactor
        + +np.where(
            np.abs(E) <= delta,
            unpolarised_prefactor * (2 * Au1 - 1) - np.ones_like(E) * polarised_prefactor,
            unpolarised_prefactor * (Au2 - Bu2) - Bp2 * polarised_prefactor,
        )
    )

    # Convolve and chop out the central section
    cond = np.convolve(G, gauss)
    cond = cond[(E.size // 2) : 3 * (E.size // 2)]
    # Linear interpolation back onto the V data point
    matches = np.searchsorted(E, V)
    condl = cond[matches - 1]
    condh = cond[matches]
    El = E[matches - 1]
    Er = E[matches]
    cond = (condh - condl) / (Er - El) * (V - El) + condl
    return cond


def strijkers(V, omega, delta, P, Z):
    """Strijkers Model for point-contact Andreev Reflection Spectroscopy.

    Args:
        V (array):
            bias voltages
        omega (float):
            Broadening
        delta (float):
            SC energy Gap
        P (float):
            Interface parameter
        Z (float):
            Current spin polarization through contact

    Return:
        Conductance vs bias data.

    .. note::

       PCAR fitting Strijkers modified BTK model TK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    This version only uses 1 delta, not modified for proximity

    Example:
        .. plot:: samples/lmfit_demo.py
            :include-source:
            :outname: strijkers_func
    """
    if isinstance(V, np.ma.MaskedArray):
        mask = V.mask
        V = np.array(V)
    else:
        mask = False
    ret = _strijkers_core(V, omega, delta, P, Z)
    ret = np.ma.MaskedArray(ret)
    ret.mask = mask
    return ret


def rsj_noiseless(I, Ic_p, Ic_n, Rn, V_offset):
    r"""Implement a simple noiseless RSJ model.

    Args:
        I (array-like): Current values
        Ic_p (foat): Critical current on positive branch
        Ic_n (foat): Critical current on negative branch
        Rn (float): Normal state resistance
        V_offset(float): Offset volage in measurement

    Returns:
        (array) Calculated voltages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            :math:`V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}`

    Example:
        .. plot:: samples/Fitting/rsj_fit.py
            :include-source:
            :outname: rsj_noiseless_func
    """
    normal_p = np.sign(I) * np.real(np.sqrt(I**2 - Ic_p**2)) * Rn
    normal_n = np.sign(I) * np.real(np.sqrt(I**2 - Ic_n**2)) * Rn
    p_branch = np.where(I > Ic_p, normal_p, np.zeros_like(I))
    n_branch = np.where(I < Ic_n, normal_n, p_branch)
    return n_branch + V_offset


def rsj_simple(I, Ic, Rn, V_offset):
    r"""Implement a simple noiseless symmetric RSJ model.

    Args:
        I (array-like):
            Current values
        Ic (foat):
            Critical current
        Rn (float):
            Normal state resistance
        V_offset(float):
            Offset volage in measurement

    Returns:
        (array):
            Calculated voltages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            :math:`V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}`

    Example:
        .. plot:: samples/Fitting/rsj_fit.py
            :include-source:
            :outname: rsj_simple_func
    """
    normal = Rn * np.sign(I) * np.real(np.sqrt(I**2 - Ic**2))
    ic_branch = np.zeros_like(I)
    return np.where(np.abs(I) < Ic, ic_branch, normal) + V_offset


def ic_B_airy(B, Ic0, B_offset, A):
    r"""Calculate Critical Current for a round Josepshon Junction wrt to Field.

    Args:
        B (array-like):
            Magnetic Field (structly flux density in T)
        Ic0 (float):
            Maximum critical current
        B_offset (float):
            Field offset/trapped flux in coils/remanent M in junction
        A(fl,oat):
            Area of junction in $m^2$

    Returns:
        (array):
            Values of critical current

    Notes:
        Represents the critical current as:
            :math:`I_{c0}\times\left|\frac{2 J_1\left(\frac{\pi\(B-B_{offset}) A}\right)}{\Phi_0}}
                    {\frac{\pi\(B-B_{offset}) A}){\Phi_0}}\right|`
        where :math:`J_1` is a first order Bessel function.

        For small ($<1^{-5}$)values of the Bessel function argument, this will return Ic0 to
        ensure correct evaluation for 0 flux.

    Example:
        .. plot:: samples/Fitting/ic_b_airy.py
            :include-source:
            :outname: ic_b_airy_func
    """
    arg = (B - B_offset) * A * np.pi / Phi_0

    return Ic0 * np.abs(2 * np.where(np.abs(arg) < 1e-5, np.ones_like(arg), J1(arg) / arg))


def icRN_Clean(d_f, IcRn0, E_x, v_f, d_0):
    r"""Critical Current versus ferromagnetic narrier thickness, clean limit.

    Args:
        d_f (array):
            ferromagnetic barrier thickness (nm)
        IcRn0 (float):
            Characteristic voltage scaling factor
        E_x (float):
            Exchange energy (eV)
        v_f (float):
            Fermi velocity (ms^-1)
        d_0 (float):
            barrier thickness offset

    Returns:
        (array):
            IcRn values

    Notes:
        Implements

        Lmath:`I_cR_N = I_cR_N^0\zfrac{\sin(2E_x (d_f-d_0)/hv_f)}{2E_x(d_f-d_0)/hv_f}`
    """
    h = physical_constants["Planck constant in eV s"]
    x = (d_f - d_0) * 1e-9
    A = 2 * E_x / (v_f * h)
    return IcRn0 * np.abs(np.sin(A * x)) / (A * x)


def ic_RN_Dirty(d_f, IcRn0, E_x, v_f, d_0, tau, delta, T):
    r"""Critical Current versus ferromagnetic narrier thickness, clean limit.

    Args:
        d_f (array):
            ferromagnetic barrier thickness (nm)
        IcRn0 (float):
            Characteristic voltage scaling factor
        E_x (float):
            Exchange energy (eV)
        v_f (float):
            Fermi velocity (ms^-1)
        d_0 (float):
            barrier thickness offset
        l (float):
            mean-free-path (nm)

    Returns:
        (array):
            IcRn values

    Notes:
        Implements Eq 18 from F.S. Bergeret, A.F. Volkov, and K.B. Efetov, Phys. Rev. B 64, 134506 (2001).
    """
    L = v_f * tau * 1e-9
    t = tau / hbar
    w_m = lambda m: (2 * m + 1) * T * np.pi * kb
    k_m = lambda m: (1 + 2 * np.abs(w_m(m)) * t) + (0 - 2j) * E_x * t
    integrad = lambda mu, m: np.real(mu / (np.sinh(d_f * k_m(m) / (mu * L))))
    prefactor = lambda m: delta**2 / (delta**2 + w_m(m) ** 2)

    term = lambda m: prefactor(m) * quad(integrad, -1, 1, (m,))  # pylint: disable=W0612


class Strijkers(Model):
    """strijkers Model for point-contact Andreev Reflection Spectroscopy.

    Args:
        V (array): bias voltages
        omega (float): Broadening
        delta (float): SC energy Gap
        P (float): Interface parameter
        Z (float): Current spin polarization through contact

    Return:
        Conductance vs bias data.

    .. note::

       PCAR fitting Strijkers modified BTK model TK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    This version only uses 1 delta, not modified for proximity

    Example:
        .. plot:: samples/lmfit_demo.py
            :include-source:
            :outname: strijkers_class
    """

    display_names = [r"\omega", r"\Delta", "P", "Z"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(strijkers, *args, **kwargs)

    def guess(self, data, **kwargs):  # pylint: disable=unused-argument
        """Guess starting values for a good Nb contact to a ferromagnet at 4.2K."""
        pars = self.make_params(omega=0.5, delta=1.50, P=0.42, Z=0.15)
        pars["omega"].min = 0.36
        pars["omega"].max = 5.0
        pars["delta"].min = 0.5
        pars["delta"].max = 2.0
        pars["Z"].min = 0.1
        pars["P"].min = 0.0
        pars["P"].max = 1.0
        return update_param_vals(pars, self.prefix, **kwargs)


class RSJ_Noiseless(Model):
    r"""Implement a simple noiseless RSJ model.

    Args:
        I (array-like): Current values
        Ic_p (foat): Critical current on positive branch
        Ic_n (foat): Critical current on negative branch
        Rn (float): Normal state resistance
        V_offset(float): Offset volage in measurement

    Returns:
        (array) Calculated voltages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            :math:`V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}`

    Example:
        .. plot:: samples/Fitting/rsj_fit.py
            :include-source:
            :outname: rsj_noiseless_class
    """

    display_names = ["I_c^p", "I_c^n", "R_N", "V_{offset}"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(rsj_noiseless, *args, **kwargs)

    def guess(self, data, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H."""
        x = kwargs.get("x", np.linspace(1, len(data), len(data) + 1))

        v_offset_guess = np.mean(data)
        v = np.abs(data - v_offset_guess)
        x = np.abs(x)

        v_low = np.max(v) * 0.05
        v_high = np.max(v) * 0.90

        ic_index = v < v_low
        rn_index = v > v_high
        ic_guess = np.max(x[ic_index])  # Guess Ic from a 2% of max V threhsold creiteria

        rn_guess = np.mean(v[rn_index] / x[rn_index])

        pars = self.make_params(Ic_p=ic_guess, Ic_n=-ic_guess, Rn=rn_guess, V_offset=v_offset_guess)
        pars["Ic_p"].min = 0
        pars["Ic_n"].max = 0
        return update_param_vals(pars, self.prefix, **kwargs)


class RSJ_Simple(Model):
    r"""Implements a simple noiseless symmetric RSJ model.

    Args:
        I (array-like): Current values
        Ic (foat): Critical current
        Rn (float): Normal state resistance
        V_offset(float): Offset volage in measurement

    Returns:
        (array) Calculated voltages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            :math:`V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}`

    Example:
        .. plot:: samples/Fitting/rsj_fit.py
            :include-source:
            :outname: rsj_simple_class

    """

    display_names = ["I_c", "R_N", "V_{offset}"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(rsj_simple, *args, **kwargs)

    def guess(self, data, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H."""
        x = kwargs.get("x", np.linspace(1, len(data), len(data) + 1))

        v_offset_guess = np.mean(data)
        v = np.abs(data - v_offset_guess)
        x = np.abs(x)

        v_low = np.max(v) * 0.05
        v_high = np.max(v) * 0.90

        ic_index = v < v_low
        rn_index = v > v_high
        ic_guess = np.max(x[ic_index])  # Guess Ic from a 2% of max V threhsold creiteria

        rn_guess = np.mean(v[rn_index] / x[rn_index])

        pars = self.make_params(Ic=ic_guess, Rn=rn_guess, V_offset=v_offset_guess)
        # pars["Ic"].min = 0
        return update_param_vals(pars, self.prefix, **kwargs)


class Ic_B_Airy(Model):
    r"""Critical Current for a round Josepshon Junction wrt to Field.

    Args:
        B (array-like):
            Magnetic Field (structly flux density in T)
        Ic0 (float):
            Maximum critical current
        B_offset (float):
            Field offset/trapped flux in coils/remanent M in junction
        A(fl,oat):
            Area of junction in $m^2$

    Returns:
        (array):
            Values of critical current

    Notes:
        Represents the critical current as:
            :math:`I_{c0}\times\left|\frac{2 J_1\left(\frac{\pi\(B-B_{offset}) A}\right)}
                        {\Phi_0}}{\frac{\pi\(B-B_{offset}) A}){\Phi_0}}\right|`
        where `J_1` is a first order Bessel function.

    Example:
        .. plot:: samples/Fitting/ic_b_airy.py
            :include-source:
            :outname: ic_b_airy_class

    """

    display_names = ["I_{c0}", "B_{offset}"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(ic_B_airy, *args, **kwargs)

    def guess(self, data, **kwargs):
        """Guess parameters as max(data), x[argmax(data)] and from FWHM of peak."""
        x = kwargs.get("x", np.linspace(-len(data) / 2, len(data) / 2, len(data)))

        Ic0_guess = data.max()
        B_offset_guess = x[data.argmax()]
        tmp = np.abs(data - (data.max() / 2))
        x0 = np.abs(x[tmp.argmin()] - B_offset_guess)
        A_guess = 2.2 * Phi_0 / (np.pi * x0)

        pars = self.make_params(Ic0=Ic0_guess, B_offset=B_offset_guess, A=A_guess)
        pars["Ic0"].min = 0
        return update_param_vals(pars, self.prefix, **kwargs)
