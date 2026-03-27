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
from lmfit import Model
from lmfit.models import update_param_vals
from scipy.constants import physical_constants
from scipy.integrate import quad
from scipy.special import jv
from scipy.signal import fftconvolve

hbar = physical_constants["Planck constant over 2 pi"]
kb = physical_constants["Boltzmann constant"]
Phi_0 = physical_constants["mag. flux quantum"][0]

J1 = partial(jv, 1)


try:  # numba is an optional dependency
    from numba import float64, jit, njit
except ImportError:
    from ....compat import _dummy
    from ....compat import _jit as jit

    njit = jit
    float64 = _dummy()


# @jit(float64[:](float64[:], float64, float64, float64, float64), nopython=True, parallel=True, nogil=True)
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
    if np.isclose(omega, 0):
        E = V
    else:
        E = np.linspace(
            -2 * np.max(np.abs(V)), 2 * np.max(np.abs(V)), V.size * 20
        )  # Energy range in meV - we use a mesh 20x denser than data points

    U = (
        (delta**2) / ((E**2) + (((delta**2) - (E**2)) * (1 + 2 * (Z**2)) ** 2)),
        (((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) ** 2) - 1)
        / (((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) + (1 + 2 * (Z**2))) ** 2),
        (4 * (Z**2) * (1 + (Z**2))) / (((np.abs(E) / (np.sqrt((E**2) - (delta**2)))) + (1 + 2 * (Z**2))) ** 2),
    )

    # Optimised for a single use of np.where
    G = (
        (1 - P) * (1 + (Z**2))
        + 1 * (P) * (1 + (Z**2))
        + +np.where(
            np.abs(E) <= delta,
            (1 - P) * (1 + (Z**2)) * (2 * U[0] - 1) - np.ones_like(E) * 1 * (P) * (1 + (Z**2)),
            (1 - P) * (1 + (Z**2)) * (U[1] - U[2]) - (U[2] / (1 - U[1])) * 1 * (P) * (1 + (Z**2)),
        )
    )

    # Convolve and chop out the central section
    if np.isclose(omega, 0):
        return G

    gauss = np.exp(-(E**2 / (2 * omega**2)))
    gauss /= gauss.sum()  # Normalised gaussian for the convolution

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


# ============================================================
# Shared helper functions
# ============================================================


@njit
def _beta_func(V, Delta):
    """Compute β = V / sqrt(V² − Δ²).

    Args:
        V (complex or float): Bias voltage.
        Delta (float): Energy gap Δ.

    Returns:
        complex: β value (complex below the gap).
    """
    if np.abs(V - Delta) < 1e-9:
        return 1e9
    return V / np.sqrt(np.abs(V * V - Delta * Delta))


@njit
def _F_func(x, Z):
    """Compute F(x) = arccosh(2Z² + x) / sqrt((2Z² + x)² − 1).

    Args:
        x (complex): Argument to the F function.
        Z (float): Barrier strength.

    Returns:
        complex: Value of F(x).
    """
    arg = 2.0 * Z * Z + x
    return np.arccosh(arg) / np.sqrt(arg * arg - 1.0)


@njit
def _safe_above_v(v, Delta):
    """Return a v slightly above the gap if v is numerically at |v| = Δ."""
    tol = 1e-12 * (1.0 + Delta)
    av = abs(v)
    if abs(av - Delta) < tol:
        s = 1.0 if v >= 0.0 else -1.0
        return s * (Delta + tol)
    return v


@njit
def _safe_below_v(v, Delta):
    """Return a v slightly below the gap if v is numerically at |v| = Δ."""
    tol = 1e-12 * (1.0 + Delta)
    av = abs(v)
    if abs(av - Delta) < tol:
        s = 1.0 if v >= 0.0 else -1.0
        return s * (Delta - tol)
    return v


@njit
def _fill_nans_with_interpolation(V, I):
    """
    Replace NaNs in I with linear interpolation using the positions in V.
    Works for any 1D arrays of equal length.
    """
    V = np.asarray(V)
    I = np.asarray(I)

    nan_mask = np.isnan(I)

    if not nan_mask.any():
        return I  # nothing to fix

    # Indices where I is good
    good = ~nan_mask

    # Interpolate over the NaN positions
    I_filled = I.copy()
    I_filled[nan_mask] = np.interp(
        V[nan_mask],  # x-values needing interpolation
        V[good],  # x-values with valid data
        I[good],  # corresponding y-values
    )

    return I_filled


# ============================================================
# Ballistic non-magnetic
# ============================================================


@njit
def _ballistic_nonmagnetic(Delta, V, Z):
    """Compute ballistic non-magnetic interface current.

    Args:
        Delta (float): Energy gap Δ.
        V (ndarray): Bias voltages.
        Z (float): Barrier strength.

    Returns:
        ndarray: Interface current for each V.
    """
    N = V.size
    I = np.zeros(N, dtype=np.float64)
    b = _beta_func(100 * Delta, Delta)
    I_0 = (2.0 * b) / (1.0 + b + 2.0 * Z * Z)

    for i in range(N):
        v = np.abs(V[i])
        if v < Delta:
            b = _beta_func(v, Delta)
            bb_real = b * b
            num = 2.0 * (1.0 + bb_real)
            den = bb_real + (1.0 + 2.0 * Z * Z) ** 2
            I[i] = num / den
        else:
            b = _beta_func(v, Delta)
            I[i] = (2.0 * b) / (1.0 + b + 2.0 * Z * Z)

    return I / I_0


# ============================================================
# Ballistic half-metallic
# ============================================================


@njit
def _ballistic_halfmetal(Delta, V, Z):
    """Compute ballistic half-metallic interface current.

    Args:
        Delta (float): Energy gap Δ.
        V (ndarray): Bias voltages.
        Z (float): Barrier strength.

    Returns:
        ndarray: Interface current for each V.
    """
    N = V.size
    I = np.zeros(N, dtype=np.float64)
    b = _beta_func(100 * Delta, Delta)
    I_0 = (4.0 * b) / ((1.0 + b) * (1.0 + b) + 4.0 * Z * Z)

    for i in range(N):
        v = np.abs(V[i])
        if v < Delta:
            I[i] = 0.0
        else:
            b = _beta_func(v, Delta)
            I[i] = (4.0 * b) / ((1.0 + b) * (1.0 + b) + 4.0 * Z * Z)

    return I / I_0


# ============================================================
# Diffusive non-magnetic
# ============================================================


@njit
def _diffusive_nonmagnetic(Delta, V, Z):
    """Compute diffusive non-magnetic interface current.

    Args:
        Delta (float): Energy gap Δ.
        V (ndarray): Bias voltages.
        Z (float): Barrier strength.

    Returns:
        ndarray: Interface current for each V.
    """
    N = V.size
    I = np.zeros(N, dtype=np.float64)
    b = _beta_func(100 * Delta, Delta)
    I_0 = b * _F_func(b, Z).real

    for i in range(N):
        v = np.abs(V[i])
        if np.isclose(v, 0):
            v = 1e-9
        if v < Delta:
            b = _beta_func(v, Delta)
            F1 = _F_func(-1j * b, Z)
            F2 = _F_func(+1j * b, Z)
            imag_part = (F1 - F2).imag
            bb_real = np.abs(b) ** 2
            I[i] = ((1.0 + bb_real) / (2.0 * b)) * imag_part
        else:
            b = _beta_func(v, Delta)
            I[i] = b * _F_func(b, Z)

    return I / I_0


# ============================================================
# Diffusive half-metallic
# ============================================================


@njit
def _diffusive_halfmetal(Delta, V, Z):
    """Compute diffusive half-metallic interface current.

    Args:
        Delta (float): Energy gap Δ.
        V (ndarray): Bias voltages.
        Z (float): Barrier strength.

    Returns:
        ndarray: Interface current for each V.
    """
    N = V.size
    I = np.zeros(N, dtype=np.float64)
    b = _beta_func(100 * Delta, Delta)
    x = (1.0 + b) * (1.0 + b) / 2.0 - 1.0
    I_0 = b * _F_func(x, Z).real

    for i in range(N):
        v = np.abs(V[i])
        if v < Delta:
            I[i] = 0.0
        else:
            b = _beta_func(v, Delta)
            x = (1.0 + b) * (1.0 + b) / 2.0 - 1.0
            I[i] = b * _F_func(x, Z).real

    return I / I_0


# ============================================================
# Gaussian convolution for non-uniform V
# ============================================================


# -----------------------------------------------------------
# 1. Build Gaussian kernel (Numba)
# -----------------------------------------------------------
@njit
def _make_gaussian_kernel(dV, dV_sampling):
    half = int(4 * dV / dV_sampling)
    size = 2 * half + 1

    g = np.empty(size)
    norm = 0.0

    # Generate kernel
    for i in range(size):
        x = (i - half) * dV_sampling
        val = np.exp(-0.5 * (x / dV) ** 2)
        g[i] = val
        norm += val

    # Normalize
    for i in range(size):
        g[i] /= norm

    return g


# -----------------------------------------------------------
# 2. Reflective padding (Numba)
# -----------------------------------------------------------
@njit
def _reflect_pad(arr, half):
    n = len(arr)
    out = np.empty(n + 2 * half)

    # middle
    for i in range(n):
        out[i + half] = arr[i]

    # left pad
    for i in range(half):
        out[half - 1 - i] = arr[i + 1]

    # right pad
    for i in range(half):
        out[n + half + i] = arr[n - 2 - i]

    return out


# ============================================================
# Gaussian Conmvolution functions
# ============================================================


@njit
def _gaussian_convolution_numba(V, I, dV):
    """
    Numba-optimized Gaussian convolution for irregular (V, I) data.

    Parameters
    ----------
    V : 1D array
        Voltage values (unsorted, duplicates allowed)
    I : 1D array
        Intensity values
    dV : float
        Gaussian sigma (width)

    Returns
    -------
    Iconv : 1D float array
        Convolved intensities evaluated at each V[i]
    """

    N = len(V)
    Iconv = np.zeros(N)

    two_sigma2 = 2.0 * dV * dV
    if np.isclose(two_sigma2, 0.0):
        two_sigma2 = (V.max() - V.min()) / 1e6

    for i in range(N):
        Vi = V[i]
        num = 0.0
        den = 0.0

        for j in range(N):
            diff = Vi - V[j]
            w = np.exp(-(diff * diff) / two_sigma2)
            num += w * I[j]
            den += w

        Iconv[i] = num / den

    return Iconv


# ============================================================
# Top-level interface
# ============================================================


def woods(ballistic, V, omega, delta, P, Z):
    """Compute the interface current with spin polarisation and optional
    Gaussian broadening.

    This function mixes non-magnetic and half-metallic channels using:
        I = (1 - P) * I_nonmag + P * I_half
    using the models from:
        Woods, G. ~T. T. et al.
        Analysis of point-contact Andreev reflection spectra in spin polarization measurements.
        Phys. Rev. B 70, 054416/1–8 (2004).

    Args:
        ballistic (bool): True for ballistic regime, False for diffusive.
        V (array-like):
            Bias voltages (1D, non-uniform allowed).
        omega (float):
            Gaussian broadening width. Defaults to 0.
        delta (float):
            Energy gap Δ.
        P (float):
            Spin polarisation (0 = non-magnetic, 1 = half-metallic).
        Z (float):
            Barrier strength.

    Returns:
        (ndarray):
            Final (optionally convolved) interface current.
    """
    V = np.asarray(V, dtype=np.float64)

    if ballistic:
        I_nonmag = _ballistic_nonmagnetic(delta, V, Z)
        I_half = _ballistic_halfmetal(delta, V, Z)
    else:
        I_nonmag = _diffusive_nonmagnetic(delta, V, Z)
        I_half = _diffusive_halfmetal(delta, V, Z)

    I = (1.0 - P) * I_nonmag + P * I_half
    I = _fill_nans_with_interpolation(V, I)
    return _gaussian_convolution_numba(V, I, omega)


woods_diffusive = partial(woods, False)
woods_diffusive.__name__ = "woods_diffusive"
woods_ballistic = partial(woods, True)
woods_ballistic.__name__ = "woods_ballistic"


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


def ic_RN_Dirty(d_f, IcRn0, E_x, v_f, d_0, tau, delta, T):  # pylint: disable=unused-argument
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

    def _w_m(m):
        return (2 * m + 1) * T * np.pi * kb

    def _k_m(m):
        return (1 + 2 * np.abs(_w_m(m)) * t) + (0 - 2j) * E_x * t

    def _integrad(mu, m):
        return np.real(mu / (np.sinh(d_f * _k_m(m) / (mu * L))))

    def _prefactor(m):
        return delta**2 / (delta**2 + _w_m(m) ** 2)

    def term(m):  # pylint: disable=unused-variable
        return _prefactor(m) * quad(_integrad, -1, 1, (m,))  # pylint


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

    def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


class Woods_Diffusive(Model):
    """Compute the interface current with spin polarisation and optional
    Gaussian broadening.

    Args:
        V (array-like):
            Bias voltages (1D, non-uniform allowed).
        omega (float):
            Gaussian broadening width. Defaults to 0.
        delta (float):
            Energy gap Δ.
        P (float):
            Spin polarisation (0 = non-magnetic, 1 = half-metallic).
        Z (float):
            Barrier strength.

    Returns:
        (ndarray):
            Final (optionally convolved) interface current.

    Notes:
        This model mixes non-magnetic and half-metallic channels using:
            I = (1 - P) * I_nonmag + P * I_half
        using the models from:
            Woods, G. ~T. T. et al.
            Analysis of point-contact Andreev reflection spectra in spin polarization measurements.
            Phys. Rev. B 70, 054416/1–8 (2004).
    """

    display_names = [r"\omega", r"\Delta", "P", "Z"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(woods_diffusive, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


class Woods_Ballistic(Model):
    """Compute the interface current with spin polarisation and optional
    Gaussian broadening.

    Args:
        V (array-like):
            Bias voltages (1D, non-uniform allowed).
        omega (float):
            Gaussian broadening width. Defaults to 0.
        delta (float):
            Energy gap Δ.
        P (float):
            Spin polarisation (0 = non-magnetic, 1 = half-metallic).
        Z (float):
            Barrier strength.

    Returns:
        (ndarray):
            Final (optionally convolved) interface current.

    Notes:
        This model mixes non-magnetic and half-metallic channels using:
            I = (1 - P) * I_nonmag + P * I_half
        using the models from:
            Woods, G. ~T. T. et al.
            Analysis of point-contact Andreev reflection spectra in spin polarization measurements.
            Phys. Rev. B 70, 054416/1–8 (2004).
    """

    display_names = [r"\omega", r"\Delta", "P", "Z"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(woods_ballistic, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


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

    display_names = [r"\omega", r"\Delta", "P", "Z"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super().__init__(strijkers, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


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

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H."""

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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


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

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H."""

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

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)


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

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as max(data), x[argmax(data)] and from FWHM of peak."""

        Ic0_guess = data.max()
        B_offset_guess = x[data.argmax()]
        tmp = np.abs(data - (data.max() / 2))
        x0 = np.abs(x[tmp.argmin()] - B_offset_guess)
        A_guess = 2.2 * Phi_0 / (np.pi * x0)

        pars = self.make_params(Ic0=Ic0_guess, B_offset=B_offset_guess, A=A_guess)
        pars["Ic0"].min = 0
        return update_param_vals(pars, self.prefix, **kwargs)

    def copy(self, **kwargs):
        """Make a new copy of the model."""
        return self.__class__(**kwargs)
