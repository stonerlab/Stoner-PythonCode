"""Library of functions that nlfit can use to fit to. 
=======================================================

Functions should accept an array of x values and a list of parmeters, they should then return an array of y values 
the same size as the x array.

Please do keep documentation up to date, see other functions for documentation examples 

These are just wrappers to Stoner.Fit which has the functions defined in terms of multiple parameter arguments.
New functions should be added to Stoner.Fit first and aliased here.

"""

import Stoner.Fit as _SF

def Linear(x, p):
    """Simple linear function"""
    return _SF.Lineaer(x,*p)

def Arrhenius(x, p):
    """Arrhenius Equation without T dependendent prefactor"""
    return _SF.Arrhehius(x,*p)

def NDimArrhenius(x, p):
    """Arrhenius Equation without T dependendent prefactor"""
    return _SF.NDimArrhehius(x,*p)

def  ModArrhenius(x, p):
    """Arrhenius Equation with a variable T power dependent prefactor"""
    return _SF.ModArrhehius(x,*p)

def PowerLaw(x,p):
    """Power Law Fitting Equation"""
    return _SF.PowerLaw(x,*p)

def Quadratic(x, p):
    """Simple Qudratic Function."""
    return _SF.Quadratic(x,*p)


def Simmons(V, params):
    """Simmons model tunnelling
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    return _SF.Simmons(V,*params)

def BDR(V, params):
    """BDR model tunnelling
    V=bias voltage, params=[A, phi, dphi, d, mass]
    A: in m^2, phi: average barrier height in eV, dphi: change in barrier height in eV,
    d: barrier width in angstrom, mass: effective electron mass as a fraction of electron rest mass

    See Brinkman et. al. J. Appl. Phys. 41 1915 (1970)
    or Tuan Comm. in Phys. 16, 1, (2006)"""
    return _SF.BDR(V,*params)

def FowlerNordheim(V, params):
    """Simmons model tunnelling at V>phi
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    return _SF.FowlerNordheim(V,*params)

def TersoffHammann(V, params):
    """TersoffHamman model for tunnelling through STM tip
    V=bias voltage, params=[A]
    """
    return _SF.TersoffHammann(V,*params)

def WLfit(B, params):
    """Weak localisation
    VRH(B, params):
    B = mag. field, params=list of parameter values, s0, B1, B2

    2D WL model as per
    Wu PRL 98, 136801 (2007)
    Porter PRB 86, 064423 (2012)
    """
    return _SF.WLfit(B,*params)

def strijkers(V, params):
    """strijkers(V, params):
    
    Args:
    V (array): bias voltages
    params (list): parameter values: omega, delta,P and Z
    
    Note:
        PCAR fitting Strijkers modified BTK model
        BTK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000
        
        Only using 1 delta, not modified for proximity
    """
    return _SF.Strijkers(V,*params)


def FluchsSondheimer(t,params):
    """Evaluate a Fluchs-Sondheumer model function for conductivity.

    Args:
        t (array): Thickness values
        params (array): [mean-free-path, reflection co-efficient,sigma_0]

    Returns:
        Reduced Resistivity

    Note:
        Expression used from: G.N.Gould and L.A. Moraga, Thin Solid Films 10 (2), 1972 pp 327-330
"""
    return _SF.FluchsSondheimer(t,*params)

