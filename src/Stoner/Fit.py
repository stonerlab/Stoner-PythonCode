"""Library of functions that ncan be used for fitting. 
=======================================================

Functions should accept an array of x values and a number of parmeters, 
they should then return an array of y values the same size as the x array.

Please do keep documentation up to date, see other functions for documentation examples.

All the functions here defined for scipy.optimize.curve_fit to call themm
i.e. the parameters are expanded to separate arguements, other fitting routines prefere
to have the parameters as a single list or vector. For this reason, :py:mod:`Stoner.FittingFuncs`
has aliases of these functions that use *tuple magic to make that conversion.

If you are writing new functions, please add them here first and then alias then with the parameter
list form in FittingFuncs.
"""

import numpy
from scipy.special import digamma
from scipy.integrate import quad
import scipy.constants.codata as consts

def Linear(x, m, c):
    """Simple linear function"""
    return m*x+c

def Arrhenius(x, A, DE):
    """Arrhenius Equation without T dependendent prefactor"""
    _kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]
    return A*numpy.exp(-DE/(_kb*x))


def NDimArrhenius(x, A, DE, n):
    """Arrhenius Equation without T dependendent prefactor"""
    return Arrhenius(x**n, A, DE)

def  ModArrhenius(x, A, DE, n):
    """Arrhenius Equation with a variable T power dependent prefactor"""
    return (x**n)*Arrhenius(x, A, DE)

def PowerLaw(x, A, n):
    """Power Law Fitting Equation"""
    return A*x**n

def Quadratic(x, a,b,c):
    return a*x**2+b*x+c

def Simmons(V, A, phi, d):
    """
    Simmons model tunnelling
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    I=6.2e6*A/d**2*((phi-V/2)*numpy.exp(-1.025*d*numpy.sqrt(phi-V/2))-(phi+V/2)*numpy.exp(-1.025*d*numpy.sqrt(phi+V/2)))
    return I

def BDR(V, A, phi, dphi, d, mass):
    """BDR model tunnelling
    V=bias voltage, params=[]
    A: in m^2, phi: average barrier height in eV, dphi: change in barrier height in eV,
    d: barrier width in angstrom, mass: effective electron mass as a fraction of electron rest mass

    See Brinkman et. al. J. Appl. Phys. 41 1915 (1970)
    or Tuan Comm. in Phys. 16, 1, (2006)"""
    I=3.16e10*A**2*numpy.sqrt(phi)/d*numpy.exp(-1.028*numpy.sqrt(phi)*d)*(V-0.0214*numpy.sqrt(mass)*d*dphi/phi**1.5*V**2+0.0110*mass*d**2/phi*V**3)
    return I

def FowlerNordheim(V, A, phi, d):
    """
    Simmons model tunnelling at V>phi
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    I=V/numpy.abs(V)*3.38e6*A*V**2/(d**2*phi)*numpy.exp(-0.689*phi**1.5*d/numpy.abs(V))
    return I

def TersoffHammann(V, A):
    """TersoffHamman model for tunnelling through STM tip
    V=bias voltage, params=[A]
    """
    I=A*V
    return I

def WLfit(B, s0,DS,B1,B2):
    """
    Weak localisation
    
    Args:
        B = mag. field, params=list of parameter values, s0, B1, B2
        s0 (float): zero field conductance
        DS (float): scaling parameter
        B1 (float): elastic characteristic field (B1)
        B2 (float): inelastic characteristic field (B2)

    2D WL model as per
    Wu PRL 98, 136801 (2007)
    Porter PRB 86, 064423 (2012)
    """

    e = 1.6e-19 #C
    h = 6.62e-34 #Js
    #Sets up conductivity fit array
    cond=numpy.zeros(len(B));
    if B2 == B1:
        B2 = B1*1.00001 #prevent dividing by zero

        #performs calculation for all parts
    for tt in range(len(B)):
        if B[tt] != 0: #prevent dividing by zero
            WLpt1 = digamma( 0.5 + B2 / numpy.abs(B[tt]))
            WLpt2 = digamma( 0.5 + B1 / numpy.abs(B[tt]))
        else:
            WLpt1 = ( digamma( 0.5 + B2 / numpy.abs(B[tt - 1])) + digamma( 0.5 + B2 / numpy.abs(B[tt + 1])) ) / 2
            WLpt2 = ( digamma( 0.5 + B1 / numpy.abs(B[tt - 1])) + digamma( 0.5 + B1 / numpy.abs(B[tt + 1])) ) / 2

        WLpt3 = numpy.log(B2 / B1)

    #Calculates fermi level smearing
        cond[tt] = ( e**2 / (h*numpy.pi) )*( WLpt1 - WLpt2 - WLpt3 )
    #cond = s0*cond / min(cond)
    cond = s0 + DS*cond
    return cond

def strijkers(V, omega,delta,P,Z):
    """
    strijkers(V, params):
    Args:
        V = bias voltages, params=list of parameter values, imega, delta,P and Z
        omega (float): Broadening
        delta (float): SC energy Gap
        P (float): Interface parameter
        Z (float): Current spin polarization through contact

    PCAR fitting
    Strijkers modified BTK model
        BTK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    Only using 1 delta, not modified for proximity
    """
    #   Parameters

    E = numpy.arange(-50, 50, 0.05) # Energy range in meV

    #Reflection prob arrays
    Au=numpy.zeros(len(E))
    Bu=numpy.zeros(len(E))
    Bp=numpy.zeros(len(E))

    #Conductance calculation
    """
    % For ease of calculation, epsilon = E/(sqrt(E^2 - delta^2))
    %Calculates reflection probabilities when E < or > delta
    %A denotes Andreev Reflection probability
    %B denotes normal reflection probability
    %subscript p for polarised, u for unpolarised
    %Ap is always zero as the polarised current has 0 prob for an Andreev
    %event
    """

    for ss in range(len(E)):
        if numpy.abs(E[ss])<=delta:
            Au[ss]=(delta**2)/((E[ss]**2)+(((delta**2)-(E[ss]**2))*(1+2*(Z**2))**2));
            Bu[ss] = 1-Au[ss];
            Bp[ss] = 1;
        else:
            Au[ss] = (((numpy.abs(E[ss])/(numpy.sqrt((E[ss]**2)-(delta**2))))**2)-1)/(((numpy.abs(E[ss])/(numpy.sqrt((E[ss]**2)-(delta**2)))) + (1+2*(Z**2)))**2);
            Bu[ss] = (4*(Z**2)*(1+(Z**2)))/(((numpy.abs(E[ss])/(numpy.sqrt((E[ss]**2)-(delta**2)))) + (1+2*(Z**2)))**2);
            Bp[ss] = Bu[ss]/(1-Au[ss]);

    #  Calculates reflection 'probs' for pol and unpol currents
    Guprob = 1+Au-Bu;
    Gpprob = 1-Bp;

    #Calculates pol and unpol conductance and normalises
    Gu = (1-P)*(1+(Z**2))*Guprob;
    Gp = 1*(P)*(1+(Z**2))*Gpprob;

    G = Gu + Gp;


    #Sets up gaus
    gaus=numpy.zeros(len(V));
    cond=numpy.zeros(len(V));

    #computes gaussian and integrates over all E(more or less)
    for tt in range(len(V)):
    #Calculates fermi level smearing
        gaus=(1/(2*omega*numpy.sqrt(numpy.pi)))*numpy.exp(-(((E-V[tt])/(2*omega))**2))
        cond[tt]=numpy.trapz(gaus*G,E);
    return cond


def FluchsSondheimer(t,l,p,sigma_0):
    """Evaluate a Fluchs-Sondheumer model function for conductivity.

    Args:
        t (array): Thickness values
        l (float): mean-free-path
        p (float): reflection co-efficient
        sigma_0 (float): intrinsic conductivity

    Returns:
        Reduced Resistivity

    Note:
        Expression used from: G.N.Gould and L.A. Moraga, Thin Solid Films 10 (2), 1972 pp 327-330
"""

    k=t/l

    kernel=lambda x,k:(x-x**3)*numpy.exp(-k*x)/(1-numpy.exp(-k*x))

    result=numpy.zeros(k.shape)
    for i in range(len(k)):
        v=k[i]
        result[i]=1-(3*(1-p)/(8*v))+(3*(1-p)/(2*v))*quad(kernel,0,1,v)
    return result/sigma_0