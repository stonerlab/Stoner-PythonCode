"""Stoner.Fit: Functions and lmfit.Models for fitting data
==================================================================

Functions should accept an array of x values and a number of parmeters,
they should then return an array of y values the same size as the x array.

Models are subclasses of lmfit.Model that represent the corresponding function

Please do keep documentation up to date, see other functions for documentation examples.

All the functions here defined for scipy.optimize.curve\_fit to call themm
i.e. the parameters are expanded to separate arguements, other fitting routines prefere
to have the parameters as a single list or vector. For this reason, :py:mod:`Stoner.FittingFuncs`
has aliases of these functions that use *tuple magic to make that conversion.

If you are writing new functions, please add them here first and then alias then with the parameter
list form in FittingFuncs.
"""

import numpy as _np_
from scipy.special import digamma
from lmfit import Model
from lmfit.models import LinearModel as Linear
from lmfit.models import PowerLawModel as PowerLaw
from lmfit.models import QuadraticModel as Quadratic
from lmfit.models import update_param_vals
from scipy.integrate import quad
import scipy.constants.codata as consts

try:
    from numba import jit
except ImportError:
    pass


def linear(x, intercept, slope):
    """Simple linear function"""
    return slope*x+intercept

#Linear already builtin to lmfit.models

def arrhenius(x, A, DE):
    """Arrhenius Equation without T dependendent prefactor"""
    _kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]
    return A*_np_.exp(-DE/(_kb*x))

class Arrhenius(Model):
    """Arrhenius Equation without T dependendent prefactor"""

    def __init__(self, *args, **kwargs):
        super(Arrhenius, self).__init__(arrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]

        d1,d2 = 1.,0.0
        if x is not None:
            d1,d2=_np_.polyfit(-1.0/x,_np_.log(data),1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb*d1)
        return update_param_vals(pars, self.prefix, **kwargs)

def nDimArrhenius(x, A, DE, n):
    """Arrhenius Equation without T dependendent prefactor"""
    return arrhenius(x**n, A, DE)

class NDimArrhenius(Model):
    """Arrhenius Equation without T dependendent prefactor"""

    def __init__(self, *args, **kwargs):
        super(NDimArrhenius, self).__init__(nDimArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]

        d1,d2 = 1.,0.0
        if x is not None:
            d1,d2=_np_.polyfit(-1.0/x,_np_.log(data),1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb*d1,n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)

def  modArrhenius(x, A, DE, n):
    """Arrhenius Equation with a variable T power dependent prefactor"""
    return (x**n)*Arrhenius(x, A, DE)

class ModArrhenius(Model):
    """Arrhenius Equation with a variable T power dependent prefactor"""

    def __init__(self, *args, **kwargs):
        super(ModArrhenius, self).__init__(modArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]

        d1,d2 = 1.,0.0
        if x is not None:
            d1,d2=_np_.polyfit(-1.0/x,_np_.log(data),1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb*d1,n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)

def powerLaw(x, A, k):
    """Power Law Fitting Equation"""
    return A*x**k

def quadratic(x, a,b,c):
    return a*x**2+b*x+c

def simmons(V, A, phi, d):
    """
    Simmons model tunnelling
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    I=6.2e6*A/d**2*((phi-V/2)*_np_.exp(-1.025*d*_np_.sqrt(phi-V/2))-(phi+V/2)*_np_.exp(-1.025*d*_np_.sqrt(phi+V/2)))
    return I

class Simmons(Model):
    """    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """

    def __init__(self, *args, **kwargs):
        super(Simmons, self).__init__(simmons, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12,phi=3.0,d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)

def bdr(V, A, phi, dphi, d, mass):
    """BDR model tunnelling
    V=bias voltage, params=[]
    A: in m^2, phi: average barrier height in eV, dphi: change in barrier height in eV,
    d: barrier width in angstrom, mass: effective electron mass as a fraction of electron rest mass

    See Brinkman et. al. J. Appl. Phys. 41 1915 (1970)
    or Tuan Comm. in Phys. 16, 1, (2006)"""
    I=3.16e10*A**2*_np_.sqrt(phi)/d*_np_.exp(-1.028*_np_.sqrt(phi)*d)*(V-0.0214*_np_.sqrt(mass)*d*dphi/phi**1.5*V**2+0.0110*mass*d**2/phi*V**3)
    return I

class BDR(Model):
    """BDR model tunnelling
    V=bias voltage, params=[]
    A: in m^2, phi: average barrier height in eV, dphi: change in barrier height in eV,
    d: barrier width in angstrom, mass: effective electron mass as a fraction of electron rest mass

    See Brinkman et. al. J. Appl. Phys. 41 1915 (1970)
    or Tuan Comm. in Phys. 16, 1, (2006)"""

    def __init__(self, *args, **kwargs):
        super(BDR, self).__init__(bdr, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi,dphi,d and mass values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12,phi=3.0,d=10.0,dphi=1.0,mass=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def fowlerNordheim(V, A, phi, d):
    """
    Simmons model tunnelling at V>phi
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """
    I=V/_np_.abs(V)*3.38e6*A*V**2/(d**2*phi)*_np_.exp(-0.689*phi**1.5*d/_np_.abs(V))
    return I

class FowlerNordheim(Model):
    """
    Simmons model tunnelling at V>phi
    V=bias voltage, params=[A, phi, d]
    A in m^2, phi barrier height in eV, d barrier width in angstrom

    Simmons model as in
    Simmons J. App. Phys. 34 6 1963
    """

    def __init__(self, *args, **kwargs):
        super(FowlerNordheim, self).__init__(fowlerNordheim, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12,phi=3.0,d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)

def tersoffHammann(V, A):
    """TersoffHamman model for tunnelling through STM tip
    V=bias voltage, params=[A]
    """
    I=A*V
    return I

class TersoffHammann(Model):
    """TersoffHamman model for tunnelling through STM tip
    V=bias voltage, params=[A]
    """

    def __init__(self, *args, **kwargs):
        super(TersoffHammann, self).__init__(tersoffHammann, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=_np_.mean(data/V))
        return update_param_vals(pars, self.prefix, **kwargs)

def wlfit(B, s0,DS,B1,B2):
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
    cond=_np_.zeros(len(B));
    if B2 == B1:
        B2 = B1*1.00001 #prevent dividing by zero

        #performs calculation for all parts
    for tt in range(len(B)):
        if B[tt] != 0: #prevent dividing by zero
            WLpt1 = digamma( 0.5 + B2 / _np_.abs(B[tt]))
            WLpt2 = digamma( 0.5 + B1 / _np_.abs(B[tt]))
        else:
            WLpt1 = ( digamma( 0.5 + B2 / _np_.abs(B[tt - 1])) + digamma( 0.5 + B2 / _np_.abs(B[tt + 1])) ) / 2
            WLpt2 = ( digamma( 0.5 + B1 / _np_.abs(B[tt - 1])) + digamma( 0.5 + B1 / _np_.abs(B[tt + 1])) ) / 2

        WLpt3 = _np_.log(B2 / B1)

    #Calculates fermi level smearing
        cond[tt] = ( e**2 / (h*_np_.pi) )*( WLpt1 - WLpt2 - WLpt3 )
    #cond = s0*cond / min(cond)
    cond = s0 + DS*cond
    return cond

class WLfit(Model):
    """
    Weak localisation

    def wlfit(B, s0,DS,B1,B2):
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

    def __init__(self, *args, **kwargs):
        super(WLfit, self).__init__(wlfit, *args, **kwargs)

    def guess(self, data, B=None, **kwargs):
        s0,DS,B1,B2=1.0,1.0,1.0,1.0
        if B is not None:
            zpos=_np_.argmin(_np_.abs(B))
            s0=data[zpos]
            B1=_np_.max(B)/2.0
            B2=B1
            DS=1.0
        pars = self.make_params(s0=s0,DS=DS,B1=B1,B2=B2)
        return update_param_vals(pars, self.prefix, **kwargs)

@jit
def _strijkers_core(V, omega,delta,P,Z):
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

    E = _np_.arange(2*_np_.min(V), 2*_np_.max(V), 0.025) # Energy range in meV

    #Reflection prob arrays
    Au=_np_.zeros(len(E))
    Bu=_np_.zeros(len(E))
    Bp=_np_.zeros(len(E))

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

    Au1=(delta**2)/((E**2)+(((delta**2)-(E**2))*(1+2*(Z**2))**2))
    Au2=(((_np_.abs(E)/(_np_.sqrt((E**2)-(delta**2))))**2)-1)/(((_np_.abs(E)/(_np_.sqrt((E**2)-(delta**2)))) + (1+2*(Z**2)))**2)
    Bu1 = 1-Au1
    Bu2 = (4*(Z**2)*(1+(Z**2)))/(((_np_.abs(E)/(_np_.sqrt((E**2)-(delta**2)))) + (1+2*(Z**2)))**2)
    Bp1 = _np_.ones(len(E))
    Bp2 = Bu2/(1-Au2);

    Au=_np_.where(_np_.abs(E)<=delta,Au1,Au2)
    Bu=_np_.where(_np_.abs(E)<=delta,Bu1,Bu2)
    Bp=_np_.where(_np_.abs(E)<=delta,Bp1,Bp2)

    #  Calculates reflection 'probs' for pol and unpol currents
    Guprob = 1+Au-Bu;
    Gpprob = 1-Bp;

    #Calculates pol and unpol conductance and normalises
    Gu = (1-P)*(1+(Z**2))*Guprob;
    Gp = 1*(P)*(1+(Z**2))*Gpprob;

    G = Gu + Gp;


    #Sets up gaus
    gaus=_np_.zeros(len(V));
    cond=_np_.zeros(len(V));

    #computes gaussian and integrates over all E(more or less)
    for tt in range(len(V)):
    #Calculates fermi level smearing
        gaus=(1/(2*omega*_np_.sqrt(_np_.pi)))*_np_.exp(-(((E-V[tt])/(2*omega))**2))
        cond[tt]=_np_.trapz(gaus*G,E);
    return cond

def strijkers(V, omega,delta,P,Z):
    return _strijkers_core(V, omega,delta,P,Z)

class Strijkers(Model):
    """
    strijkers(V, params):
    Args:
        V = bias voltages, params=list of parameter values, imega, delta,P and Z
        omega (float): Broadening
        delta (float): SC energy Gap
        P (float): Interface parameter
        Z (float): Current spin polarization through contact

    PCAR fitting
    Strijkers modified BTK model - BTK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    Only using 1 delta, not modified for proximity
    """
    def __init__(self, *args, **kwargs):
        super(Strijkers, self).__init__(strijkers, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Guess starting values for a good Nb contact to a ferromagnet at 4.2K"""
        pars = self.make_params(omega=0.36,delta=1.50,P=0.42,Z=0.15)
        return update_param_vals(pars, self.prefix, **kwargs)

def fluchsSondheimer(t,l,p,sigma_0):
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

    kernel=lambda x,k:(x-x**3)*_np_.exp(-k*x)/(1-_np_.exp(-k*x))

    result=_np_.zeros(k.shape)
    for i in range(len(k)):
        v=k[i]
        result[i]=1-(3*(1-p)/(8*v))+(3*(1-p)/(2*v))*quad(kernel,0,1,v)
    return result/sigma_0

class FluchsSondheimer(Model):
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
    def __init__(self, *args, **kwargs):
        super(FluchsSondheimer, self).__init__(fluchsSondheimer, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):
        """Guess some starting values - not very clever"""
        pars = self.make_params(l=10.0,p=0.5,sigma_0=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)

def _bgintegrand(x,n):
    return x**n/((_np_.exp(x)-1)*(1-_np_.exp(-x)))

def blochGrueneisen(T,thetaD,rho0,A,n):
    """BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Returns:
        Evaluation of the BlochGrueneisen function for R(T)"""
    ret=_np_.zeros(T.shape)
    for i,t in enumerate(T):
        intg=quad(_bgintegrand,0,thetaD/(t),(n,))[0]
        ret[i]=rho0+A*(t/thetaD)**n*intg
    return ret

class BlochGrueneisen(Model):
    """BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Returns:
        Evaluation of the BlochGrueneisen function for R(T)"""
    def __init__(self, *args, **kwargs):
        super(BlochGrueneisen, self).__init__(blochGrueneisen, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):
        """Guess some starting values - not very clever"""
        pars = self.make_params(thetaD=900,rho0=0.01,A=0.2,n=5.0)
        return update_param_vals(pars, self.prefix, **kwargs)
        
def langevin(H,M_s,m,T):
    """"The Langevin function for paramagnetic M-H loops/
    
    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float) is the moment of a cluster
        T (float): Temperature
        
    Rerturns:
        Magnetic Momemnts (array).
        
    The Langevin Function is $\\coth(\\frac{\\mu_0HM_s}{k_BT})-\\frac{k_BT}{\\mu_0HM_s}$
    """
    from scipy.constants import k,mu_0
    
    x=mu_0*m*H/(k*T)
    return M_s*_np_.coth(x)-1.0/x
    
class Langevin(Model):
    """"The Langevin function for paramagnetic M-H loops/
    
    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float): is the moment of a single cluster
        T (float): Temperature
        
    Rerturns:
        Magnetic Momemnts (array).
        
    The Langevin Function is $\\coth(\\frac{\\mu_0HM_s}{k_BT})-\\frac{k_BT}{\\mu_0HM_s}$
    """
    def __init__(self, *args, **kwargs):
        super(Langevin, self).__init__(langevin, *args, **kwargs)

    def guess(self, data, h=None, **kwargs):
        """Guess some starting values.
        
        M_s is taken as half the difference of the range of thew M data, 
        we can find m/T from the susceptibility chi= M_s \mu_o m / kT,"""
        M_s=(_np_.max(data)-_np_.min(data))/2.0
        if h is not None:
            from scipy.signal import savgol_filter
            from scipy.constants import k,mu_0,e,electron_mass,hbar            
            d=_np_.sort(_np_.row_stack((h,data)))
            dd=savgol_filter(d,7,1)
            yd=dd[1]/dd[0]
            chi=_np_.interp(_np_.array([0]),d[0],yd)[0]
            mT=chi/M_s*(k/mu_0)
            #Assume T=150K for no good reason
            m=mT*150
        else:
            m=1E6*(e*hbar)/(2*electron_mass) # guess 1 million Bohr Magnetrons
        T=150
        pars = self.make_params(M_s=M_s,m=m,T=T)
        return update_param_vals(pars, self.prefix, **kwargs)
        
   



