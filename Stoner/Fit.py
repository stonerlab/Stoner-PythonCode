"""Stoner.Fit: Functions and lmfit.Models for fitting data
==================================================================

Functions should accept an array of x values and a number of parmeters,
they should then return an array of y values the same size as the x array.

Models are subclasses of lmfit.Model that represent the corresponding function

Please do keep documentation up to date, see other functions for documentation examples.

All the functions here defined for scipy.optimize.curve\_fit to call themm
i.e. the parameters are expanded to separate arguements.
"""

from Stoner.compat import *
from Stoner.Core import DataFile
import numpy as _np_
from scipy.special import digamma
from lmfit import Model
from lmfit.models import LinearModel as Linear
from lmfit.models import PowerLawModel as PowerLaw
from lmfit.models import QuadraticModel as Quadratic
from lmfit.models import update_param_vals
from scipy.integrate import quad
import scipy.constants.codata as consts
if python_v3:
    import configparser as ConfigParser
else:
    import ConfigParser

try: # numba is an optional dependency
    from numba import jit
except ImportError:
    def jit(func):
        """Null decorator function."""
        return func

def _get_model_(model):
    """Utility meothd to manage creating an lmfit.Model.

    Args:
        model (str, callable, Model): The model to be setup.

    Returns:
        An llmfit.Model instance

    model can be of several different types that determine what to do:

    -   A string. In which ase it should be a fully qualified name of a function or class to be imported.
        The part after the final period will be assumed to be the name and the remainder the module to be
        imported.
    -   A callable object. In this case the callable will be passed to the constructor of Model and a fresh
        Model instance is constructed
    -   A subclass of lmfit.Model - in whcih case it is instantiated.
    -   A Model instance - in which case no further action is necessary.

    """
    if isinstance(model,string_types): #model is a string, so we;ll try importing it now
        parts=model.split(".")
        model=parts[-1]
        module=".".join(parts[:-1])
        model=__import__(module,globals(),locals(),(model)).__getattribute__(model)
    if type(model).__name__=="type" and issubclass(model,Model): # ok perhaps we've got a model class rather than an instance
        model=model()
    if not isinstance(model,Model) and callable(model): # Ok, wrap the callable in a model
        model=Model(model)
    if not isinstance(model,Model):
        raise TypeError("model {} is not an instance of llmfit.Model".format(model.__name__))
    return model


def linear(x, intercept, slope):
    """Simple linear function"""
    return slope * x + intercept

def cfg_data_from_ini(inifile,filename=None):
    """Read an inifile and load and configure a DataFile from it.

    Args:
        inifile (str or file): Path to the ini file to be read.

    Keyword Arguments:
        filename (strig,boolean or None): File to load that contains the data.

    Returns:
        An instance of :py:class:`Stoner.Util.Data` with data loaded and columns configured.

    The inifile should contain a [Data] section that contains the following keys:

    -  **type (str):** optional name of DataFile subclass to import.
    -  **filename (str or boolean):** optionally used if *filename* parameter is None.
    - **xcol (column index):** defines the x-column data for fitting.
    - **ycol (column index):** defines the y-column data for fitting.
    - **yerr (column index):** Optional column with uncertainity values for the data
    """
    config = ConfigParser.SafeConfigParser()
    from Stoner.Util import Data
    if isinstance(inifile,string_types):
        config.read(inifile)
    elif isinstance(inifile,file):
        config.readfp(inifile)
    if not config.has_section("Data"):
        raise RuntimeError("Configuration file lacks a [Data] section to describe data.")

    if config.has_option("Data","type"):
        typ=config.get("Data","type").split(".")
        typ_mod=".".join(typ[:-1])
        typ=typ[-1]
        typ = __import__(typ_mod,fromlist=[typ]).__getattribute__(typ)
    else:
        typ = None
    data=Data()
    if filename is None:
        if not config.has_option("Data","filename"):
            filename=False
        else:
            filename=config.get("Data","filename")
            if filename in ["False","True"]:
                filename=bool(filename)
    data.load(filename,auto_load=False,filetype=typ)
    cols={"x":0,"y":1,"e":None} # Defaults

    for c in ["x","y","e"]:
        if not config.has_option("Data",c):
            pass
        else:
            try:
                cols[c]=config.get("Data",c)
                cols[c]=int(cols[c])
            except ValueError:
                pass
        if cols[c] is None:
            del cols[c]

    data.setas(**cols)
    return data

def cfg_model_from_ini(inifile,model=None,data=None):
    """Utility function to configure an lmfit Model from an inifile.

    Args:
        inifile (str or file): Path to the ini file to be read.

    Keyword Arguments:
        model (str, callable, lmfit.Model instance or sub-class or None): What to use as a model function.
        data (DataFile): if supplied, the details of the parameter hints and labels and units are included in the data's metadata.

    Returns:
        An llmfit.Model,, a 2D array of starting values for each parameter

    model can be of several different types that determine what to do:

    -   A string. In which ase it should be a fully qualified name of a function or class to be imported.
        The part after the final period will be assumed to be the name and the remainder the module to be
        imported.
    -   A callable object. In this case the callable will be passed to the constructor of Model and a fresh
        Model instance is constructed
    -   A subclass of lmfit.Model - in whcih case it is instantiated.
    -   A Model instance - in which case no further action is necessary.

    The returned model is configured with parameter hints for fitting with. The second return value is
    a 2D array which lists the starting values for one or more fits. If the inifile describes mapping out
    the :math:`\\Chi^2` as a function of the parameters, then this array has a separate row for each iteration.


    """
    config = ConfigParser.SafeConfigParser()
    if isinstance(inifile,string_types):
        config.read(inifile)
    elif isinstance(inifile,file):
        config.readfp(inifile)

    if model is None: # Check to see if config file specified a model
        try:
            model=config.get("Options","model")
        except:
            raise RuntimeError("Model is notspecifed either as keyword argument or in inifile")
    model=_get_model_(model)
    if config.has_option("option","prefix"):
        prefix = config.get("option","prefix")
    else:
        prefix = model.__class__.__name__
    prefix += ":"
    vals=[]
    for p in model.param_names:
        if not config.has_section(p):
            raise RuntimeError("Config file does not have a section for parameter {}".format(p))
        keys={"vary":bool,"value":float,"min":float,"max":float,"expr":str,"step":float,"label":str,"units":str}
        kargs=dict()
        for k in keys:
           if config.has_option(p,k):
               if keys[k]==bool:
                   kargs[k]=config.getboolean(p,k)
               elif keys[k]==float:
                   kargs[k]=config.getfloat(p,k)
               elif keys[k]==str:
                    kargs[k]=config.get(p,k)
        if isinstance(data,DataFile): # stuff the parameter hint data into metadata
            for k in keys: # remove keywords not needed
                if k in kargs:
                    data["{}{} {}".format(prefix,p,k)]=kargs[k]
            if "lmfit.prerfix" in data:
                data["lmfit.prefix"].append(prefix)
            else:
                data["lmfit.prefix"]=[prefix]
        if "step" in kargs: #We use step for creating a chi^2 mapping, but not for a parameter hint
            step=kargs.pop("step")
            if "vary" in kargs and "min" in kargs and "max" in kargs and not kargs["vary"]: # Make chi^2?
                vals.append(_np_.arange(kargs["min"],kargs["max"]+step/10,step))
            else: # Nope, just make a single value step here
                vals.append(_np_.array(kargs["value"]))
        else: # Nope, just make a single value step here
            vals.append(_np_.array(kargs["value"]))
        kargs={k:kargs[k] for k in kargs if k in ["value","max","min","vary"]}
        model.set_param_hint(p,**kargs) # set the model parameter hint
    msh=_np_.meshgrid(*vals) # make a mesh of all possible parameter values to test
    msh=[m.ravel() for m in msh] # tidy it up and combine into one 2D array
    msh=_np_.column_stack(msh)
    return model,msh

def arrhenius(x, A, DE):
    """Arrhenius Equation without T dependendent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=A\\exp\\left(\\frac{-\\Delta E}{k_B x}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """
    _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]
    return A * _np_.exp(-DE / (_kb * x))


class Arrhenius(Model):
    """Arrhenius Equation without T dependendent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=A\\exp\\left(\\frac{-\\Delta E}{k_B x}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """

    def __init__(self, *args, **kwargs):
        super(Arrhenius, self).__init__(arrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]

        d1, d2 = 1., 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb * d1)
        return update_param_vals(pars, self.prefix, **kwargs)


def nDimArrhenius(x, A, DE, n):
    """Arrhenius Equation without T dependendent prefactor for various dimensions.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The dimensionalirty of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=A\\exp\\left(\\frac{-\\Delta E}{k_B x^n}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """
    return arrhenius(x ** n, A, DE)


class NDimArrhenius(Model):
    """Arrhenius Equation without T dependendent prefactor for various dimensions.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The dimensionalirty of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=A\\exp\\left(\\frac{-\\Delta E}{k_B x^n}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """

    def __init__(self, *args, **kwargs):
        super(NDimArrhenius, self).__init__(nDimArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]

        d1, d2 = 1., 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb * d1, n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def modArrhenius(x, A, DE, n):
    """Arrhenius Equation with a variable T power dependent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The exponent of the temperature pre-factor of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=Ax^n\\exp\\left(\\frac{-\\Delta E}{k_B x}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """
    return (x ** n) * Arrhenius(x, A, DE)


class ModArrhenius(Model):
    """Arrhenius Equation with a variable T power dependent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The exponent of the temperature pre-factor of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\\tau=Ax^n\\exp\\left(\\frac{-\\Delta E}{k_B x}\\right)` where
    :math:`k_B` is Boltzmann's constant.
    """

    def __init__(self, *args, **kwargs):
        super(ModArrhenius, self).__init__(modArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]

        d1, d2 = 1., 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb * d1, n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def powerLaw(x, A, k):
    """Power Law Fitting Equation.

    Args:
        x (array): Input data
        A (float): Prefactor
        k (float): Pwoer

    Return:
        Power law.

    :math:`p=Ax^k`"""
    return A * x ** k


def quadratic(x, a, b, c):
    """A Simple quadratic fitting function.

    Args:
        x (aray): Input data
        a (float): Quadratic term co-efficient
        b (float): Linear term co-efficient
        c (float): Constant offset term

    Returns:
        Array of data.

    :math:`y=ax^2+bx+c`"""
    return a * x ** 2 + b * x + c


def simmons(V, A, phi, d):
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
    """
    I = 6.2e6 * A / d ** 2 * ((phi - V / 2) * _np_.exp(-1.025 * d * _np_.sqrt(phi - V / 2)) -
                              (phi + V / 2) * _np_.exp(-1.025 * d * _np_.sqrt(phi + V / 2)))
    return I


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
    """

    def __init__(self, *args, **kwargs):
        super(Simmons, self).__init__(simmons, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12, phi=3.0, d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


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

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)"""
    I = 3.16e10 * A ** 2 * _np_.sqrt(phi) / d * _np_.exp(-1.028 * _np_.sqrt(phi) * d) * (
        V - 0.0214 * _np_.sqrt(mass) * d * dphi / phi ** 1.5 * V ** 2 + 0.0110 * mass * d ** 2 / phi * V ** 3)
    return I


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

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)"""

    def __init__(self, *args, **kwargs):
        super(BDR, self).__init__(bdr, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi,dphi,d and mass values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12, phi=3.0, d=10.0, dphi=1.0, mass=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def fowlerNordheim(V, A, phi, d):
    """Fowler Nordhiem Model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in m^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Tunneling rate according to Fowler Nordheim model.
    """
    I = V / _np_.abs(V) * 3.38e6 * A * V ** 2 / (d ** 2 * phi) * _np_.exp(-0.689 * phi ** 1.5 * d / _np_.abs(V))
    return I


class FowlerNordheim(Model):
    """Fowler Nordhiem Model of electron tunnelling.

    Args:
        V (array): Bias voltage
        A (float): Area of barrier in m^2
        phi (float): barrier height in eV
        d (float): barrier width in angstroms

    Return:
        Tunneling rate according to Fowler Nordheim model.
    """

    def __init__(self, *args, **kwargs):
        super(FowlerNordheim, self).__init__(fowlerNordheim, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1E-12, phi=3.0, d=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def tersoffHammann(V, A):
    """TersoffHamman model for tunnelling through STM tip.

    Args:
        V (array): bias voltage
        A (float): Tip conductance

    Return:
        A linear fit.
    """
    I = A * V
    return I


class TersoffHammann(Model):
    """TersoffHamman model for tunnelling through STM tip.

    Args:
        V (array): bias voltage
        A (float): Tip conductance

    Return:
        A linear fit.
    """

    def __init__(self, *args, **kwargs):
        super(TersoffHammann, self).__init__(tersoffHammann, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=_np_.mean(data / V))
        return update_param_vals(pars, self.prefix, **kwargs)


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
    """

    e = 1.6e-19  #C
    h = 6.62e-34  #Js
    #Sets up conductivity fit array
    cond = _np_.zeros(len(B))
    if B2 == B1:
        B2 = B1 * 1.00001  #prevent dividing by zero

#performs calculation for all parts
    for tt in range(len(B)):
        if B[tt] != 0:  #prevent dividing by zero
            WLpt1 = digamma(0.5 + B2 / _np_.abs(B[tt]))
            WLpt2 = digamma(0.5 + B1 / _np_.abs(B[tt]))
        else:
            WLpt1 = (digamma(0.5 + B2 / _np_.abs(B[tt - 1])) + digamma(0.5 + B2 / _np_.abs(B[tt + 1]))) / 2
            WLpt2 = (digamma(0.5 + B1 / _np_.abs(B[tt - 1])) + digamma(0.5 + B1 / _np_.abs(B[tt + 1]))) / 2

        WLpt3 = _np_.log(B2 / B1)

        #Calculates fermi level smearing
        cond[tt] = (e ** 2 / (h * _np_.pi)) * (WLpt1 - WLpt2 - WLpt3)
    #cond = s0*cond / min(cond)
    cond = s0 + DS * cond
    return cond


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
    """

    def __init__(self, *args, **kwargs):
        super(WLfit, self).__init__(wlfit, *args, **kwargs)

    def guess(self, data, B=None, **kwargs):
        s0, DS, B1, B2 = 1.0, 1.0, 1.0, 1.0
        if B is not None:
            zpos = _np_.argmin(_np_.abs(B))
            s0 = data[zpos]
            B1 = _np_.max(B) / 2.0
            B2 = B1
            DS = 1.0
        pars = self.make_params(s0=s0, DS=DS, B1=B1, B2=B2)
        return update_param_vals(pars, self.prefix, **kwargs)


@jit
def _strijkers_core(V, omega, delta, P, Z):
    """strijkers Model for point-contact Andreev Reflection Spectroscopy
    Args:
        V = bias voltages, params=list of parameter values, imega, delta,P and Z
        omega (float): Broadening
        delta (float): SC energy Gap
        P (float): Interface parameter
        Z (float): Current spin polarization through contact

    Return:
        Conductance vs bias data.

    .. note::

       PCAR fitting Strijkers modified BTK model TK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    This version only uses 1 delta, not modified for proximity
    """
    #   Parameters

    E = _np_.arange(2 * _np_.min(V), 2 * _np_.max(V), 0.025)  # Energy range in meV

    #Reflection prob arrays
    Au = _np_.zeros(len(E))
    Bu = _np_.zeros(len(E))
    Bp = _np_.zeros(len(E))

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

    Au1 = (delta ** 2) / ((E ** 2) + (((delta ** 2) - (E ** 2)) * (1 + 2 * (Z ** 2)) ** 2))
    Au2 = (((_np_.abs(E) / (_np_.sqrt((E ** 2) - (delta ** 2)))) ** 2) - 1) / (((_np_.abs(E) /
                                                                                 (_np_.sqrt((E ** 2) - (delta ** 2)))) +
                                                                                (1 + 2 * (Z ** 2))) ** 2)
    Bu1 = 1 - Au1
    Bu2 = (4 * (Z ** 2) * (1 + (Z ** 2))) / (((_np_.abs(E) / (_np_.sqrt((E ** 2) - (delta ** 2)))) + (1 + 2 *
                                                                                                      (Z ** 2))) ** 2)
    Bp1 = _np_.ones(len(E))
    Bp2 = Bu2 / (1 - Au2)

    Au = _np_.where(_np_.abs(E) <= delta, Au1, Au2)
    Bu = _np_.where(_np_.abs(E) <= delta, Bu1, Bu2)
    Bp = _np_.where(_np_.abs(E) <= delta, Bp1, Bp2)

    #  Calculates reflection 'probs' for pol and unpol currents
    Guprob = 1 + Au - Bu
    Gpprob = 1 - Bp

    #Calculates pol and unpol conductance and normalises
    Gu = (1 - P) * (1 + (Z ** 2)) * Guprob
    Gp = 1 * (P) * (1 + (Z ** 2)) * Gpprob

    G = Gu + Gp

    #Sets up gaus
    gaus = _np_.zeros(len(V))
    cond = _np_.zeros(len(V))

    #computes gaussian and integrates over all E(more or less)
    for tt in range(len(V)):
        #Calculates fermi level smearing
        gaus = (1 / (2 * omega * _np_.sqrt(_np_.pi))) * _np_.exp(-(((E - V[tt]) / (2 * omega)) ** 2))
        cond[tt] = _np_.trapz(gaus * G, E)
    return cond


def strijkers(V, omega, delta, P, Z):
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
    """
    return _strijkers_core(V, omega, delta, P, Z)


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
    """

    def __init__(self, *args, **kwargs):
        super(Strijkers, self).__init__(strijkers, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):
        """Guess starting values for a good Nb contact to a ferromagnet at 4.2K"""
        pars = self.make_params(omega=0.36, delta=1.50, P=0.42, Z=0.15)
        return update_param_vals(pars, self.prefix, **kwargs)


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
"""

    k = t / l

    kernel = lambda x, k: (x - x ** 3) * _np_.exp(-k * x) / (1 - _np_.exp(-k * x))

    result = _np_.zeros(k.shape)
    for i in range(len(k)):
        v = k[i]
        result[i] = 1 - (3 * (1 - p) / (8 * v)) + (3 * (1 - p) / (2 * v)) * quad(kernel, 0, 1, v)
    return result / sigma_0


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
    """

    def __init__(self, *args, **kwargs):
        super(FluchsSondheimer, self).__init__(fluchsSondheimer, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):
        """Guess some starting values - not very clever"""
        pars = self.make_params(l=10.0, p=0.5, sigma_0=10.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def _bgintegrand(x, n):
    return x ** n / ((_np_.exp(x) - 1) * (1 - _np_.exp(-x)))


def blochGrueneisen(T, thetaD, rho0, A, n):
    """BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Return:
        Evaluation of the BlochGrueneisen function for R(T)"""
    ret = _np_.zeros(T.shape)
    for i, t in enumerate(T):
        intg = quad(_bgintegrand, 0, thetaD / (t), (n, ))[0]
        ret[i] = rho0 + A * (t / thetaD) ** n * intg
    return ret


class BlochGrueneisen(Model):
    """BlochGrueneiseen Function for fitting R(T).

    Args:
        T (array): Temperature Values to fit
        thetaD (float): Debye Temperature
        rho0 (float): Residual resisitivity
        A (float): scattering scaling factor
        n (float): Exponent term

    Return:
        Evaluation of the BlochGrueneisen function for R(T)"""

    def __init__(self, *args, **kwargs):
        super(BlochGrueneisen, self).__init__(blochGrueneisen, *args, **kwargs)

    def guess(self, data, t=None, **kwargs):
        """Guess some starting values - not very clever"""
        pars = self.make_params(thetaD=900, rho0=0.01, A=0.2, n=5.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def langevin(H, M_s, m, T):
    """"The Langevin function for paramagnetic M-H loops/

    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float) is the moment of a cluster
        T (float): Temperature

    Returns:
        Magnetic Momemnts (array).

    The Langevin Function is :math:`\\coth(\\frac{\\mu_0HM_s}{k_BT})-\\frac{k_BT}{\\mu_0HM_s}`.
    """
    from scipy.constants import k, mu_0

    x = mu_0 * m * H / (k * T)
    return M_s * _np_.coth(x) - 1.0 / x


class Langevin(Model):
    """"The Langevin function for paramagnetic M-H loops/

    Args:
        H (array): The applied magnetic field
        M_s (float): Saturation magnetisation
        m (float): is the moment of a single cluster
        T (float): Temperature

    Returns:
        Magnetic Momemnts (array).

    The Langevin Function is :math:`\\coth(\\frac{\\mu_0HM_s}{k_BT})-\\frac{k_BT}{\\mu_0HM_s}`.
    """

    def __init__(self, *args, **kwargs):
        super(Langevin, self).__init__(langevin, *args, **kwargs)

    def guess(self, data, h=None, **kwargs):
        """Guess some starting values.

        M_s is taken as half the difference of the range of thew M data,
        we can find m/T from the susceptibility chi= M_s \mu_o m / kT,"""
        M_s = (_np_.max(data) - _np_.min(data)) / 2.0
        if h is not None:
            from scipy.signal import savgol_filter
            from scipy.constants import k, mu_0, e, electron_mass, hbar
            d = _np_.sort(_np_.row_stack((h, data)))
            dd = savgol_filter(d, 7, 1)
            yd = dd[1] / dd[0]
            chi = _np_.interp(_np_.array([0]), d[0], yd)[0]
            mT = chi / M_s * (k / mu_0)
            #Assume T=150K for no good reason
            m = mT * 150
        else:
            m = 1E6 * (e * hbar) / (2 * electron_mass)  # guess 1 million Bohr Magnetrons
        T = 150
        pars = self.make_params(M_s=M_s, m=m, T=T)
        return update_param_vals(pars, self.prefix, **kwargs)


def vftEquation(x, A, DE, x_0):
    r"""Vogel-Flucher-Tammann (VFT) Equation without T dependendent prefactor.

    Args:
        x (float): Temperature in K
        A (float): Prefactror (not temperature dependent)
        DE (float): Energy barrier in eV
        x_0 (float): Offset temeprature in K

    Return:
        Rates according the VFT equation.

    The VFT equation is defined as as :math:`\tau = A\exp\left(\frac{DE}{x-x_0}\right)` and represents
    a modifed form of the Arrenhius distribution with a freezing point of :math:`x_0`.
    """
    _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]
    return A * _np_.exp(-DE / (_kb * (x - x_0)))


class VFTEquation(Model):
    r"""Vogel-Flucher-Tammann (VFT) Equation without T dependendent prefactor.

    Args:
        x (array): Temperature in K
        A (float): Prefactror (not temperature dependent)
        DE (float): Energy barrier in eV
        x_0 (float): Offset temeprature in K

    Return:
        Rates according the VFT equation.

    The VFT equation is defined as as :math:`\tau = A\exp\left(\frac{DE}{x-x_0}\right)` and represents
    a modifed form of the Arrenhius distribution with a freezing point of :math:`x_0`.
    """

    def __init__(self, *args, **kwargs):
        super(VFTEquation, self).__init__(vftEquation, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        _kb = consts.physical_constants['Boltzmann constant'][0] / consts.physical_constants['elementary charge'][0]

        d1, d2, x0 = 1., 0.0, 1.0
        if x is not None:
            x0 = x[_np_.argmin(_np_.abs(data))]
            d1, d2 = _np_.polyfit(-1.0 / (x - x0), _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb * d1, x_0=x0)
        return update_param_vals(pars, self.prefix, **kwargs)


def stretchedExp(x, A, beta, x_0):
    """A stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\\exp\\left[\\left(\\frac{-x}{x_0}\\right)^\\beta\\right]`.
    """
    return A * _np_.exp(-(x / x_0) ** beta)


class StretchedExp(Model):
    """A stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\\exp\\left[\\left(\\frac{-x}{x_0}\\right)^\\beta\\right]`.
    """

    def __init__(self, *args, **kwargs):
        super(StretchedExp, self).__init__(stretchedExp, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):

        A, beta, x0 = 1.0, 1.0, 1.0
        if x is not None:
            A = data[_np_.argmin(_np_.abs(x))]
            x=_np_.log(x)
            y=_np_.log(-_np_.log(data / A))
            d=_np_.column_stack((x,y))
            d=d[~_np_.isnan(d).any(axis=1)]
            d=d[~_np_.isinf(d).any(axis=1)]
            d1, d2 = _np_.polyfit(d[:,0],d[:,1], 1)
            beta = d1
            x0 = _np_.exp(d2 / beta)
        pars = self.make_params(A=A, beta=beta, x_0=x0)
        return update_param_vals(pars, self.prefix, **kwargs)

def kittelEquation(H,gamma,M_s,H_k):
    """Kittel Equation for finding ferromagnetic resonance peak in frequency with field.

    Args:
        H (array): Magnetic fields in A/m
        gamma (float): gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz
    """
    return (consts.mu0*gamme/(2*_np_.pi))*_np_.sqrt((H+H_k)*(H+H_k+M_s))

class KittelEquation(Model):
    """Kittel Equation for finding ferromagnetic resonance peak in frequency with field.

    Args:
        H (array): Magnetic fields in A/m
        gamma (float): gyromagnetic radius
        M_s (float): Magnetisation of sample in A/m
        H_k (float): Anisotropy field term (including demagnetising factors) in A/m

    Returns:
        Reesonance peak frequencies in Hz
    """
    def __init__(self, *args, **kwargs):
        super(KittelEquation, self).__init__(kittelEquation, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""
        M_s=(_np_.pi*data/consts.mu0)/H-H

        pars = self.make_params(gamma=2, M_s=M_s, H_k=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


