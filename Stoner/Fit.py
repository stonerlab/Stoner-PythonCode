"""Stoner.Fit: Functions and lmfit.Models for fitting data.

Functions should accept an array of x values and a number of parmeters,
they should then return an array of y values the same size as the x array.

Models are subclasses of lmfit.Model that represent the corresponding function

Please do keep documentation up to date, see other functions for documentation examples.

All the functions here defined for scipy.optimize.curve\_fit to call themm
i.e. the parameters are expanded to separate arguements.
"""
__all__ = [
    "Arrhenius",
    "BDR",
    "BlochGrueneisen",
    "FMR_Power",
    "FluchsSondheimer",
    "FowlerNordheim",
    "Inverse_Kittel",
    "KittelEquation",
    "Langevin",
    "Linear",
    "Lorentzian_diff",
    "make_model",
    "ModArrhenius",
    "NDimArrhenius",
    "PowerLaw",
    "Quadratic",
    "Simmons",
    "StretchedExp",
    "Strijkers",
    "TersoffHammann",
    "VFTEquation",
    "WLfit",
    "_strijkers_core",
    "arrhenius",
    "bdr",
    "blochGrueneisen",
    "cfg_data_from_ini",
    "cfg_model_from_ini",
    "fluchsSondheimer",
    "fmr_power",
    "fowlerNordheim",
    "inverse_kittel",
    "kittelEquation",
    "langevin",
    "linear",
    "lorentzian_diff",
    "modArrhenius",
    "nDimArrhenius",
    "powerLaw",
    "quadratic",
    "simmons",
    "stretchedExp",
    "strijkers",
    "vftEquation",
    "wlfit",
]
import Stoner.Core as _SC_
from .compat import python_v3, string_types
from . import Data
from functools import wraps
import numpy as _np_
from collections import Mapping
from io import IOBase
from scipy.special import digamma

try:
    from lmfit import Model
    from lmfit.models import LinearModel as _Linear  # NOQA pylint: disable=unused-import
    from lmfit.models import PowerLawModel as _PowerLaw  # NOQA pylint: disable=unused-import
    from lmfit.models import QuadraticModel as _Quadratic  # NOQA pylint: disable=unused-import
    from lmfit.models import update_param_vals
except ImportError:
    Model = object
    _Linear = object
    _PowerLaw = object
    _Quadratic = object
    update_param_vals = None

from scipy.integrate import quad
import scipy.constants.codata as consts
import scipy.constants as cnst

try:
    if python_v3:
        from configparser import ConfigParser as SafeConfigParser
    else:
        from ConfigParser import SafeConfigParser
except ImportError:
    SafeConfigParser = None

try:  # numba is an optional dependency
    from numba import jit, float64
except ImportError:

    def jit(func, *args):
        """Null decorator function."""
        return func

    class _dummy(object):
        """A class that does nothing so that float64 can be an instance of it safely."""

        def __call__(self, *args):
            return self

        def __getitem__(self, *args):
            return self

    float64 = _dummy()


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
    if isinstance(model, string_types):  # model is a string, so we;ll try importing it now
        parts = model.split(".")
        model = parts[-1]
        module = ".".join(parts[:-1])
        model = __import__(module, globals(), locals(), (model)).__getattribute__(model)
    if type(model).__name__ == "type" and issubclass(
        model, Model
    ):  # ok perhaps we've got a model class rather than an instance
        model = model()
    if not isinstance(model, Model) and callable(model):  # Ok, wrap the callable in a model
        model = Model(model)
    if not isinstance(model, Model):
        raise TypeError("model {} is not an instance of llmfit.Model".format(model.__name__))
    return model


def make_model(model_func):
    """A decorator that turns a function into an lmfit model.

    Notes:
        The function being wrapped into the model should have the form::

            def model_func(x_data,*parameters):
                ....

        (i.e. similar to what :py:func:`scipy.optimize.curve_fit` expects). The resulting
        class is a sub-class of :py:class:`lmfit.Model` but also adds a class method
        :py:method:`_ModelDectorator.guesser` which can be used as a decorator to convert another function into
        a :py:meth:`lmfit.Model.guess` method. If using this decorator, the function that does the guessing should
        take the form::

            def guesser_function(y_data,x=x_data,**kargs):
                return (param_1,param_2,....,pram_n)

        Similarly, the class provides a :py:meth:`_ModelDecorator.hinter` decorator which can be used to mark a function
        as something that can generate prameter hints for the model. In this case the function should take the form::

            def hinter(**kwargs):
                return {"param_1":{"max":max_val,"min":min_value,"value":start_value},"param_2":{.....}}

        Finally the new model_func class can be instantiated or just passed to :py:meth:`Data.lmfit` etc. directly.
    """

    class _ModelDecorator(Model):

        __doc__ = model_func.__doc__

        def __init__(self, *args, **kargs):
            super(_ModelDecorator, self).__init__(model_func, *args, **kargs)
            if hasattr(self, "_limits"):
                for param, limit in self._limits().items():
                    self.set_param_hint(param, **limit)
            self.__name__ = self.func.__name__

        def guess(self, y, x=None):
            """A default parameter guess method that just guesses 1.0 for everything like :py:func:`scipy.optimize.curve_fit` does."""
            return _np_.ones(len(self.param_names))

        @classmethod
        def hinter(cls, func):
            """Use the given function to determine the parameter hints.

            Args:
                func (callable): A fimction that rturns a dictionary of dictionaries

            Returns:
                The wrapped hinter function.

            Notes:
                This decorator will modify the instance attributes so that the instance has a method to generate parameter hints.

                func should only take keyword arguments as by default it will be called with no arguments during model initialisation.
            """

            @wraps(func)
            def _limits_proxy(self, **kargs):
                limits = func(**kargs)
                for param in limits:
                    if param not in self.param_names:
                        raise RuntimeError("Unrecognised parameter in hinter function: {}".format(param))
                    if not isinstance(limits[param], Mapping):
                        raise RuntimeError("Parameter hint for {} was not a mapping".format(param))
                return limits

            cls._limits = _limits_proxy
            return _limits_proxy

        @classmethod
        def guesser(cls, func):
            """Use the given function as the guess method.

            Args:
                func (callable): A function that guesses the parameter values

            Returns:
                The wrapped guess function.

            Notes:
                This decorator will modify the instance attributes so that the instance has a working guess method.

                func should take at least one positional argument, being the y-data values used to guess parameters.
                It should return a list, tuple of guesses parameter values with one entry for each parameter in the model.
            """

            @wraps(func)
            def guess_proxy(self, *args, **kargs):
                """A magic proxy call around a function to guess initial prameters."""
                guesses = func(*args, **kargs)
                pars = {x: y for x, y in zip(self.param_names, guesses)}
                pars = self.make_params(**pars)
                return update_param_vals(pars, self.prefix, **kargs)

            cls.guess = guess_proxy
            return guess_proxy

    return _ModelDecorator


def linear(x, intercept, slope):
    """Simple linear function"""
    return slope * x + intercept


class Linear(_Linear):

    """Simple linear fit"""

    pass


def cfg_data_from_ini(inifile, filename=None, **kargs):
    """Read an inifile and load and configure a DataFile from it.

    Args:
        inifile (str or file): Path to the ini file to be read.

    Keyword Arguments:
        filename (strig,boolean or None): File to load that contains the data.
        **kargs: All other keywords are passed to the Data constructor

    Returns:
        An instance of :py:class:`Stoner.Core.Data` with data loaded and columns configured.

    The inifile should contain a [Data] section that contains the following keys:

    -  **type (str):** optional name of DataFile subclass to import.
    -  **filename (str or boolean):** optionally used if *filename* parameter is None.
    - **xcol (column index):** defines the x-column data for fitting.
    - **ycol (column index):** defines the y-column data for fitting.
    - **yerr (column index):** Optional column with uncertainity values for the data
    """
    if SafeConfigParser is None:
        raise RuntimeError("Need to have ConfigParser module installed for this to work.")
    config = SafeConfigParser()
    if isinstance(inifile, string_types):
        config.read(inifile)
    elif isinstance(inifile, IOBase):
        config.readfp(inifile)
    if not config.has_section("Data"):
        raise RuntimeError("Configuration file lacks a [Data] section to describe data.")

    if config.has_option("Data", "type"):
        typ = config.get("Data", "type").split(".")
        typ_mod = ".".join(typ[:-1])
        typ = typ[-1]
        typ = __import__(typ_mod, fromlist=[typ]).__getattribute__(typ)
    else:
        typ = None
    data = Data(**kargs)
    if filename is None:
        if not config.has_option("Data", "filename"):
            filename = False
        else:
            filename = config.get("Data", "filename")
            if filename in ["False", "True"]:
                filename = bool(filename)
    data.load(filename, auto_load=False, filetype=typ)
    cols = {"x": 0, "y": 1, "e": None}  # Defaults

    for c in ["x", "y", "e"]:
        if not config.has_option("Data", c):
            pass
        else:
            try:
                cols[c] = config.get("Data", c)
                cols[c] = int(cols[c])
            except ValueError:
                pass
        if cols[c] is None:
            del cols[c]

    data.setas(**cols)  # pylint: disable=not-callable
    return data


def cfg_model_from_ini(inifile, model=None, data=None):
    r"""Utility function to configure an lmfit Model from an inifile.

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
    the :math:`\Chi^2` as a function of the parameters, then this array has a separate row for each iteration.
    """
    config = SafeConfigParser()
    if isinstance(inifile, string_types):
        config.read(inifile)
    elif isinstance(inifile, IOBase):
        config.readfp(inifile)

    if model is None:  # Check to see if config file specified a model
        try:
            model = config.get("Options", "model")
        except Exception:
            raise RuntimeError("Model is notspecifed either as keyword argument or in inifile")
    model = _get_model_(model)
    if config.has_option("option", "prefix"):
        prefix = config.get("option", "prefix")
    else:
        prefix = model.__class__.__name__
    prefix += ":"
    vals = []
    for p in model.param_names:
        if not config.has_section(p):
            raise RuntimeError("Config file does not have a section for parameter {}".format(p))
        keys = {
            "vary": bool,
            "value": float,
            "min": float,
            "max": float,
            "expr": str,
            "step": float,
            "label": str,
            "units": str,
        }
        kargs = dict()
        for k in keys:
            if config.has_option(p, k):
                if keys[k] == bool:
                    kargs[k] = config.getboolean(p, k)
                elif keys[k] == float:
                    kargs[k] = config.getfloat(p, k)
                elif keys[k] == str:
                    kargs[k] = config.get(p, k)
        if isinstance(data, _SC_.DataFile):  # stuff the parameter hint data into metadata
            for k in keys:  # remove keywords not needed
                if k in kargs:
                    data["{}{} {}".format(prefix, p, k)] = kargs[k]
            if "lmfit.prerfix" in data:
                data["lmfit.prefix"].append(prefix)
            else:
                data["lmfit.prefix"] = [prefix]
        if "step" in kargs:  # We use step for creating a chi^2 mapping, but not for a parameter hint
            step = kargs.pop("step")
            if "vary" in kargs and "min" in kargs and "max" in kargs and not kargs["vary"]:  # Make chi^2?
                vals.append(_np_.arange(kargs["min"], kargs["max"] + step / 10, step))
            else:  # Nope, just make a single value step here
                vals.append(_np_.array(kargs["value"]))
        else:  # Nope, just make a single value step here
            vals.append(_np_.array(kargs["value"]))
        kargs = {k: kargs[k] for k in kargs if k in ["value", "max", "min", "vary"]}
        model.set_param_hint(p, **kargs)  # set the model parameter hint
    msh = _np_.meshgrid(*vals)  # make a mesh of all possible parameter values to test
    msh = [m.ravel() for m in msh]  # tidy it up and combine into one 2D array
    msh = _np_.column_stack(msh)
    return model, msh


def arrhenius(x, A, DE):
    r"""Arrhenius Equation without T dependendent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\tau=A\exp\left(\frac{-\Delta E}{k_B x}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/Arrhenius.py
            :include-source:
            :outname: arrhenius

    """
    _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]
    return A * _np_.exp(-DE / (_kb * x))


class Arrhenius(Model):

    r"""Arrhenius Equation without T dependendent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\tau=A\exp\left(\frac{-\Delta E}{k_B x}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/Arrhenius.py
            :include-source:
            :outname: arrhenius-class
    """

    display_names = ["A", r"\Delta E"]

    def __init__(self, *args, **kwargs):
        """Configure default function to fit."""
        super(Arrhenius, self).__init__(arrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate fitting parameters from data."""
        _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]

        d1, d2 = 1.0, 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), DE=_kb * d1)
        return update_param_vals(pars, self.prefix, **kwargs)


def nDimArrhenius(x, A, DE, n):
    r"""Arrhenius Equation without T dependendent prefactor for various dimensions.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The dimensionalirty of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\tau=A\exp\left(\frac{-\Delta E}{k_B x^n}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/nDimArrhenius.py
            :include-source:
            :outname: nDimarrehenius
    """
    return arrhenius(x ** n, A, DE)


class NDimArrhenius(Model):

    r"""Arrhenius Equation without T dependendent prefactor for various dimensions.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The dimensionalirty of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\tau=A\exp\left(\frac{-\Delta E}{k_B x^n}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/nDimArrhenius.py
            :include-source:
            :outname: nDimarrhenius-class
    """

    display_names = ["A", r"\Delta E", "n"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(NDimArrhenius, self).__init__(nDimArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess paramneters from a set of data."""
        _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]

        d1, d2 = 1.0, 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), DE=_kb * d1, n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def modArrhenius(x, A, DE, n):
    r"""Arrhenius Equation with a variable T power dependent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The exponent of the temperature pre-factor of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The modified Arrhenius function is defined as :math:`\tau=Ax^n\exp\left(\frac{-\Delta E}{k_B x}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/modArrhenius.py
            :include-source:
            :outname: modarrhenius
    """
    return (x ** n) * arrhenius(x, A, DE)


class ModArrhenius(Model):

    r"""Arrhenius Equation with a variable T power dependent prefactor.

    Args:
        x (array): temperatyre data in K
        A (float): Prefactor - temperature independent. See :py:func:modArrhenius for temperaure dependent version.
        DE (float): Energy barrier in *eV*.
        n (float): The exponent of the temperature pre-factor of the model

    Return:
        Typically a rate corresponding to the given temperature values.

    The Arrhenius function is defined as :math:`\tau=Ax^n\exp\left(\frac{-\Delta E}{k_B x}\right)` where
    :math:`k_B` is Boltzmann's constant.

    Example:
        .. plot:: samples/Fitting/modArrhenius.py
            :include-source:
            :outname: modarrhenius-class
    """

    display_names = ["A", r"\Delta E", "n"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(ModArrhenius, self).__init__(modArrhenius, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess paramneters from a set of data."""
        _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]

        d1, d2 = 1.0, 0.0
        if x is not None:
            d1, d2 = _np_.polyfit(-1.0 / x, _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), DE=_kb * d1, n=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


def powerLaw(x, A, k):
    r"""Power Law Fitting Equation.

    Args:
        x (array): Input data
        A (float): Prefactor
        k (float): Pwoer

    Return:
        Power law.

    :math:`p=Ax^k`

    Example:
        .. plot:: samples/Fitting/Powerlaw.py
            :include-source:
            :outname: powerlaw
    """
    return A * x ** k


class PowerLaw(_PowerLaw):

    r"""Power Law Fitting Equation.

    Args:
        x (array): Input data
        A (float): Prefactor
        k (float): Pwoer

    Return:
        Power law.

    :math:`p=Ax^k`

    Example:
        .. plot:: samples/Fitting/Powerlaw.py
            :include-source:
            :outname: powerlaw-class
    """

    pass


def quadratic(x, a, b, c):
    r"""A Simple quadratic fitting function.

    Args:
        x (aray): Input data
        a (float): Quadratic term co-efficient
        b (float): Linear term co-efficient
        c (float): Constant offset term

    Returns:
        Array of data.

    :math:`y=ax^2+bx+c`

    Example:
        .. plot:: samples/Fitting/Quadratic.py
            :include-source:
            :outname: quadratic
    """
    return a * x ** 2 + b * x + c


class Quadratic(_Quadratic):

    r"""A Simple quadratic fitting function.

    Args:
        x (aray): Input data
        a (float): Quadratic term co-efficient
        b (float): Linear term co-efficient
        c (float): Constant offset term

    Returns:
        Array of data.

    :math:`y=ax^2+bx+c`

    Example:
        .. plot:: samples/Fitting/Quadratic.py
            :include-source:
            :outname: quadratic-class
    """

    pass


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

    Example:
        .. plot:: samples/Fitting/Simmons.py
            :include-source:
            :outname: simmons
    """
    I = (
        6.2e6
        * A
        / d ** 2
        * (
            (phi - V / 2) * _np_.exp(-1.025 * d * _np_.sqrt(phi - V / 2))
            - (phi + V / 2) * _np_.exp(-1.025 * d * _np_.sqrt(phi + V / 2))
        )
    )
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

    Example:
        .. plot:: samples/Fitting/Simmons.py
            :include-source:
            :outname: simmons-class
    """

    display_names = ["A", r"\phi", "d"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(Simmons, self).__init__(simmons, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1e3, phi=3.0, d=10.0)
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

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)

    Example:
        .. plot:: samples/Fitting/BDR.py
            :include-source:
            :outname: bdr
    """
    mass = abs(mass)
    phi = abs(phi)
    d = abs(d)
    I = (
        3.16e10
        * A ** 2
        * _np_.sqrt(phi)
        / d
        * _np_.exp(-1.028 * _np_.sqrt(phi) * d)
        * (V - 0.0214 * _np_.sqrt(mass) * d * dphi / phi ** 1.5 * V ** 2 + 0.0110 * mass * d ** 2 / phi * V ** 3)
    )
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

       See Brinkman et. al. J. Appl. Phys. 41 1915 (1970) or Tuan Comm. in Phys. 16, 1, (2006)

    Example:
        .. plot:: samples/Fitting/BDR.py
            :include-source:
            :outname: bdr-class
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(BDR, self).__init__(bdr, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Just set the A, phi,dphi,d and mass values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1e-12, phi=3.0, d=10.0, dphi=1.0, mass=1.0)
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

    Example:
        .. plot:: samples/Fitting/FowlerNordheim.py
            :include-source:
            :outname: fowlernordheim
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

    Example:
        .. plot:: samples/Fitting/FowlerNordheim.py
            :include-source:
            :outname: fowlernordheim-class
    """

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(FowlerNordheim, self).__init__(fowlerNordheim, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
        """Just set the A, phi and d values to typical answers for a small tunnel junction"""
        pars = self.make_params(A=1e-12, phi=3.0, d=10.0)
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
        """Configure Initial fitting function."""
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

    Example:
        .. plot:: samples/Fitting/weak_localisation.py
            :include-source:
            :outname: wlfit
    """

    e = 1.6e-19  # C
    h = 6.62e-34  # Js
    # Sets up conductivity fit array
    cond = _np_.zeros(len(B))
    if B2 == B1:
        B2 = B1 * 1.00001  # prevent dividing by zero

    # performs calculation for all parts
    for tt, Bi in enumerate(B):
        if Bi != 0:  # prevent dividing by zero
            WLpt1 = digamma(0.5 + B2 / _np_.abs(Bi))
            WLpt2 = digamma(0.5 + B1 / _np_.abs(Bi))
        else:
            WLpt1 = (digamma(0.5 + B2 / _np_.abs(B[tt - 1])) + digamma(0.5 + B2 / _np_.abs(B[tt + 1]))) / 2
            WLpt2 = (digamma(0.5 + B1 / _np_.abs(B[tt - 1])) + digamma(0.5 + B1 / _np_.abs(B[tt + 1]))) / 2

        WLpt3 = _np_.log(B2 / B1)

        # Calculates fermi level smearing
        cond[tt] = (e ** 2 / (h * _np_.pi)) * (WLpt1 - WLpt2 - WLpt3)
    # cond = s0*cond / min(cond)
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
            zpos = _np_.argmin(_np_.abs(B))
            s0 = data[zpos]
            B1 = _np_.max(B) / 2.0
            B2 = B1
            DS = 1.0
        pars = self.make_params(s0=s0, DS=DS, B1=B1, B2=B2)
        return update_param_vals(pars, self.prefix, **kwargs)


@jit(float64[:](float64[:], float64, float64, float64, float64))
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

    mv = _np_.max(_np_.abs(V))  # Limit for evaluating the integrals
    E = _np_.linspace(-2 * mv, 2 * mv, len(V) * 20)  # Energy range in meV - we use a mesh 20x denser than data points
    gauss = (1 / _np_.sqrt(2 * _np_.pi * omega ** 2)) * _np_.exp(-(E ** 2 / (2 * omega ** 2)))
    gauss /= gauss.sum()  # Normalised gaussian for the convolution

    # Conductance calculation
    #    For ease of calculation, epsilon = E/(sqrt(E^2 - delta^2))
    #    Calculates reflection probabilities when E < or > delta
    #    A denotes Andreev Reflection probability
    #    B denotes normal reflection probability
    #    subscript p for polarised, u for unpolarised
    #    Ap is always zero as the polarised current has 0 prob for an Andreev
    #    event

    Au1 = (delta ** 2) / ((E ** 2) + (((delta ** 2) - (E ** 2)) * (1 + 2 * (Z ** 2)) ** 2))
    Au2 = (((_np_.abs(E) / (_np_.sqrt((E ** 2) - (delta ** 2)))) ** 2) - 1) / (
        ((_np_.abs(E) / (_np_.sqrt((E ** 2) - (delta ** 2)))) + (1 + 2 * (Z ** 2))) ** 2
    )
    Bu2 = (4 * (Z ** 2) * (1 + (Z ** 2))) / (
        ((_np_.abs(E) / (_np_.sqrt((E ** 2) - (delta ** 2)))) + (1 + 2 * (Z ** 2))) ** 2
    )
    Bp2 = Bu2 / (1 - Au2)

    unpolarised_prefactor = (1 - P) * (1 + (Z ** 2))
    polarised_prefactor = 1 * (P) * (1 + (Z ** 2))
    # Optimised for a single use of np.where
    G = (
        unpolarised_prefactor
        + polarised_prefactor
        + +_np_.where(
            _np_.abs(E) <= delta,
            unpolarised_prefactor * (2 * Au1 - 1) - _np_.ones_like(E) * polarised_prefactor,
            unpolarised_prefactor * (Au2 - Bu2) - Bp2 * polarised_prefactor,
        )
    )

    # Convolve and chop out the central section
    cond = _np_.convolve(G, gauss)
    cond = cond[int(E.size / 2) : 3 * int(E.size / 2)]
    # Linear interpolation back onto the V data point
    matches = _np_.searchsorted(E, V)
    condl = cond[matches - 1]
    condh = cond[matches]
    El = E[matches - 1]
    Er = E[matches]
    cond = (condh - condl) / (Er - El) * (V - El) + condl
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

    Example:
        .. plot:: samples/lmfit_demo.py
            :include-source:
            :outname: strijkers
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

    Example:
        .. plot:: samples/lmfit_demo.py
            :include-source:
            :outname: strijkers-class
    """

    display_names = [r"\omega", r"\Delta", "P", "Z"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(Strijkers, self).__init__(strijkers, *args, **kwargs)

    def guess(self, data, V=None, **kwargs):  # pylint: disable=unused-argument
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

    Example:
        .. plot:: samples/Fitting/f_s.py
            :include-source:
            :outname: fluchssondheimer
        """
    k = t / l

    kernel = lambda x, k: (x - x ** 3) * _np_.exp(-k * x) / (1 - _np_.exp(-k * x))

    result = _np_.zeros(k.shape)
    for i, v in enumerate(k):
        ret1 = 1 - (3 * (1 - p) / (8 * v)) + (3 * (1 - p) / (2 * v))
        ret2 = quad(kernel, 0, 1, (v,))[0]
        result[i] = ret1 * ret2
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


def _bgintegrand(x, n):
    """The integrand for the Bloch Grueneisen model."""
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
        Evaluation of the BlochGrueneisen function for R(T)

    Example:
        .. plot:: samples/Fitting/b_g.py
            :include-source:
            :outname: blochgruneisen
    """
    ret = _np_.zeros(T.shape)
    for i, t in enumerate(T):
        intg = quad(_bgintegrand, 0, thetaD / (t), (n,))[0]
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

    return m * n * (1.0 / _np_.tanh(x) - 1.0 / x)


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

        M_s = (_np_.max(data) - _np_.min(data)) / 2.0
        if h is not None:
            d = _np_.sort(_np_.row_stack((h, data)))
            dd = savgol_filter(d, 7, 1)
            yd = dd[1] / dd[0]
            chi = _np_.interp(_np_.array([0]), d[0], yd)[0]
            mT = chi / M_s * (k / mu_0)
            # Assume T=150K for no good reason
            m = mT * 150
        else:
            m = 1e6 * (e * hbar) / (2 * electron_mass)  # guess 1 million Bohr Magnetrons
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

    The VFT equation is defined as as rr`\tau = A\exp\left(\frac{DE}{x-x_0}\right)` and represents
    a modifed form of the Arrenhius distribution with a freezing point of :math:`x_0`.

    Example:
        .. plot:: samples/Fitting/vftEquation.py
            :include-source:
            :outname: vft
    """
    _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]
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

    See :py:func:`Stoner.Fit.vftEquation` for an example.

    Example:
        .. plot:: samples/Fitting/vftEquation.py
            :include-source:
            :outname: vft-class
    """

    display_names = ["A", r"\Delta E", "x_0"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(VFTEquation, self).__init__(vftEquation, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess paramneters from a set of data."""
        _kb = consts.physical_constants["Boltzmann constant"][0] / consts.physical_constants["elementary charge"][0]

        d1, d2, x0 = 1.0, 0.0, 1.0
        if x is not None:
            x0 = x[_np_.argmin(_np_.abs(data))]
            d1, d2 = _np_.polyfit(-1.0 / (x - x0), _np_.log(data), 1)
        pars = self.make_params(A=_np_.exp(d2), dE=_kb * d1, x_0=x0)
        return update_param_vals(pars, self.prefix, **kwargs)


def stretchedExp(x, A, beta, x_0):
    r"""A stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\exp\left[\left(\frac{-x}{x_0}\right)^\beta\right]`.
    """
    return A * _np_.exp(-(x / x_0) ** beta)


class StretchedExp(Model):

    r"""A stretched exponential fuinction.

    Args:
        x (array): x data values
        A (float): Constant prefactor
        beta (float): Stretch factor
        x_0 (float): Scaling factor for x data

    Return:
        Data for a stretched exponentional function.

    The stretched exponential is defined as :math:`y=A\exp\left[\left(\frac{-x}{x_0}\right)^\beta\right]`.
    """

    display_names = ["A", r"\beta", "x_0"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(StretchedExp, self).__init__(stretchedExp, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters for the stretched exponential from data."""
        A, beta, x0 = 1.0, 1.0, 1.0
        if x is not None:
            A = data[_np_.argmin(_np_.abs(x))]
            x = _np_.log(x)
            y = _np_.log(-_np_.log(data / A))
            d = _np_.column_stack((x, y))
            d = d[~_np_.isnan(d).any(axis=1)]
            d = d[~_np_.isinf(d).any(axis=1)]
            d1, d2 = _np_.polyfit(d[:, 0], d[:, 1], 1)
            beta = d1
            x0 = _np_.exp(d2 / beta)
        pars = self.make_params(A=A, beta=beta, x_0=x0)
        return update_param_vals(pars, self.prefix, **kwargs)


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
    return (consts.mu0 * gamma / (2 * _np_.pi)) * _np_.sqrt((H + H_k) * (H + H_k + M_s))


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
        + _np_.sqrt(M_s ** 2 * gamma ** 2 * cnst.mu_0 ** 2 + 16 * _np_.pi ** 2 * f ** 2) / (2 * gamma * cnst.mu_0)
    )


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
        M_s = (4 * _np_.pi ** 2 * data ** 2 - gamma ** 2 * cnst.mu_0 ** 2 * (x ** 2 + 2 * x * H_k + H_k ** 2)) / (
            gamma ** 2 * cnst.mu_0 ** 2 * (x + H_k)
        )
        M_s = _np_.mean(M_s)

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
        M_s = (4 * _np_.pi ** 2 * x ** 2 - gamma ** 2 * cnst.mu_0 ** 2 * (data ** 2 + 2 * data * H_k + H_k ** 2)) / (
            gamma ** 2 * cnst.mu_0 ** 2 * (data + H_k)
        )
        M_s = _np_.mean(M_s)

        pars = self.make_params(g=g, M_s=M_s, H_k=H_k)
        pars["M_s"].min = 0
        pars["g"].min = g / 100
        pars["H_k"].min = 0
        pars["H_k"].max = M_s.max()
        return update_param_vals(pars, self.prefix, **kwargs)


def lorentzian_diff(x, A, sigma, mu):
    r"""Implement a differential form of a Lorentzian peak.

    Args:
        x (array): x data
        A (flaot): Peak amplitude
        sigma (float): peak wideth
        mu (float): peak location in x

        Returns
            :math:`\frac{A \sigma \left(2 \mu - 2 x\right)}{\pi \left(\sigma^{2} + \left(- \mu + x\right)^{2}\right)^{2}}`
        """
    return A * sigma * (2 * mu - 2 * x) / (_np_.pi * (sigma ** 2 + (-mu + x) ** 2) ** 2)


class Lorentzian_diff(Model):
    r"""lmfit Model rerprenting the differential form of a Lorentzian Peak.

    Args:
        x (array): x data
        A (flaot): Peak amplitude
        sigma (float): peak wideth
        mu (float): peak location in x

        Returns
            :math:`\frac{A \sigma \left(2 \mu - 2 x\right)}{\pi \left(\sigma^{2} + \left(- \mu + x\right)^{2}\right)^{2}}`
    """
    display_names = ["A", r"\sigma", r"\mu"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(Lorentzian_diff, self).__init__(lorentzian_diff, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""

        if x is None:
            x = _np_.linspace(1, len(data), len(data) + 1)

        x1 = x[_np_.argmax(data)]
        x2 = x[_np_.argmin(data)]
        sigma = abs(x1 - x2)
        mu = (x1 + x2) / 2.0
        y1 = _np_.max(data)
        y2 = _np_.min(data)
        dy = y1 - y2
        A = dy * (4 * _np_.pi * sigma ** 2) / (3 * _np_.sqrt(3))

        pars = self.make_params(A=A, sigma=sigma, mu=mu)
        pars["A"].min = 0
        pars["sigma"].min = 0
        pars["mu"].min = _np_.min(x)
        pars["mu"].max = _np_.max(x)
        return update_param_vals(pars, self.prefix, **kwargs)


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
            x = _np_.linspace(1, len(data), len(data) + 1)

        x1 = x[_np_.argmax(data)]
        x2 = x[_np_.argmin(data)]
        Delta_H = abs(x1 - x2)
        H_res = (x1 + x2) / 2.0
        y1 = _np_.max(data)
        y2 = _np_.min(data)
        dy = y1 - y2
        K_2 = dy * (4 * _np_.pi * Delta_H ** 2) / (3 * _np_.sqrt(3))
        ay = (y1 + y2) / 2
        K_1 = ay * _np_.pi / Delta_H

        pars = self.make_params(Delta_H=Delta_H, H_res=H_res, K_1=K_1, K_2=K_2)
        pars["K_1"].min = 0
        pars["K_2"].min = 0
        pars["Delta_H"].min = 0
        pars["H_res"].min = _np_.min(x)
        pars["H_res"].max = _np_.max(x)
        return update_param_vals(pars, self.prefix, **kwargs)


def rsj_noiseless(I, Ic_p, Ic_n, Rn, V_offset):
    r"""Implements a simple noiseless RSJ model.

    Args:
        I (array-like): Current values
        Ic_p (foat): Critical current on positive branch
        Ic_n (foat): Critical current on negative branch
        Rn (float): Normal state resistance
        V_offset(float): Offset volage in measurement

    Returns:
        (array) Calculated volatages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            $V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}$
    """

    normal_p = _np_.sign(I) * _np_.real(_np_.sqrt(I ** 2 - Ic_p ** 2)) * Rn
    normal_n = _np_.sign(I) * _np_.real(_np_.sqrt(I ** 2 - Ic_n ** 2)) * Rn
    p_branch = _np_.where(I > Ic_p, normal_p, _np_.zeros_like(I))
    n_branch = _np_.where(I < Ic_n, normal_n, p_branch)
    return n_branch + V_offset


class RSJ_Noiseless(Model):
    r"""Implements a simple noiseless RSJ model.

    Args:
        I (array-like): Current values
        Ic_p (foat): Critical current on positive branch
        Ic_n (foat): Critical current on negative branch
        Rn (float): Normal state resistance
        V_offset(float): Offset volage in measurement

    Returns:
        (array) Calculated volatages

    Notes:
        Impleemtns a simple form of the RSJ model for a Josephson Junction:

            $V(I)=R_N\frac{I}{|I|}\sqrt{I^2-I_c^2}-V_{offset}$
    """

    display_names = ["I_c^p", "I_c^n", "R_N", "V_{offset}"]

    def __init__(self, *args, **kwargs):
        """Configure Initial fitting function."""
        super(RSJ_Noiseless, self).__init__(rsj_noiseless, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Guess parameters as gamma=2, H_k=0, M_s~(pi.f)^2/(mu_0^2.H)-H"""

        if x is None:
            x = _np_.linspace(1, len(data), len(data) + 1)

        v_offset_guess = _np_.mean(data)
        v = _np_.abs(data - v_offset_guess)
        x = _np_.abs(x)

        v_low = _np_.max(v) * 0.05
        v_high = _np_.max(v) * 0.90

        ic_index = v < v_low
        rn_index = v > v_high
        ic_guess = _np_.max(x[ic_index])  # Guess Ic from a 2% of max V threhsold creiteria

        rn_guess = _np_.mean(v[rn_index] / x[rn_index])

        pars = self.make_params(Ic_p=ic_guess, Ic_n=-ic_guess, Rn=rn_guess, V_offset=v_offset_guess)
        pars["Ic_p"].min = 0
        pars["Ic_n"].max = 0
        return update_param_vals(pars, self.prefix, **kwargs)
