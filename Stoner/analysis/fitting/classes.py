# -*- coding: utf-8 -*-
"""Classes to support data fitting."""

from copy import deepcopy as copy
from dataclasses import dataclass, field
from inspect import getfullargspec, isclass
from typing import Union, Optional

import lmfit as lmfit_mod
import numpy as np
from lmfit.model import Model
from scipy.odr import Model as odrModel

from ...compat import get_func_params
from ...tools import AttributeStore

_lmfit = True


class ODR_Model(odrModel):
    """A wrapper for converting lmfit models to odr models."""

    def __init__(self, *args, **kargs):
        """Initialise with lmfit_mod.Models.Model or callable."""
        meta = kargs.pop("meta", {})
        kargs = copy(kargs)
        for n in list(kargs.keys()):
            if n in ["replace", "header", "result", "output", "residuals", "prefix"]:
                del kargs[n]
        p0 = kargs.pop("p0", kargs.pop("estimate", None))
        if args:
            args = list(args)
            model = args.pop(0)
        else:
            raise RuntimeError("Need at least one argument to make a fitting model.")

        if isclass(model) and issubclass(model, Model):  # Instantiate if only a class passed in
            model = model()
        if isclass(model) and issubclass(model, odrModel):
            model = model()
        if isinstance(model, Model):
            self.model = model
            self.func = model.func

            def modelfunc(beta, x, **kargs):
                return self.func(x, *beta, **kargs)

            meta["param_names"] = self.model.param_names
            meta["param_hints"] = self.model.param_hints
            meta["name"] = type(self.model).__name__
        elif isinstance(model, odrModel):
            self.model = model
            meta.update(model.meta)
            meta["param_names"] = model.meta.pop("param_names", [f"Param_{ix}" for ix, p in enumerate(p0)])
            meta["name"] = model.fcn.__name__

            modelfunc = model.fcn
            self.model.meta.update(meta)
        elif callable(model):
            self.model = None
            meta["name"] = model.__name__
            arguments = getfullargspec(model)[0]  # pylint: disable=W1505
            meta["param_names"] = list(arguments[1:])
            meta["param_hints"] = {x: {"value": 1.0} for x in arguments[1:]}
            # print(arguments,carargs,jeywords,defaults)
            self.func = model

            def modelfunc(beta, x, **_):  # pylint: disable=E0102
                """Warapper for model function."""
                return model(x, *beta)

            meta["__name__"] = meta["name"]
        else:
            raise ValueError(
                "".join(
                    [
                        f"Cannot construct a model instance from a {model} - ",
                        "need a callable, lmfit_mod.Model or scipy.odr.Model",
                    ]
                )
            )
        if not isinstance(p0, lmfit_mod.Parameters):  # This can happen if we are creating an ODR_Model in advance.
            tmp_model = AttributeStore(meta)
            p0 = _prep_lmfit_p0(tmp_model, None, None, p0, kargs)[0]
        p_new = []
        meta["params"] = copy(p0)
        for p in p0.values():
            p_new.append(p.value)
        p0 = p_new
        kargs["estimate"] = p0

        kargs["meta"] = meta

        super().__init__(modelfunc, *args, **kargs)

    @property
    def p0(self):
        """Convert an estimate attribute as p0."""
        return getattr(self, "estimate", None)

    @property
    def param_names(self):
        """Convert the meta parameter key param_names to an attribute."""
        return self.meta["param_names"]


class MimizerAdaptor:
    """Work with an lmfit_mod.Model or generic callable to use with scipy.optimize global minimization functions.

    The :pymod:`scipy.optimize` module's minimizers generally expect functions  which take an array like parameter
    space variable and then other arguments. This class will produce a suitable wrapper function and bounds
    variables from information int he lmfit_mod.Model.
    """

    def __init__(self, model, *args, **kargs):  # pylint: disable=unused-argument
        """Prepare the wrapper from the minimuzer.

        Args:
            modelower (lmfit):
                The model that has been fitted.
            *args (tuple):
                Positional parameters to initialise class.

        Keyword Arguments:
            params (lmfit:parameter or dict):
                Parameters used to fit model.
            **kargs (dict):
                Keyword arguments to initialise the result object/.

        Raises:
            RuntimeError:
                Fails if a *params* Parameter does not supply a fitted value.
        """
        self.func = model.func
        hints = kargs.pop("params")
        p0 = []
        upper = []
        lower = []
        for name, hint in hints.items():
            if not isinstance(hint, lmfit_mod.Parameter):
                hint = lmfit_mod.Parameter(**hint)
            if not hasattr(hint, "value"):
                raise RuntimeError(f"At the very least we need a starting value for the {name} parameter")
            v = hint.value
            p0.append(v)
            limits = [v * 10, v * 0.1]
            hint_upper = getattr(hint, "max", max(limits))
            hint_lower = getattr(hint, "min", min(limits))
            upper.append(hint_upper if not np.isinf(hint_upper) else max(limits))
            lower.append(hint_lower if not np.isinf(hint_lower) else min(limits))
        self.p0 = p0
        self.bounds = list(zip(lower, upper))

        def wrapper(beta, x, y, sigma, *args):
            """Calculate a least-squares goodness from the model functiuon."""
            beta = tuple(beta) + tuple(args)
            if sigma is None:
                sigma = np.ones_like(x)
            elif isinstance(sigma, float):
                sigma = np.ones_like(x) * sigma
            sigma = sigma / sigma.sum()  # normalise uncertainties
            sigma += np.finfo(float).eps
            weights = 1.0 / sigma**2
            variance = ((y - self.func(x, *beta)) ** 2) * weights
            return np.sum(variance) / (len(x) - len(beta))

        self.minimize_func = wrapper


class _Curve_Fit_Result:
    """Represent a result from fitting using :py:func:`scipy.optimize.curve_fit`
    as a class to make handling easier.
    """

    def __init__(self):
        """Store the results of the curve fit full_output fit.

        Args:
            popt (1D array):
                Optimal parameters for fit.
            pcov (2D array):
                Variance-co-variance matrix.
            infodict (dict):
                Additional information from curve_fit.
            mesg (str):
                Descriptive information from curve_fit.
            ier (int):
                Numerical error message.
        """
        self._mapping = {}
        self.func = lambda *args: None
        self.f_name = None
        self.args = []
        self.kwargs = {}
        self.p0 = None
        self._residual_vals = None
        self.f_name = None
        self._infodict = {}
        self._results = {}
        self.infodict = {k: None for k in ["mfev", "fvec", "fjac", "ipvt", "qtf"]}
        self.results = {k: None for k in ["pop", "perr", "pcov", "mesg", "ier", "chisq", "nfree"]}
        self.data = None
        self.labels = None
        self.units = None

    # Following properties used to return desired information

    @property
    def name(self):
        """Name of the model fitted."""
        return self.func.__name__

    @property
    def dict(self):
        """Optimal parameters and errors as a Dictionary."""
        ret = {}
        for p, v, e in zip(self.params, self.popt, self.perr):
            ret[p] = v
            ret["d_{}", format(p)] = e
        ret["chi-square"] = self.chisqr
        ret["red. chi-sqr"] = self.redchi
        ret["nfev"] = self.nfev
        return ret

    @property
    def full(self):
        """Return the same as :py:attr:`_curve_fit_result.row`."""
        return self, self.row

    @property
    def row(self):
        """Optimal parameters and errors as a single row."""
        ret = np.zeros(len(self.popt) * 2 + 1)
        ret[0:-2:2] = self.popt
        ret[1::2] = self.perr
        ret[-1] = self.chisq
        return ret

    @property
    def infodict(self):
        """Wrapper infodict."""
        return self._infodict

    @infodict.setter
    def infodict(self, value):
        """Wrapper for setting infodict and subkeys."""
        self._infodict = value
        if isinstance(value, dict):
            for k in value:
                self._mapping[k] = "_infodict"

    @property
    def results(self):
        """Wrapper for a results dictionary."""
        return self._results

    @results.setter
    def results(self, value):
        """Wrapper for setting results dictionary and nting keys."""
        self._results = value
        if isinstance(value, dict):
            for k in value:
                self._mapping[k] = "_results"

    @property
    def fit(self):
        """Copy of the fit report and optimal parameters and covariance."""
        return (self.popt, self.pcov)

    @property
    def fit_values(self):
        """Return the fit values if x data is set."""
        if self.data is None or self.func is None or self.popt is None:
            raise ValueError(
                "Need to have some x-data, the fitting functions and optimal parameters  before calculating fit"
            )
        return self.func(self.data.data[:, self.settings.columns.xcol], *self.popt)

    @property
    def perr(self):
        """Return standar error from covariance matrix."""
        if "perr" in self.results and self.results["perr"] is not None:
            return self.results["perr"]
        return np.sqrt(np.diag(self.pcov))

    @property
    def report(self):
        """Copy of the fit report."""
        return self

    @property
    def residual_vals(self):
        """If we have residual values, return the,."""
        if self._residual_vals is None and self.data is not None:
            self._residual_vals = self.data - self.fit_values
        return getattr(self, "_residual_vals", None)

    @residual_vals.setter
    def residual_vals(self, values):
        """Update residual values."""
        self._residual_vals = values

    @property
    def N(self):
        """Return th number of data points in dataset."""
        return len(self.data)

    @property
    def n_p(self):
        """Return the number of parameters in model."""
        return len(self.popt)

    @property
    def redchi(self):
        r"""Reduced $\chi^2$ Statistic."""
        return self.chisq

    @property
    def chisqr(self):
        r"""$\chi^2$ Statistic."""
        return self.chisq * (self.N - self.n_p)

    @property
    def aic(self):
        """Return the Akaike Information Criterion statistic."""
        return self.N * np.log(self.chisqr / self.N) + 2 * self.n_p

    @property
    def bic(self):
        """Return the Bayesian Information Criterion statistic."""
        return self.N * np.log(self.chisqr / self.N) + np.log(self.N) * self.n_p

    @property
    def params(self):
        """List the parameter class objects."""
        return get_func_params(self.func)

    def __dir__(self):
        """Extend the attribute directory."""
        return super().__dir__() + list(self._mapping.keys())

    def __getattr__(self, name):
        """Defer to using things we're tracking in the mapping."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name in self._mapping:
            return getattr(self, self._mapping[name]).get(name, None)
        raise AttributeError(f"{name} is not an attribute or {self.__class__.__name__}.")

    def __setattr__(self, name, value):
        """Pass through if an atrbute is already set."""
        if name != "_mapping" and name in self._mapping:
            getattr(self, self._mapping[name])[name] = value
            return
        super().__setattr__(name, value)

    def fit_report(self):
        """Create a Fit report like lmfit does."""
        template = f"""[[ Model ]]
    {self.name}
[[ Fit Statistics ]]
    # function evals   = {self.nfev}
    # data points      = {self.N}
    # variables        = {self.n_p}
    chi-square         = {self.chisqr}
    reduced chi-square = {self.redchi}
    Akaike info crit   = {self.aic}
    Bayesian info crit = {self.bic}
[[ Variables ]]\n"""
        for p, v, e, p0 in zip(self.params, self.popt, self.perr, self.p0):
            template += f"\t{p}: {v} +/- {e} ({(e * 100 / v):.3f}%) (init {p0})\n"
        template += "[[Correlations]] (unreported correlations are <  0.100)\n"
        for i, p in enumerate(self.params):
            for j in range(i + 1, len(self.params)):
                if np.abs(self.pcov[i, j]) > 0.1:
                    template += f"\t({p},{list(self.params)[j]})\t\t={self.pcov[i, j]:.3f}"
        return template

    def add_metadata(self, datafile):
        """Adds metadata keys to datafile based on this fit report.

        Args:
            datafile (Data):
                Data instance to add metadata keys to.

        Returns:
            (Data):
                Updated data file.
        """
        f_name = self.f_name
        for val, err, name, label, unit in zip(self.popt, self.perr, self.args, self.labels, self.units):
            datafile[f"{f_name}:{name}"] = val
            datafile[f"{f_name}:{name} err"] = err
            datafile[f"{f_name}:{name} label"] = datafile.metadata.get(f"{f_name}:{name} label", label)
            datafile[f"{f_name}:{name} unit"] = datafile.metadata.get(f"{f_name}:{name} unit", unit)
        if self.nfev is not None:
            datafile[f"{f_name}:nfev"] = self.nfev
        return datafile


@dataclass
class _Curve_Fit_Output:
    """Dataclass for gathering together information about the output from a curve fitting operation."""

    prefix: str = ""
    columns: AttributeStore = field(default_factory=dict)
    result: Union[bool, str, None] = None
    replace: bool = False
    header: Optional[str] = None
    residuals: bool = False
    asrow: bool = False
    output: str = "row"
    scale_covar: bool = True
    nan_policy: str = "raise"


def _prep_lmfit_model(model, kargs):
    """Prepare an lmfit model instance.

    Arguments:
        model (lmfit Model class or instance, or callable): the model to be fitted to the data.
        p0 (iterable or floats): The initial values of the fitting parameters.
        kargs (dict):Other keyword arguments passed to the fitting function

    Returns:
        model,p0, prefix (lmfit_mod.Model instance, iterable, str)

    Converts the model parameter into an instance of lmfit_mod.Model - either by instantiating the class or wrapping a
    callable into an lmfit_mod.Model class and establishes a prefix string from the model if not provided in the
    keyword arguments.
    """
    if Model is None:  # Will be the case if lmfit is not imported.
        raise RuntimeError(
            """To use the lmfit function you need to be able to import the lmfit module\n Try pip install lmfit\nat
            a command prompt."""
        )
    # Enure that model is an instance of an lmfit_mod.Model() class
    if isinstance(model, Model):
        pass
    elif isclass(model) and issubclass(model, Model):
        model = model(nan_policy="propagate")
    elif callable(model):
        model = Model(model)
    else:
        raise TypeError(f"{model} must be an instance of lmfit_mod.Model or a cllable function!")
    # Nprmalise p0 to be lmfit_mod.Parameters
    # Get a default prefix for the model
    prefix = str(kargs.pop("prefix", type(model).__name__))
    return model, prefix


def _prep_lmfit_p0(model, ydata, xdata, p0, kargs):
    """Prepare the initial start vector for an lmfit.

    Arguments:
        model (lmfit_mod.Model instance): model to fit with
        ydata,xdata (array): y and x data ppoints for fitting
        p0 (iterable of float): Existing p0 vector if defined
        kargs (dict): Other keyword arguments for the lmfit method.

    Returns:
        p0,single_fit (iterable of floats, bool): The revised initial starting vector and whether this is a single
        fit operation.
    """
    single_fit = True
    if p0 is None:  # First guess the p0 values using the model
        if isinstance(model, odrModel):
            p0 = model.estimate
        else:
            for p_name in model.param_names:
                if p_name in kargs:
                    model.set_param_hint(p_name, value=kargs.get(p_name))
            try:
                p0 = model.guess(ydata[0], x=xdata)
            except Exception:  # pylint: disable=W0703
                # Don't be fussy here about the potential exceptions
                p0 = lmfit_mod.Parameters()
                for p_name in model.param_names:
                    if p_name in kargs:
                        p0[p_name] = lmfit_mod.Parameter(name=p_name, value=kargs.get(p_name))
        single_fit = True

    if callable(p0):
        p0 = p0(ydata, xdata)
    if isinstance(p0, (list, tuple)):
        p0 = np.array(p0)

    if isinstance(p0, np.ndarray) and (p0.ndim == 1 or (p0.ndim == 2 and np.max(p0.shape) == p0.size)):
        single_fit = True
        p_new = lmfit_mod.Parameters()
        p0 = p0.ravel()
        for n, v in zip(model.param_names, p0):
            if hasattr(model, "param_hints"):
                hint = model.param_hints.get(n, {})
            else:
                hint = {}
            hint["value"] = v
            hint["name"] = n
            p_new[n] = lmfit_mod.Parameter(**hint)
        p0 = p_new
        for p_name in model.param_names:
            if p_name in kargs:
                p0[p_name] = lmfit_mod.Parameter(p_name, value=kargs.pop(p_name))
    elif isinstance(p0, np.ndarray) and p0.ndim == 2:  # chi^2 mapping
        single_fit = False
        return p0, single_fit

    if not isinstance(p0, lmfit_mod.Parameters):
        raise RuntimeError(f"Unknown data type for initial guess vector p0: {type(p0)}")
    if set(p0.keys()) < set(model.param_names):
        raise RuntimeError(
            f"Missing some values from the initial guess vector p0: {set(model.param_names) - set(p0.keys())}"
        )
    return p0, single_fit
