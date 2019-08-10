#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:44:56 2019

@author: phygbu
"""
__all__ = ["odr_Model", "FittingMixin"]
from inspect import isclass
from collections import Mapping, OrderedDict
import numpy as _np_
import numpy.ma as ma
from distutils.version import LooseVersion

import scipy as _sp_
from scipy.odr import Model as odrModel
from scipy.optimize import curve_fit, differential_evolution

from Stoner.compat import python_v3, string_types, index_types, get_func_params
from Stoner.tools import isNone, isiterable, islike_list, _attribute_store

try:  # Allow lmfit to be optional
    import lmfit

    if LooseVersion(lmfit.__version__) < LooseVersion("0.9.0"):
        from lmfit.model import Model
    else:
        from lmfit.model import Model
    _lmfit = True
except ImportError:
    Model = None
    Parameters = None
    _lmfit = False
from copy import deepcopy as copy

# from matplotlib.pylab import * #Surely not?
if python_v3:
    from inspect import getfullargspec
else:
    from inspect import getargspec as getfullargspec


class odr_Model(odrModel):

    """A wrapper for converting lmfit models to odr models."""

    def __init__(self, *args, **kargs):
        """Initialise with lmfit.models.Model or callable."""
        meta = kargs.pop("meta", dict())
        kargs = copy(kargs)
        for n in list(kargs.keys()):
            if n in ["replace", "header", "result", "output", "residuals", "prefix"]:
                del kargs[n]
        p0 = kargs.pop("p0", kargs.pop("estimate", None))
        if args:
            args = list(args)
            model = args.pop(0)
        else:
            raise RuntimeError("Need at least one argument to make a fitting model." "")

        if isclass(model) and issubclass(model, Model):  # Instantiate if only a class passed in
            model = model()
        if isclass(model) and issubclass(model, odrModel):
            model = model()
        if isinstance(model, Model):
            self.model = model
            self.func = model.func
            model = lambda beta, x, **kargs: self.model.func(x, *beta, **kargs)
            meta["param_names"] = self.model.param_names
            meta["param_hints"] = self.model.param_hints
            meta["name"] = self.model.__class__.__name__
        elif isinstance(model, odrModel):
            self.model = model

            def _func(x, *beta):
                return model.fcn(beta, x)

            self.func = _func
            meta["param_names"] = model.meta.pop("param_names", ["Param_{}".format(ix) for ix, p in enumerate(p0)])
            meta["name"] = model.fcn.__name__
        elif callable(model):
            self.model = None
            meta["name"] = model.__name__
            arguments = getfullargspec(model)[0]  # pylint: disable=W1505
            meta["param_names"] = list(arguments[1:])
            meta["param_hints"] = {x: {"value": 1.0} for x in arguments[1:]}
            # print(arguments,carargs,jeywords,defaults)
            func = model
            self.func = model

            def model(beta, x, **_):  # pylint: disable=E0102
                """Warapper for model function."""
                return func(x, *beta)

            meta["__name__"] = meta["name"]
        else:
            raise ValueError(
                "Cannot construct a model instance from a {} - need a callable, lmfit.Model or scipy.odr.Model".format(
                    type(model)
                )
            )
        if not isinstance(p0, lmfit.Parameters):  # This can happen if we are creating an odr_Model in advance.
            tmp_model = _attribute_store(meta)
            p0 = _prep_lmfit_p0(tmp_model, None, None, p0, kargs)[0]
        p_new = list()
        meta["params"] = copy(p0)
        for p in p0.values():
            p_new.append(p.value)
        p0 = p_new
        kargs["estimate"] = p0

        kargs["meta"] = meta

        super(odr_Model, self).__init__(model, *args, **kargs)

    @property
    def p0(self):
        """Convert an estimate attribute as p0."""
        return getattr(self, "estimate", None)

    @property
    def param_names(self):
        """Convert the meta parameter key param_names to an attribute."""
        return self.meta["param_names"]


class MimizerAdaptor(object):
    """A class that will work with an lmfit.Model or generic callable to use with scipy.optimize global minimization functions.

    The :pymod:`scipy.optimize` module's minimizers generally expect functions  which take an array like parameter space variable and
    then other arguments. This class will produce a suitable wrapper function and bounds variables from information int he lmfit.Model.
    """

    def __init__(self, model, *args, **kargs):
        """
        Setup the wrapper from the minimuzer.

        Args:
            model (lmfit):
                The model that has been fitted.
            *args (tuple):
                Positional parameters to initialise class.

        Keyword Arguments:
            params (lmfit:parameter or dict):
                Parameters used to fit model.
            **kargs (dict):
                Keyword arguments to intialise the reuslt object/.

        Raises:
            RuntimeError:
                Fails if a *params* Parameter does not supply a fitted value.
        """

        self.func = model.func
        hints = kargs.pop("params")
        p0 = list()
        upper = list()
        lower = list()
        for name, hint in hints.items():
            if not isinstance(hint, lmfit.Parameter):
                hint = lmfit.Parameter(**hint)
            if not hasattr(hint, "value"):
                raise RuntimeError("At the very least we need a starting value for the {} parameter".format(name))
            v = hint.value
            p0.append(v)
            limits = [v * 10, v * 0.1]
            u = getattr(hint, "max", max(limits))
            l = getattr(hint, "min", min(limits))
            upper.append(u if not _np_.isinf(u) else max(limits))
            lower.append(l if not _np_.isinf(l) else min(limits))
        self.p0 = p0
        self.bounds = [ix for ix in zip(upper, lower)]

        def wrapper(beta, x, y, sigma, *args):
            """Function that calculates a least-squares goodness from the model functiuon."""
            beta = tuple(beta) + tuple(args)
            if sigma is None:
                sigma = _np_.ones_like(x)
            sigma = sigma / sigma.sum()  # normalise uncertainties
            sigma += _np_.finfo(float).eps
            weights = 1.0 / sigma ** 2
            variance = ((y - self.func(x, *beta)) ** 2) * weights
            return _np_.sum(variance) / (len(x) - len(beta))

        self.minimize_func = wrapper


class _curve_fit_result(object):

    """Represent a result from fitting using :py:func:`scipy.optimize.curve_fit` as a class to make handling easier."""

    def __init__(self, popt, pcov, infodict, mesg, ier):
        """
        Store the results of the curve fit full_output fit.

        Args:
            popt (1D array):
                Optimal parameters for fit.
            pcov (2D array):
                Variance-co-variance matrix.
            infodict (dict):
                Additional information from curve_fit.
            mesg (str):
                Descriptive information frok curve_fit.
            ier (int):
                Numerical error message.
        """
        self.popt = popt
        self.pcov = pcov
        self.perr = _np_.sqrt(_np_.diag(pcov))
        self.mesg = mesg
        self.ier = ier
        self.infodict = infodict
        for k in infodict:
            setattr(self, k, infodict[k])

    # Following peroperties used to return desired information

    @property
    def name(self):
        """Name of the model fitted."""
        return self.func.__name__

    @property
    def full(self):
        """The same as :py:attr:`_curve_fit_result.row`"""
        return self, self.row

    @property
    def row(self):
        """Optimal parameters and errors as a single row."""
        ret = _np_.zeros(self.popt.size * 2)
        ret[0::2] = self.popt
        ret[1::2] = self.perr
        return ret

    @property
    def fit(self):
        """Copy of the fit report and optimal parameters and covariance."""
        return (self.popt, self.pcov)

    @property
    def data(self):
        """The data that was fitted."""
        self._data = getattr(self, "_data", _np_.array([]))
        return self._data

    @data.setter
    def data(self, data):
        """The data that was fitted."""
        self._data = data

    @property
    def report(self):
        """Copy of the fit report."""
        return self

    @property
    def N(self):
        """Number of data points in dataset."""
        return len(self.data)

    @property
    def n_p(self):
        """Number of parameters in model."""
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
        """Akaike Information Criterion statistic"""
        return self.N * _np_.log(self.chisqr / self.N) + 2 * self.n_p

    @property
    def bic(self):
        """Bayesian Information Criterion statistic"""
        return self.N * _np_.log(self.chisqr / self.N) + _np_.log(self.N) * self.n_p

    @property
    def params(self):
        """A list of parameter class objects."""
        return get_func_params(self.func)

    def fit_report(self):
        """A Fit report like lmfit does."""
        template = """[[ Model ]]
    {}
[[ Fit Statistics ]]
    # function evals   = {}
    # data points      = {}
    # variables        = {}
    chi-square         = {}
    reduced chi-square = {}
    Akaike info crit   = {}
    Bayesian info crit = {}
[[ Variables ]]\n""".format(
            self.name, self.nfev, self.N, self.n_p, self.chisqr, self.redchi, self.aic, self.bic
        )
        for p, v, e, p0 in zip(self.params, self.popt, self.perr, self.p0):
            template += "\t{}: {} +/- {} ({:.3f}%) (init {})\n".format(p, v, e, (e * 100 / v), p0)
        template += "[[Correlations]] (unreported correlations are <  0.100)\n"
        for i, p in enumerate(self.params):
            for j in range(i + 1, len(self.params)):
                if _np_.abs(self.pcov[i, j]) > 0.1:
                    template += "\t({},{})\t\t={:.3f}".format(p, list(self.params)[j], self.pcov[i, j])
        return template


def _get_model_parnames(model):
    """get a list of the model parameter names."""
    if isinstance(model, type) and (issubclass(model, Model) or issubclass(model, odrModel)):
        model = Model()

    if isinstance(model, Model):
        return model.param_names
    if isinstance(model, odrModel):
        if "param_names" in model.meta:
            return model.meta["param_names"]
        else:
            model = model.fcn
    if not callable(model):
        raise ValueError(
            "Unrecognised type for model! - should be lmfit.Model, scipy.odr.Model or callable, not {}",
            format(type(model)),
        )
    arguments = getfullargspec(model)[0]  # pylint: disable=W1505
    return list(arguments[1:])


def _curve_fit_p0_list(p0, model):
    """Takes something containing an initial vector and turns it into a list for curve_fit.

    Args:
        model (callable, lmfit/Model, odr.Model): miodel object for parameter names
        o0 (list,array,type or Mapping): Object containing the parameter gues values

    Returns:
        A list of starting values in the order in which they appear in the model.
    """
    if p0 is None:
        return p0

    if isinstance(p0, Mapping):
        p_new = OrderedDict()
        for x, v in p0.items():
            p_new[x] = getattr(v, "value", float(v))
        ret = []
        for x in _get_model_parnames(model):
            ret.append(p_new.get(x, None))
        return ret
    elif isiterable(p0):
        return [float(x) for x in p0]


def _prep_lmfit_model(model, kargs):
    """Prepare an lmfit model instance.

    Arguments:
        model (lmfit Model class or instance, or callable): the model to be fitted to the data.
        p0 (iterable or floats): The initial values of the fitting parameters.
        kargs (dict):Other keyword arguments passed to the fitting function

    Returns:
        model,p0, prefix (lmfit.Model instance, iterable, str)

    Converts the model parameter into an instance of lmfit.Model - either by instantiating the class or wrapping a
    callable into an lmfit.Model class and establishes a prefix string from the model if not provided in the keyword arguments.
    """
    if Model is None:  # Will be the case if lmfit is not imported.
        raise RuntimeError(
            "To use the lmfit function you need to be able to import the lmfit module\n Try pip install lmfit\nat a command prompt."
        )
    # Enure that model is an instance of an lmfit.Model() class
    if isinstance(model, Model):
        pass
    elif isclass(model) and issubclass(model, Model):
        model = model()
    elif callable(model):
        model = Model(model)
    else:
        raise TypeError("{} must be an instance of lmfit.Model or a cllable function!".format(model))
    # Nprmalise p0 to be lmfit.Parameters
    # Get a default prefix for the model
    prefix = str(kargs.pop("prefix", model.__class__.__name__))
    return model, prefix


def _prep_lmfit_p0(model, ydata, xdata, p0, kargs):
    """Prepare the initial start vector for an lmfit.

    Arguments:
        model (lmfit.Model instance): model to fit with
        ydata,xdata (array): y and x data ppoints for fitting
        p0 (iterable of float): Existing p0 vector if defined
        kargs (dict): Other keyword arguments for the lmfit method.

    Returns:
        p0,single_fit (iterable of floats, bool): The revised initial starting vector and whether this is a single fit operation.
    """
    single_fit = True
    if p0 is None:  # First guess the p0 values using the model
        try:
            p0 = model.guess(ydata, x=xdata)
        except Exception:  # Don't be fussy here
            p0 = lmfit.Parameters()
        for p_name in model.param_names:
            if p_name in kargs:
                p0[p_name] = lmfit.Parameter(value=kargs.pop(p_name))
        single_fit = True

    if callable(p0):
        p0 = p0(ydata, xdata)
    if isinstance(p0, (list, tuple)):
        p0 = _np_.array(p0)

    if isinstance(p0, _np_.ndarray) and (p0.ndim == 1 or (p0.ndim == 2 and _np_.max(p0.shape) == p0.size)):
        single_fit = True
        p_new = lmfit.Parameters()
        p0 = p0.ravel()
        for n, v in zip(model.param_names, p0):
            if hasattr(model, "param_hints"):
                hint = model.param_hints.get(n, {})
            else:
                hint = {}
            hint["value"] = v
            p_new[n] = lmfit.Parameter(**hint)
        p0 = p_new
        for p_name in model.param_names:
            if p_name in kargs:
                p0[p_name] = lmfit.Parameter(value=kargs.pop(p_name))
    elif isinstance(p0, _np_.ndarray) and p0.ndim == 2:  # chi^2 mapping
        single_fit = False
        return p0, single_fit

    if not isinstance(p0, lmfit.Parameters):
        raise RuntimeError("Unknown data type for initial guess vector p0: {}".format(type(p0)))
    if set(p0.keys()) < set(model.param_names):
        raise RuntimeError(
            "Missing some values from the initial guess vector p0: {}".format(set(model.param_names) - set(p0.keys()))
        )

    return p0, single_fit


class FittingMixin(object):

    """A mixin calss designed to work with :py:class:`Stoner.Core.DataFile` to provide additional curve_fiotting methods."""

    def annotate_fit(self, model, x=None, y=None, z=None, text_only=False, **kargs):
        """Annotate a plot with some information about a fit.

        Args:
            mode (callable or lmfit.Model):
                The function/model used to describe the fit to be annotated.

        Keyword Parameters:
            x (float):
                x co-ordinate of the label
            y (float):
                y co-ordinate of the label
            z (float):
                z co-ordinbate of the label if the current axes are 3D
            prefix (str):
                The prefix placed ahead of the model parameters in the metadata.
            text_only (bool):
                If False (default), add the text to the plot and return the current object, otherwise,
                return just the text and don't add to a plot.
            prefix(str):
                If given  overridges the prefix from the model to determine a prefix to the parameter names in the metadata

        Returns:
            (Datam, str):
                A copy of the current Data instance if text_only is False, otherwise returns the text.

        If *prefix* is not given, then the first prefix in the metadata lmfit.prefix is used if present,
        otherwise a prefix is generated from the model.prefix attribute. If *x* and *y* are not specified then they
        are set to be 0.75 * maximum x and y limit of the plot.
        """
        mode = kargs.pop("mode", "float")
        if isclass(model) and ((_lmfit and issubclass(model, Model)) or issubclass(model, odrModel)):
            model = model()  # Instantiate a bare class first

        if isinstance(model, odrModel):  # Get predix from odrModel
            model_prefix = model.meta.get("__name__", model.__class__.__name__)
            prefix = kargs.pop("prefix", self.get("odr.prefix", model_prefix))
            param_names = model.meta.get("param_names", [])
            display_names = model.meta.get("display_names", param_names)
        elif _lmfit and isinstance(model, Model):  # Get prefix from lmfit
            prefix = kargs.pop("prefix", self.get("lmfit.prefix", model.__class__.__name__))
            param_names = model.param_names
            display_names = getattr(model, "display_names", model.param_names)
        elif callable(model):  # Get prefix from callable name
            prefix = kargs.pop("prefix", model.__name__)
            model = Model(model)
            param_names = model.param_names
            display_names = getattr(model, "display_names", model.param_names)
        else:
            raise RuntimeError(
                "model should be either an lmfit.Model or a callable function, not a {}".format(type(model))
            )

        if prefix is not None:

            if isinstance(prefix, (list, tuple)):
                prefix = prefix[0]

            prefix = prefix.strip(" :")
            prefix = "" if prefix == "" else prefix + ":"

        else:
            if isinstance(prefix, (list, tuple)):
                prefix = prefix[0]

            if model.prefix == "":
                prefix = ""
            else:
                prefix = model.prefix + ":"

        x = 0.75 if x is None else x
        y = 0.5 if y is None else y

        try:  # if the model has an attribute display params then use these as the parameter anmes
            for k, display_name in zip(param_names, display_names):
                if prefix:
                    self["{}{} label".format(prefix, k)] = display_name
                else:
                    self[k + " label"] = display_name
        except (AttributeError, KeyError):
            pass

        text = "\n".join([self.format("{}{}".format(prefix, k), fmt="latex", mode=mode) for k in model.param_names])
        try:
            self["{}chi^2 label".format(prefix)] = r"\chi^2"
            text += "\n" + self.format("{}chi^2".format(prefix), fmt="latex", mode=mode)
        except KeyError:
            pass

        if not text_only:
            ax = self.fig.gca()
            if "zlim" in ax.properties():
                # 3D plot then
                if z is None:
                    zb, zt = ax.properties()["zlim"]
                    z = 0.5 * (zt - zb) + zb
                ax.text3D(x, y, z, text)
            elif "arrowprops" in kargs:
                ax.annotate(text, xy=(x, y), **kargs)
            else:
                kargs.pop("xycoords", None)
                kargs["transform"] = ax.transAxes
                ax.text(x, y, text, **kargs)
            ret = self
        else:
            ret = text
        return ret

    def _get_curve_fit_data(self, xcol, ycol, bounds, sigma):
        """Gather up the xdata and sigma columns for curve_fit."""
        working = self.search(xcol, bounds)
        working = ma.mask_rowcols(working, axis=0)
        if not isNone(sigma):
            sigma = working[:, self.find_col(sigma)]
        else:
            sigma = None
        if isinstance(xcol, string_types):
            xdat = working[:, self.find_col(xcol)]
        elif isiterable(xcol):
            xdat = ()
            for c in xcol:
                xdat = xdat + (working[:, self.find_col(c)],)
        else:
            xdat = working[:, self.find_col(xcol)]

        for i, yc in enumerate(ycol):

            if isinstance(yc, index_types):
                ydat = working[:, self.find_col(yc)]
            elif isinstance(yc, _np_.ndarray) and len(yc.shape) == 1 and len(yc) == len(self):
                ydat = yc
            else:
                raise RuntimeError(
                    "Y-data for fitting not defined - should either be an index or a 1D numpy array of the same length as the dataset"
                )
            if i == 0:
                ydata = _np_.atleast_2d(ydat)
            else:
                ydata = _np_.row_stack([ydata, ydat])

        return xdat, ydata, sigma

    def _assemnle_data_to_fit(self, xcol, ycol, sigma, bounds, scale_covar, sigma_x=None):
        """Marshall the data for doing a curve_fit or equivalent.

        Parameters:
            xcol (index):
                Column with xdata in it
            ycol(index):
                Column with ydata in it
            sigma (index or array-like):
                column of y-errors or uncertainity values.
            bounds (callable):
                Used to select the data rows to fit
            scale_covar (bool,None):
                If set the flag to scale the covariance.

        Returns:
            (data,scale_covar,col_assignments):
                data is a tuple of (x,y,sigma). scale_covar is False if sigma is real errors.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)
        working = self.search(_.xcol, bounds)
        working = ma.mask_rowcols(working, axis=0)
        xdata = working[:, self.find_col(_.xcol)]
        ydata = working[:, self.find_col(_.ycol)]
        # Now check for sigma_y and sigma_x and have them default to sigma (which in turn defaults to None)

        if sigma is not None:
            if isinstance(sigma, index_types):
                _ = self._col(xcol=xcol, ycol=ycol, yerr=sigma)
                sigma = working[:, self.find_col(sigma)]
            elif isinstance(sigma, (list, tuple)):
                sigma = ma.array(sigma)
            elif isinstance(sigma, _np_.ndarray):
                sigma = ma.array(sigma)  # ensure masked
            else:
                raise RuntimeError("Sigma should have been a column index or list of values")
        elif not isNone(_.yerr):
            sigma = working[:, self.find_col(_.yerr)]
        else:
            sigma = ma.ones(len(xdata))
            scale_covar = True

        if sigma_x is None:
            if _.xerr is None:
                sigma_x = sigma
            else:
                sigma_x = working[:, self.find_col(_.xerr)]
        else:
            if isinstance(sigma_x, index_types):
                _ = self._col(xcol=xcol, ycol=ycol, yerr=_.yerr, xerr=sigma_x)
                sigma_x = working[:, self.find_col(sigma_x)]
            elif isinstance(sigma_x, (list, tuple)):
                sigma_x = ma.array(sigma_x)
            elif isinstance(sigma_x, _np_.ndarray):
                sigma_x = ma.array(sigma_x)  # ensure masked
            else:
                raise RuntimeError("Sigma_x should have been a column index or list of values")

        mask = _np_.invert(ydata.mask)
        sigma = sigma[mask]
        ydata = ydata[mask]
        xdata = xdata[mask]  # lmfit doesn't seem to work well with masked data - here we just delete masked points

        return (xdata, ydata, sigma, sigma_x), scale_covar, _

    def __lmfit_one(self, model, data, params, prefix, columns, scale_covar, **kargs):
        """Carry out a single fit wioth lmfit.

        Args:
            model (lmfit.Model):
                Configured model
            data (tuple of xdata,ydata,sigma):
                Data and errors to use in fitting
            params (lmfit.Parameters):
                The parameters to use on the model for the fitting.
            prefix (str):
                Prefix for labels in metadata
            columns (attribute dict):
                Column assignments
            scale_covat (bool):
                Whether sigmas are absolute or relative.

        Keyword Arguments:
            result (bool,str):
                Where the result goes
            header (str):
                Name of new data column if used
            replace (bool):
                whether to add new dataa
            output (str):
                What to return

        Returns:
            (various):
                Results froma  fit or raises and exception.
        """
        if not _lmfit:
            raise RuntimeError("lmfit module not available.")

        replace = kargs.pop("replace", False)
        result = kargs.pop("result", False)
        header = kargs.pop("header", "")
        residuals = kargs.pop("residuals", False)
        output = kargs.pop("output", "row")
        kargs[model.independent_vars[0]] = data[0]
        fit = model.fit(data[1], params, scale_covar=scale_covar, weights=1.0 / data[2], **kargs)
        if fit.success:
            row = self._record_curve_fit_result(
                model,
                fit,
                columns.xcol,
                header,
                result,
                replace,
                residuals=residuals,
                prefix=prefix,
                ycol=columns.ycol,
            )

            retval = {"fit": (row[::2], fit.covar), "report": fit, "row": row, "full": (fit, row), "data": self}
            if output not in retval:
                raise RuntimeError("Failed to recognise output format:{}".format(output))
            else:
                return retval[output]
        else:
            raise RuntimeError("Failed to complete fit. Error was:\n{}\n{}".format(fit.lmdif_message, fit.message))

    def _record_curve_fit_result(
        self, func, fit, xcol, header, result, replace, residuals=False, ycol=None, prefix=None
    ):
        """Annotate the DataFile object with the curve_fit result."""
        if isinstance(func, (lmfit.Model)):
            f_name = func.__class__.__name__
            func = func.func
        elif isclass(func) and issubclass(func, lmfit.Model):
            f_name = func.__name__
            func = func.func
        elif isinstance(func, (_sp_.odr.Model)):
            f_name = func.meta["name"]
            func = func.func
        else:
            f_name = func.__name__
        if prefix is not None:
            f_name = prefix

        args = getfullargspec(func)[0]  # pylint: disable=W1505
        del args[0]
        if isinstance(fit, _curve_fit_result):  # Come from curve_fit
            popt = fit.popt
            perr = fit.perr
            nfev = fit.nfev
            chisq = fit.chisq
        elif isinstance(fit, lmfit.model.ModelResult):  # Come form an lmfit operation
            popt = [fit.params[x].value for x in args]
            perr = [fit.params[x].stderr for x in args]
            nfev = fit.nfev
            chisq = fit.redchi
        elif isinstance(fit, _sp_.odr.Output):
            popt = fit.beta
            perr = fit.sd_beta
            delta, eps = fit.delta, fit.eps
            nfree = len(delta) - len(popt)
            chisq = _np_.sum((delta ** 2 + eps ** 2)) / nfree
            nfev = None
        elif isinstance(fit, _sp_.optimize.OptimizeResult):
            popt = fit.popt
            perr = fit.perr
            nfev = fit.nfev
            nfree = len(self) - len(popt)
            data = self.data[:, ycol]
            fit_data = func(self.data[:, xcol], *popt)
            chisq = _np_.sum((data - fit_data) ** 2) / nfree
        else:
            raise RuntimeError("Unable to understand {} as a fitting result".format(type(fit)))

        for val, err, name in zip(popt, perr, args):
            self["{}:{}".format(f_name, name)] = val
            self["{}:{} err".format(f_name, name)] = err
            self["{}:{} label".format(f_name, name)] = self.metadata.get("{}:{} label".format(f_name, name), name)

        if not isinstance(header, string_types):
            header = "Fitted with " + func.__name__

        # Store our current mask, calculate new column's mask and turn off mask
        tmp_mask = self.mask
        col_mask = _np_.any(tmp_mask, axis=1)
        self.mask = False

        if isinstance(result, bool) and result:  # Appending data to end of data
            result = None
            tmp_mask = _np_.column_stack((tmp_mask, col_mask))
        else:  # Inserting data
            tmp_mask = _np_.column_stack((tmp_mask[:, 0:result], col_mask, tmp_mask[:, result:]))
        if islike_list(xcol):
            new_col = func(self[:, xcol].T, *popt)
        else:
            new_col = func(self.column(xcol), *popt)
        self.add_column(new_col, index=result, replace=replace, header=header)
        if residuals:
            if not islike_list(ycol):
                ycol = [ycol]
            for yc in ycol:
                residual_vals = self.column(yc) - new_col
                if isinstance(residuals, bool) and residuals:
                    if result is None:
                        residuals_idx = None
                    else:
                        residuals_idx = self.find_col(result) + 1
                else:
                    residuals_idx = residuals
                self.add_column(residual_vals, index=residuals_idx, replace=False, header=header + ":residuals")
                self["{}:mean residual".format(f_name)] = _np_.mean(residual_vals)
                self["{}:std residual".format(f_name)] = _np_.std(residual_vals)
                self["{}:chi^2".format(f_name)] = chisq
                self["{}:chi^2 err".format(f_name)] = _np_.sqrt(2 / len(residual_vals)) * chisq
        if nfev is not None:
            self["{}:nfev".format(f_name)] = nfev

        self.mask = tmp_mask
        # Make row object
        row = []
        ch = []
        for v, e, a in zip(popt, perr, args):
            row.extend([v, e])
            ch.extend([a, "{} stderr".format(a)])
        row.append(chisq)
        ch.append("$\\chi^2$")
        cls = self.data.__class__
        row = cls(row)
        row.column_headers = ch
        return row

    def _odr_one(self, data, model, prefix, _, **kargs):
        """Carry out a single fit wioth odr.

        Args:
            data (odr.Data):
                configured data
            model (odr.Model):
                Configured model
            prefix (str):
                Prefix for labels in metadata

        Keyword Arguments:
            result (bool,str):
                Where the result goes
            header (str):
                Name of new data column if used
            replace (bool):
                whether to add new dataa
            output (str):
                What to return

        Returns:
            (various):
                Results froma  fit or raises and exception.
        """
        result = kargs.pop("result", None)
        replace = kargs.pop("replace", False)
        header = kargs.pop("header", None)
        residuals = kargs.pop("residuals", False)
        asrow = kargs.pop("asrow", False)
        output = kargs.pop("output", "row" if asrow else "fit")

        fit = _sp_.odr.ODR(data, model, beta0=model.estimate)
        try:
            fit_result = fit.run()
            fit_result.redchi = fit_result.sum_square / (len(fit_result.y) - len(fit_result.beta))
            fit_result.chisqr = fit_result.sum_square

            tmp = "Beta:{}\nBeta Std Error:{}\nBeta Covariance:{}\n".format(
                fit_result.beta, fit_result.sd_beta, fit_result.cov_beta
            )
            if hasattr(fit_result, "info"):
                tmp += "Residual Variance:{}\nInverse Condition #:{}\nReason(s) for Halting:\n".format(
                    fit_result.res_var, fit_result.inv_condnum
                )
                for r in fit_result.stopreason:
                    tmp += "  %s\n" % r
            tmp += "Sum of orthogonal distance (~chi^2):{}\nReduced Sum of Orthogonal distances (~reduced chi^2): {}\n".format(
                fit_result.chisqr, fit_result.redchi
            )
            fit_result.fit_report = lambda: tmp

        except _sp_.odr.OdrError as err:
            print(err)
            return None
        except _sp_.odr.OdrStop as err:
            print(err)
            return None
        self._record_curve_fit_result(
            model, fit_result, _.xcol, header, result, replace, residuals, ycol=_.ycol, prefix=prefix
        )

        row = []
        # Store our current mask, calculate new column's mask and turn off mask

        param_names = getattr(model, "param_names", None)
        for i in range(len(param_names)):
            row.extend([fit_result.beta[i], fit_result.sd_beta[i]])
        row.append(fit_result.redchi)
        retval = {
            "fit": (row[::2], fit_result.cov_beta),
            "report": fit_result,
            "row": row,
            "full": (fit_result, row),
            "data": self,
        }
        if output not in retval:
            raise RuntimeError("Failed to recognise output format:{}".format(output))
        else:
            return retval[output]

    def curve_fit(self, func, xcol=None, ycol=None, sigma=None, **kargs):
        """General curve fitting function passed through from scipy.

        Args:
            func (callable, lmfit.Model, odr.Model):
                The fitting function with the form def f(x,*p) where p is a list of fitting parameters
            xcol (index, Iterable):
                The index of the x-column data to fit. If list or other iterable sends a tuple of x columns to func for N-d fitting.
            ycol (index, list of indices or array):
                The index of the y-column data to fit. If an array, then should be 1D and
                the same length as the data. If ycol is a list of indices then the columns are iterated over in turn, fitting occuring
                for each one. In this case the return value is a list of what would be returned for a single column fit.

        Keyword Arguments:
            p0 (list, tuple, array or callable):
                A vector of initial parameter values to try. See notes below.
            sigma (index):
                The index of the column with the y-error bars
            bounds (callable):
                A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool):
                Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool):
                Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None):
                If this is a string then it is used as the name of the fitted data. (default None)
            absolute_sigma (bool):
                If False, `sigma` denotes relative weights of the data points. The default True means that
                the sigma parameter is the reciprocal of the absoluate standard deviation.
            output (str, default "fit"):
                Specifiy what to return.

        Returns:
            (various):
                The return value is determined by the *output* parameter. Options are:
                    * "fit"    (tuple of popt,pcov) Optimal values of the fitting parameters p, and the variance-co-variance matrix
                                for the fitting parameters.
                    * "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                    * "full"    a tuple of (popt,pcov,dictionary of optional outputs, message, return code, row).
                    * "data"   a copy of the :py:class:`Stoner.Core.DataFile` object with fit recorded in the metadata and optionally as a new column.

        Note:
            If the columns are not specified (or set to None) then the X and Y data are taken using the
            :py:attr:`Stoner.Core.DataFile.setas` attribute.

            The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
            The x-column and y-column can be anything that :py:meth:`Stoner.Core.DataFile.find_col` can use as an index
            but typucally either strings to be matched against column headings or integers.
            The initial parameter values and weightings default to None which corresponds to all parameters starting
            at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the
            point is to be used in the fit and false if not.


            The *absolute_sigma* keyword determines whether the returned covariance matrix `pcov` is based on *estimated* errors in
            the data, and is not affected by the overall magnitude of the values in `sigma`. Only the relative magnitudes of the
            *sigma* values matter.
            If True, `sigma` describes one standard deviation errors of the input data points. The estimated covariance in `pcov` is
            based on these values.

            The starting vector *p0* can be either a list, tuple or array, or a callable that will produce a list, tuple or array. IF callable,
            it should take the form:

                def p0_func(ydata,x=xdata):
                    ....

            and return a list of parameter values that is in the same order as the model function. If p0 is not given and a :py:class:`lmfit.Model` or
            :py:class:`scipy.odr.Model` is supplied as the model function, then the model's estimates of the starting values will be used instead.


        See Also:
            *   :py:meth:`Stoner.Data.lmfit`
            *   :py:meth:`Stoner.Data.odr`
            *   :py:meth:`Stoner.Data.differential_evolution`
            *   User guide section :ref:`curve_fit_guide`
        """

        _ = self._col_args(scalar=False, xcol=xcol, ycol=ycol, yerr=sigma)
        xcol, ycol, sigma = _.xcol, _.ycol, _.yerr

        bounds = kargs.pop("bounds", lambda x, y: True)
        result = kargs.pop("result", None)
        replace = kargs.pop("replace", False)
        header = kargs.pop("header", None)
        residuals = kargs.pop("residuals", False)
        prefix = kargs.pop("prefix", None)

        # Support either scale_covar or absolute_sigma, the latter wins if both supplied
        # If neither are specified, then if sigma is not given, absolute sigma will be False.

        scale_covar = kargs.pop("scale_covar", sigma is not None)
        absolute_sigma = kargs.pop("absolute_sigma", not scale_covar)
        # Support both asrow and output, the latter wins if both supplied
        asrow = kargs.pop("asrow", False)
        output = kargs.pop("output", "row" if asrow else "fit")
        kargs["full_output"] = True

        if not isinstance(ycol, list):
            ycol = [ycol]

        xdat, ydata, sigma = self._get_curve_fit_data(xcol, ycol, bounds, sigma)

        # Support any of our alternatives for the fitting function
        if isinstance(func, type) and issubclass(func, (Model, _sp_.odr.Model)):
            func = func()
        if isinstance(func, _sp_.odr.Model):  # scipy othrothogonal model hack

            def _func(x, *beta):
                return func.fcn(beta, x)

            p0 = kargs.pop("p0", func.estimate)
        elif isinstance(func, Model):
            _func = func.func
            try:
                if "p0" not in kargs:  # Avoid expensive guess if we have a p0
                    pguess = func.guess
                else:
                    pguess = None
            except (
                ArithmeticError,
                AttributeError,
                LookupError,
                RuntimeError,
                NameError,
                OSError,
                TypeError,
                ValueError,
            ):
                pguess = None
            p0 = kargs.pop("p0", pguess)
        elif callable(func):
            _func = func
            p0 = kargs.pop("p0", None)
        else:
            raise TypeError(
                "curve_fit parameter 1 must be either a Model class from lmfit or scipy.odr, or a callable, not a {}".format(
                    type(func)
                )
            )

        if callable(p0):  # Allow the user to suppy p0 as a callanble function
            if ydata.ndim != 1:
                yy = ydata.ravel()
            else:
                yy = ydata
            try:  # Skip the guess if it fails
                p0 = p0(yy, xdat)
            except (
                ArithmeticError,
                AttributeError,
                LookupError,
                RuntimeError,
                NameError,
                OSError,
                TypeError,
                ValueError,
            ):
                p0 = None

        p0 = _curve_fit_p0_list(p0, func)

        retvals = []
        i = None
        for i, ydat in enumerate(ydata):

            if isinstance(sigma, _np_.ndarray) and sigma.shape[0] > 1:
                if sigma.shape[0] == len(ycol):
                    s = sigma[i]
                elif len(sigma.shape) == 2 and sigma.shape[1] == len(ycol):
                    s = sigma[:, i]
                else:
                    s = sigma  # probably this will fail!
            else:
                s = sigma
            report = _curve_fit_result(
                *curve_fit(_func, xdat, ydat, p0=p0, sigma=s, absolute_sigma=absolute_sigma, **kargs)
            )
            report.func = func
            if p0 is None:
                report.p0 = _np_.ones(len(report.popt))
            else:
                report.p0 = p0
            report.data = self
            report.residual_vals = ydata - report.fvec
            report.chisq = (report.residual_vals ** 2).sum()
            report.nfree = len(self) - len(report.popt)
            report.chisq /= report.nfree

            if result is not None:
                self._record_curve_fit_result(
                    func, report, xcol, header, result, replace, residuals=residuals, ycol=ycol, prefix=prefix
                )
            try:
                retvals.append(getattr(report, output))
            except AttributeError:
                raise RuntimeError("Specified output: {}, from curve_fit not recognised".format(kargs["output"]))
        if i == 0:
            retvals = retvals[0]
        return retvals

    def differential_evolution(self, model, xcol=None, ycol=None, p0=None, sigma=None, **kargs):
        """Fit model to the data using a differential evolution algorithm.

        Args:
            model (lmfit.Model):
                An instance of an lmfit.Model that represents the model to be fitted to the data
            xcol (index or None):
                Columns to be used for the x  data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` x column
            ycol (index or None):
                Columns to be used for the  y data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` y column

        Keyword Arguments:
            p0 (list, tuple, array or callable):
                A vector of initial parameter values to try. See the notes in :py:meth:`Stoner.Data.curve_fit` for more details.
            sigma (index):
                The index of the column with the y-error bars
            bounds (callable):
                A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool):
                Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool):
                Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None):
                If this is a string then it is used as the name of the fitted data. (default None)
            scale_covar (bool) :
                whether to automatically scale covariance matrix (leastsq only)
            output (str, default "fit"):
                Specifiy what to return.

        Returns:
            ( various ) :

                The return value is determined by the *output* parameter. Options are
                    - "fit"    just the :py:class:`lmfit.model.ModelFit` instance that contains all relevant information about the fit.
                    - "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                    - "full"    a tuple of the fit instance and the row.
                    - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the emtadata and optinally as a column of data.

        This function is essentially a wrapper around the :py:func:`scipy.optimize.differential_evolution` funtion that presents the same interface as the other
        Stoner package curve fitting functions. The parent function, however, does not provide the variance-covariance matrix to estimate the fitting errors. To
        work around this, this function does the initial fit with the differential evolution, but then uses that to give a starting vector to a call to
        :py:func:`scipy.optimize.curve_fit` to calculate the covariance matrix.

        See Also:
            -   :py:meth:`Stoner.Data.curve_fit`
            -   :py:meth:`Stoner.Data.lmfit`
            -   :py:meth:`Stoner.Data.odr`
            -   User guide section :ref:`curve_fit_guide`

        Example:
            .. plot:: samples/differential_evolution_simple.py
                :include-source:
                :outname: diffev1
        """
        bounds = kargs.pop("bounds", lambda x, y: True)
        result = kargs.pop("result", None)
        replace = kargs.pop("replace", False)
        residuals = kargs.pop("residuals", False)
        header = kargs.pop("header", None)
        # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
        absolute_sigma = kargs.pop("absolute_sigma", True)
        scale_covar = kargs.pop("scale_covar", not absolute_sigma)
        # Support both asrow and output, the latter wins if both supplied
        asrow = kargs.pop("asrow", False)
        output = kargs.pop("output", "row" if asrow else "fit")

        data, scale_covar, _ = self._assemnle_data_to_fit(xcol, ycol, sigma, bounds, scale_covar)
        data = data[0:3]
        model, prefix = _prep_lmfit_model(model, kargs)
        p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kargs)

        for k in model.param_names:
            kargs.pop(k, None)

        diff_model = MimizerAdaptor(model, params=p0)

        kargs["polish"] = kargs.get("polish", True)

        if not single_fit:
            raise NotImplementedError("Sorry chi^2 mapping not implemented for differential evolution yet.")
        fit = differential_evolution(diff_model.minimize_func, diff_model.bounds, data, **kargs)
        if not fit.success:
            raise RuntimeError(fit.message)
        kargs.pop("polish", None)
        kargs["full_output"] = True
        polish = _curve_fit_result(
            *curve_fit(model.func, data[0], data[1], sigma=data[2], p0=fit.x, absolute_sigma=not scale_covar, **kargs)
        )

        polish.func = model.func
        polish.p0 = p0
        polish.data = self
        polish.residual_vals = data[1] - polish.fvec
        polish.chisq = (polish.residual_vals ** 2).sum()
        polish.nfree = len(self) - len(polish.popt)
        polish.chisq /= polish.nfree

        model.popt = polish.popt
        fit.covar = polish.pcov
        fit.popt = polish.popt
        fit.perr = polish.perr
        fit.fit_report = polish.fit_report
        row = self._record_curve_fit_result(
            model, fit, _.xcol, header, result, replace, residuals=residuals, prefix=prefix, ycol=_.ycol
        )

        retval = {"fit": (row[::2], fit.covar), "report": fit, "row": row, "full": (fit, row), "data": self}
        if output not in retval:
            raise RuntimeError("Failed to recognise output format:{}".format(output))
        else:
            return retval[output]

    def lmfit(self, model, xcol=None, ycol=None, p0=None, sigma=None, **kargs):
        r"""Wrapper around lmfit module fitting.

        Args:
            model (lmfit.Model):
                An instance of an lmfit.Model that represents the model to be fitted to the data
            xcol (index or None):
                Columns to be used for the x  data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` x column
            ycol (index or None):
                Columns to be used for the  y data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` y column

        Keyword Arguments:
            p0 (list, tuple, array or callable):
                A vector of initial parameter values to try. See the notes in :py:meth:`Stoner.Data.curve_fit` for more details.
            sigma (index):
                The index of the column with the y-error bars
            bounds (callable):
                A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool):
                Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool):
                Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None):
                If this is a string then it is used as the name of the fitted data. (default None)
            scale_covar (bool) :
                whether to automatically scale covariance matrix (leastsq only)
            output (str, default "fit"):
                Specifiy what to return.

        Returns:
            ( various ) :
                The return value is determined by the *output* parameter. Options are
                    - "fit"    just the :py:class:`lmfit.model.ModelFit` instance that contains all relevant information about the fit.
                    - "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                    - "full"    a tuple of the fit instance and the row.
                    - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the emtadata and optinally as a column of data.

        See Also:
            -   :py:meth:`Stoner.Data.curve_fit`
            -   :py:meth:`Stoner.Data.odr`
            -   :py:meth:`Stoner.Data.differential_evolution`
            -   User guide section :ref:`fitting_with_limits`

        .. note::

           If *p0* is fed a 2D array, then it assumed that you want to calculate :math:`\chi^2` for different starting parameters
           with some variables fixed. In this mode, fitting is carried out repeatedly with each row representing one attempt with different
           values of the parameters. In this mode the return value is a 2D array whose rows correspond to the inputs to the rows of p0, the
           columns are the fitted values of the parameters with an additional column for :math:`\chi^2`.

        Example:
            .. plot:: samples/lmfit_simple.py
                :include-source:
                :outname: lmfit2
        """
        bounds = kargs.pop("bounds", lambda x, y: True)
        result = kargs.pop("result", None)
        replace = kargs.pop("replace", False)
        residuals = kargs.pop("residuals", False)
        header = kargs.pop("header", None)
        # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
        absolute_sigma = kargs.pop("absolute_sigma", True)
        scale_covar = kargs.pop("scale_covar", not absolute_sigma)
        # Support both asrow and output, the latter wins if both supplied
        asrow = kargs.pop("asrow", False)
        output = kargs.pop("output", "row" if asrow else "fit")

        data, scale_covar, _ = self._assemnle_data_to_fit(xcol, ycol, sigma, bounds, scale_covar)
        model, prefix = _prep_lmfit_model(model, kargs)
        p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kargs)

        if single_fit:
            ret_val = self.__lmfit_one(
                model,
                data,
                p0,
                prefix,
                _,
                scale_covar,
                result=result,
                header=header,
                replace=replace,
                output=output,
                residuals=residuals,
            )
        else:  # chi^2 mode
            pn = p0
            ret_val = _np_.zeros((pn.shape[0], pn.shape[1] * 2 + 1))
            for i, pn_i in enumerate(pn):  # iterate over every row in the supplied p0 values
                p0, single_fit = _prep_lmfit_p0(
                    model, data[1], data[0], pn_i, kargs
                )  # model, data, params, prefix, columns, scale_covar,**kargs)
                ret_val[i, :] = self.__lmfit_one(model, data, p0, prefix, _, scale_covar, output="row")
            if output == "data":  # Create a data object and seet column headers etc correctly
                ret = self.clone
                ret.data = ret_val
                ret.column_headers = []
                ret.setas = ""
                prefix = ret["lmfit.prefix"][-1]
                ix = 0
                for ix, p in enumerate(model.param_names):
                    if "{}{} label".format(prefix, p) in self:  # Get a label for the columns
                        label = self["{}{} label".format(prefix, p)]
                    else:  # Not already defined, so user parameter name
                        label = p
                    if "{}{} units".format(prefix, p) in self:  # Get a value for the units
                        units = r"({})".format(self["{}{} units".format(prefix, p)])
                    else:  # Not defined - no units
                        units = ""
                    # Set columns
                    ret.column_headers[2 * ix] = r"${} {}$".format(label, units)
                    ret.column_headers[2 * ix + 1] = r"$\delta{} {}$".format(label, units)
                    if not ret["{}{} vary".format(prefix, p)]:
                        fixed = 2 * ix
                ret.column_headers[-1] = "$\\chi^2$"
                ret.labels = ret.column_headers
                # Workout which columns are y,e and x
                plots = list(range(0, ix * 2 + 1, 2))
                errors = list(range(1, ix * 2 + 2, 2))
                plots.append(ix * 2 + 2)
                plots.remove(fixed)
                errors.remove(fixed + 1)
                ret.setas[plots] = "y"
                ret.setas[errors] = "e"
                ret.setas[fixed] = "x"
                ret_val = ret

        return ret_val

    def odr(self, model, xcol=None, ycol=None, **kargs):
        """Wrapper around scipy.odr orthogonal distance regression fitting.

        Args:
            model (scipy.odr.Model, lmfit.models.Model or callable):
                Tje model that describes the data. See below for more details.
            xcol (index or None):
                Columns to be used for the x  data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` x column
            ycol (index or None):
                Columns to be used for the  y data for the fitting. If not givem defaults to the
                :py:attr:`Stoner.Core.DataFile.setas` y column

        Keyword Arguments:
            p0 (list, tuple, array or callable):
                A vector of initial parameter values to try. See the notes to :py:meth:`Stoner.Data.curve_fit` for more details.
            sigma_x (index):
                The index of the column with the x-error bars
            sigma_y (index):
                The index of the column with the x-error bars
            bounds (callable):
                A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool):
                Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool):
                Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None):
                If this is a string then it is used as the name of the fitted data. (default None)
            output (str, default "fit"):
                Specifiy what to return.

        Returns:
            ( various ) :
                The return value is determined by the *output* parameter. Options are
                    - "fit"    just the :py:class:`scipy.odr.Output` instance (default)
                    - "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                    - "full"    a tuple of the fit instance and the row.
                    - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the emtadata and optinally
                        as a column of data.

        Notes:
            The function tries to make use of whatever model you give it. Specifically, it accepts:

                -   A subclass or an instance of :py:class:`scipy.odr.Model` : this is the native model type for the underlying scipy
                    odr package.
                -   A subclass or instance of an lmfit.models.Model: the :py:mod:`Stoner.Fit` module has a number of useful prebuilt
                    lmfit models that can be used directly
                    by this function.
                -   A callable function which should have a signature f(x,parameter1,parameter2...) and *not* the scip.odr stadnard f(beta,x)

            This function ois designed to be as compatible as possible with :py:meth:`AnalysisMixin.curve_fit` and
                :py:meth:`AnalysisMixin.lmfit` to facilitate easy of switching between them.

        See Also:
            -   :py:meth:`AnalysisMixin.curve_fit`
            -   :py:meth:`AnalysisMixin.lmfit`
            -   :py:meth:`Stoner.Data.differential_evolution`
            -   User guide section :ref:`fitting_with_limits`

        Example:
            .. plot:: samples/odr_simple.py
                 :include-source:
                 :outname: odrfit1
        """
        # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
        absolute_sigma = kargs.pop("absolute_sigma", True)
        scale_covar = kargs.pop("scale_covar", not absolute_sigma)
        # Support both asrow and output, the latter wins if both supplied
        sigma = kargs.pop("sigma", None)
        sigma_x = kargs.pop("sigma_x", None)
        bounds = kargs.pop("boinds", lambda x, r: True)
        p0 = kargs.pop("p0", None)
        data, scale_covar, _ = self._assemnle_data_to_fit(xcol, ycol, sigma, bounds, scale_covar, sigma_x=sigma_x)
        model, prefix = _prep_lmfit_model(model, kargs)
        p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kargs)
        kargs["p0"] = p0
        model = odr_Model(model, p0=p0)
        if not absolute_sigma:
            data = _sp_.odr.Data(data[0], data[1], wd=1 / data[3] ** 2, we=1 / data[2] ** 2)
        else:
            data = _sp_.odr.RealData(data[0], data[1], sx=data[3], sy=data[2])

        if single_fit:
            ret_val = self._odr_one(data, model, prefix, _, **kargs)
        else:  # chi^2 mode
            raise NotImplementedError("Sorry cannot do chi^2 mode for orthogonal distance regression yet!")
        return ret_val
