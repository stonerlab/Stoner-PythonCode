#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fitting Functions and classes to mixin for :py:class:`~Stoner.core.data.Data`."""

from collections.abc import Mapping
from copy import deepcopy
from inspect import getfullargspec, isclass

import lmfit as lmfit_mod
import numpy as np
import scipy as sp
from lmfit.model import Model
from scipy.odr import Model as odrModel
from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize import differential_evolution as _differential_evolution

from ...compat import string_types
from ...core.utils import assemnle_data as _assemnle_data_to_fit
from ...tools import isiterable, islistlike, ordinal
from .classes import (
    MimizerAdaptor,
    ODR_Model,
    _Curve_Fit_Output,
    _Curve_Fit_Result,
    _prep_lmfit_model,
    _prep_lmfit_p0,
)
from .selection import curve_fit_inputs as _curve_fit_inputs

_lmfit = True


def _get_model_parnames(model):
    """Get a list of the model parameter names."""
    if isinstance(model, type) and (issubclass(model, Model) or issubclass(model, odrModel)):
        model = model()

    if isinstance(model, Model):
        return model.param_names
    if isinstance(model, odrModel):
        if "param_names" in model.meta:
            return model.meta["param_names"]
        model = model.fcn
    if not callable(model):
        raise ValueError(
            "".join(
                [
                    "Unrecognised type for model! - should be lmfit_mod.Model, scipy.odr.Model",
                    f" or callable, not {type(model)}",
                ]
            )
        )
    arguments = getfullargspec(model)[0]  # pylint: disable=W1505
    return list(arguments[1:])


def _curve_fit_p0_list(p0, model):
    """Take something containing an initial vector and turns it into a list for curve_fit.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (callable, lmfit/Model, odr.Model): miodel object for parameter names
        p0 (list,array,type or Mapping): Object containing the parameter gues values

    Returns:
        A list of starting values in the order in which they appear in the model.
    """
    if p0 is None:
        return p0

    if isinstance(p0, Mapping):
        p_new = {}
        for x, v in p0.items():
            p_new[x] = getattr(v, "value", float(v))
        ret = []
        for x in _get_model_parnames(model):
            ret.append(p_new.get(x, None))
        return ret
    if isiterable(p0):
        return [float(x) for x in p0]
    raise RuntimeError("Shouldn't have returned None from _curve_fit_p0_list!")


def _get_curve_fit_func(func, kwargs):
    """Construct a fitting function and initial guess set for simple curve_fit.

    Args:
        func (fitting model or callable):
            The model we are trying to use to fit the data.
        kwargs (dict):
            The Keyword Arguments pass to the cuve_fit method.

    Returns:
        (callable,list|callable):
            fitting function in the correct form for scipy.optimize.curve)fit and the initial guesses.
    """
    if isinstance(func, type) and issubclass(func, (Model, sp.odr.Model)):
        func = func()
    match func:
        case sp.odr.Model():

            def _func(x, *beta):
                return func.fcn(beta, x)

            return _func, kwargs.pop("p0", func.estimate)
        case Model():
            _func = func.func
            try:
                if "p0" not in kwargs:  # Avoid expensive guess if we have a p0
                    pguess = func.guess
                else:
                    pguess = None
            except (
                ArithmeticError,
                RuntimeError,
                ValueError,
            ):
                pguess = None
            return _func, kwargs.pop("p0", pguess)
        case _ if callable(func):
            return func, kwargs.pop("p0", None)
        case _:
            raise TypeError(
                "curve_fit model must be either a Model class from lmfit or scipy.odr or a"
                "callable and not a {yupe(func)}"
            )


def __lmfit_one(
    datafile,
    model,
    data,
    params,
    settings,
):
    """Carry out a single fit wioth lmfit.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (lmfit_mod.Model):
            Configured model
        data (tuple of xdata,ydata,sigma):
            Data and errors to use in fitting
        params (lmfit_mod.Parameters):
            The parameters to use on the model for the fitting.
        settings (str):
            Dataclass with output settings from fit.

    Returns:
        (various):
            Results froma  fit or raises and exception.
    """
    if not _lmfit:
        raise RuntimeError("lmfit module not available.")

    kwargs = {}
    kwargs[model.independent_vars[0]] = data.x
    fit = model.fit(
        data.y,
        params,
        scale_covar=settings.scale_covar,
        weights=1.0 / data.e,
        nan_policy=settings.nan_policy,
        **kwargs,
    )
    if fit.success:
        row = _record_curve_fit_result(datafile, model, fit, settings)

        res_dict = {}
        for p in fit.params:
            res_dict[p] = fit.params[p].value
            res_dict[f"d_{p}"] = fit.params[p].stderr
        res_dict["chi-square"] = fit.chisqr
        res_dict["red. chi-sqr"] = fit.redchi
        res_dict["nfev"] = fit.nfev

        retval = {
            "fit": (row[::2], fit.covar),
            "report": fit,
            "row": row,
            "full": (fit, row),
            "data": datafile,
            "dict": res_dict,
        }
        if settings.output not in retval:
            raise RuntimeError(f"Failed to recognise output format:{settings.output}")
        return retval[settings.output]
    raise RuntimeError(f"Failed to complete fit. Error was:\n{fit.lmdif_message}\n{fit.message}")


def _normalise_model_func(func, prefix, result_obj):
    """ "Normalise the model function and parameters."""
    match func:
        case lmfit_mod.Model():
            f_name = prefix or type(func).__name__
            labels = getattr(type(func), "labels", None)
            units = getattr(type(func), "units", None)
            func = func.func
            args = getfullargspec(func)[0]  # pylint: disable=W1505
            if args[0] in ["datafile", "self"]:
                del args[0]
            if len(args) > 0:
                del args[0]
        case _ if isclass(func) and issubclass(func, lmfit_mod.Model):
            f_name = prefix or func.__name__
            labels = getattr(func, "labels", None)
            units = getattr(func, "units", None)
            func = func().func
            args = getfullargspec(func)[0]  # pylint: disable=W1505
            if args[0] == "datafile":
                del args[0]
            if len(args) > 0:
                del args[0]
        case sp.odr.Model():
            f_name = prefix or func.meta["name"]
            labels = getattr(func, "labels", None)
            units = getattr(func, "units", None)
            args = func.meta["param_names"]
            model = func

            def _func(x, *beta):
                return model.fcn(beta, x)

            func = _func
        case _:
            f_name = prefix or func.__name__
            labels = getattr(func, "labels", None)
            units = getattr(func, "units", None)
            args = getfullargspec(func)[0]  # pylint: disable=W1505
            if args[0] in ["datafile", "self"]:
                del args[0]
            if len(args) > 0:
                del args[0]
    labels = labels or args
    units = units or [""] * len(args)
    result_obj.func = func
    result_obj.f_name = f_name
    result_obj.labels = labels
    result_obj.units = units
    result_obj.args = args
    return result_obj


def _normalise_fit_result(datafile, settings, fit, result_obj):
    """Normalise the fit results based on the fit instance."""
    func = result_obj.func
    args = result_obj.args
    result_obj.data = datafile
    result_obj.settings = settings

    match fit:
        case _Curve_Fit_Result():
            popt = fit.popt
            perr = fit.perr
            nfev = fit.nfev
            nfree = fit.nfree
            chisq = fit.chisq
        case lmfit_mod.model.ModelResult():
            popt = [fit.params[x].value for x in args]
            perr = [fit.params[x].stderr for x in args]
            nfev = fit.nfev
            nfree = len(datafile) - len(popt)
            chisq = fit.redchi
        case sp.odr.Output():
            popt = fit.beta
            perr = fit.sd_beta
            delta, eps = fit.delta, fit.eps
            nfree = len(delta) - len(popt)
            chisq = np.sum((delta**2 + eps**2)) / nfree
            nfev = None
        case sp.optimize.OptimizeResult():
            popt = fit.popt
            perr = fit.perr
            nfev = fit.nfev
            nfree = len(datafile) - len(popt)
            fit_data = func(datafile // settings.columns.xcol, *popt)
            chisq = np.sum((datafile.to_numpy()[:, settings.columns.ycol] - fit_data) ** 2) / nfree
        case _:
            raise RuntimeError("Unable to understand {type(fit)} as a fitting result")
    result_obj.results = {"popt": popt, "perr": perr, "nfev": nfev, "chisq": chisq, "nfree": nfree}
    return result_obj


def _record_curve_fit_result(datafile, func, fit, settings):
    """Annotate the DataFile object with the curve_fit result."""
    result_obj = _Curve_Fit_Result()
    result_obj = _normalise_model_func(func, settings.prefix, result_obj)
    result_obj = _normalise_fit_result(datafile, settings, fit, result_obj)

    result_obj.add_metadata(datafile)

    if not isinstance(settings.header, string_types):
        settings.header = f"Fitted with {result_obj.f_name}"

    store_result = settings.result is not None and settings.result is not False
    if not store_result:
        return result_obj.row
    if hasattr(fit, "model_values"):
        new_col = np.ma.array(fit.model_values, mask=fit.model_mask)
        residuals = [np.ma.array(fit.full_residuals, mask=fit.residual_mask)] if settings.residuals else []
    else:
        col_mask = np.any(np.asarray(datafile.mask), axis=1)
        new_col = np.ma.array(result_obj.fit_values, mask=col_mask)
        ycols = np.atleast_1d(settings.columns.ycol).tolist()
        residuals = [datafile.column(yc).copy() - new_col for yc in ycols] if settings.residuals else []
    if settings.result is True:
        settings.result = datafile.shape[1]
    if store_result:
        datafile.add_column(new_col, index=settings.result, replace=settings.replace, header=settings.header)
    if settings.residuals and store_result:
        for residual_vals in residuals:
            if isinstance(settings.residuals, bool) and settings.residuals:
                match settings.result:
                    case None:
                        residuals_idx = None
                    case True:
                        residuals_idx = len(datafile.column_headers)
                    case _:
                        residuals_idx = datafile.find_col(settings.result) + 1
            else:
                residuals_idx = settings.residuals
            datafile.add_column(
                residual_vals, index=residuals_idx, replace=False, header=settings.header + ":residuals"
            )
            datafile[f"{result_obj.f_name}:mean residual"] = np.mean(residual_vals)
            datafile[f"{result_obj.f_name}:std residual"] = np.std(residual_vals)
            datafile[f"{result_obj.f_name}:chi^2"] = result_obj.chisq
            datafile[f"{result_obj.f_name}:chi^2 err"] = np.sqrt(2 / len(residual_vals)) * result_obj.chisq

    # Make row object
    return result_obj.row


def _odr_one(
    datafile,
    data,
    model,
    settings,
    p0=None,
    **kwargs,
):
    """Carry out a single fit wioth odr.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        data (odr.Data):
            configured data
        model (odr.Model):
            Configured model
        settings (_Curve_Fit_Output):
            Dataclass with output settings from fit.

    Keyword Arguments:
        p0 (Iterable):
            Estimated values for model. If not given, model.estimate is used.
        **kwargs:
            Other parameters to try to create a p0 with.

    Returns:
        (various):
            Results froma  fit or raises and exception.
    """
    if settings.output is None:
        settings.output = "row" if settings.asrow else "fit"

    if p0 is not None:
        model.estimate = p0
    else:
        p0 = np.ones_like(model.param_names)
        for ix, k in enumerate(model.param_names):
            p0[ix] = kwargs.pop(k, 1.0)
        model.estimate = p0
    fit = sp.odr.ODR(data, model, beta0=[x.value for x in model.estimate.values()])
    try:
        fit_result = fit.run()
        fit_result.redchi = fit_result.sum_square / (  # pylint: disable=no-member
            len(fit_result.y) - len(fit_result.beta)  # pylint: disable=no-member
        )
        fit_result.chisqr = fit_result.sum_square  # pylint: disable=no-member

        tmp = f"""Beta:{fit_result.beta}
        Beta Std Error:{fit_result.sd_beta}
        Beta Covariance:{fit_result.cov_beta}
        """

        if hasattr(fit_result, "info"):
            tmp += f"""Residual Variance:{fit_result.res_var}
            Inverse Condition #:{fit_result.inv_condnum}
            Reason(s) for Halting:
            """
            for r in fit_result.stopreason:
                tmp += f"  {r}\n"
        tmp += f""""Sum of orthogonal distance (~chi^2):{fit_result.chisqr}
        Reduced Sum of Orthogonal distances (~reduced chi^2): {fit_result.redchi}"""

        fit_result.fit_report = lambda: tmp

    except sp.odr.OdrError as err:
        print(err)
        return None
    except sp.odr.OdrStop as err:
        print(err)
        return None
    _record_curve_fit_result(datafile, model, fit_result, settings)

    row = []
    # Store our current mask, calculate new column's mask and turn off mask

    param_names = getattr(model, "param_names", None)
    ret_dict = {}
    for i, p in enumerate(param_names):
        row.extend([fit_result.beta[i], fit_result.sd_beta[i]])
        ret_dict[p], ret_dict[f"d_{p}"] = fit_result.beta[i], fit_result.sd_beta[i]
    row.append(fit_result.redchi)
    ret_dict["chi-square"] = fit_result.chisqr
    ret_dict["red. chi-sqr"] = fit_result.redchi

    row = np.array(row)

    retval = {
        "fit": (row[::2], fit_result.cov_beta),
        "report": fit_result,
        "row": row,
        "full": (fit_result, row),
        "data": datafile,
        "dict": ret_dict,
    }
    if settings.output not in retval:
        raise RuntimeError(f"Failed to recognise output format:{settings.output}")
    return retval[settings.output]


def _chi2_fit_to_data(datafile, ret_val, model):
    """Convert a chi^2 fit to a Data instance."""
    ret = datafile.clone
    ret.data = ret_val
    ret.column_headers = [f"Column {i}" for i in range(ret.shape[1])]
    ret.setas = ""
    prefix = ret["lmfit.prefix"][-1]
    ix = fixed = 0
    for ix, p in enumerate(model.param_names):
        label = datafile.metadata.get(f"{prefix}{p} label", p)
        units = datafile.metadata.get(f"{prefix}{p} units", "")
        # Set columns
        ret.column_headers[2 * ix] = rf"${label} {units}$"
        ret.column_headers[2 * ix + 1] = rf"$\delta{label} {units}$"
        if not ret[f"{prefix}{p} vary"]:
            fixed = 2 * ix
    ret.column_headers[-1] = "$\\chi^2$"
    ret.plot.labels = ret.column_headers
    # Workout which columns are y,e and x
    plots = list(range(0, ix * 2 + 1, 2))
    errors = list(range(1, ix * 2 + 2, 2))
    plots.append(ix * 2 + 2)
    plots.remove(fixed)
    errors.remove(fixed + 1)
    ret.setas[plots] = "y"
    ret.setas[errors] = "e"
    ret.setas[fixed] = "x"
    return ret


def annotate_fit(datafile, model, x=None, y=None, z=None, text_only=False, mode="float", **kwargs):
    """Annotate a fit or return its formatted parameter text.

    Args:
        datafile (Data):
            Measurement object if this function is not used as a bound method.
        model (callable or lmfit_mod.Model):
            Function, lmfit model or ODR model describing the recorded fit.

    Keyword Arguments:
        x (float):
            Horizontal annotation coordinate. Defaults to 0.75.
        y (float):
            Vertical annotation coordinate. Defaults to 0.5.
        z (float):
            Third coordinate for 3D text; None uses the midpoint of the Z limits.
        text_only (bool):
            Return text without drawing when True. Defaults to False.
        ax (matplotlib.axes.Axes or None):
            Explicit annotation target. None uses the plotting accessor's recent
            axes, falling back to an existing legacy field plot during migration.
        prefix (str):
            Override the metadata prefix inferred from the model and recorded fit.
        mode (str):
            Number-formatting mode passed to Data.format. Defaults to float.
        **kwargs (dict):
            Text properties or annotation options, including arrowprops.

    Returns:
        Data or str:
            The same Data object after drawing, or formatted text when text_only=True.

    Raises:
        ValueError:
            If drawing is requested without an existing plot or explicit axes.

    Notes:
        Ordinary 2D text uses axes-fraction coordinates; annotation calls with
        arrowprops retain Matplotlib's coordinate options. 3D text uses data
        coordinates. Parameter labels/units may be recorded in metadata for
        formatting, including when text_only is True. Drawing creates no figure.
    """
    if isclass(model) and ((_lmfit and issubclass(model, Model)) or issubclass(model, odrModel)):
        model = model()  # Instantiate a bare class first

    if isinstance(model, odrModel):  # Get predix from odrModel
        model_prefix = model.meta.get("__name__", type(model).__name__)
        prefix = kwargs.pop("prefix", datafile.get("odr.prefix", model_prefix))
        param_names = model.meta.get("param_names", [])
        display_names = model.meta.get("display_names", param_names)
        units = model.meta.get("units", [""] * len(param_names))
    elif _lmfit and isinstance(model, Model):  # Get prefix from lmfit
        prefix = kwargs.pop("prefix", datafile.get("lmfit.prefix", type(model).__name__))
        param_names = model.param_names
        display_names = getattr(model, "display_names", model.param_names)
        units = getattr(model, "units", [""] * len(param_names))
    elif callable(model):  # Get prefix from callable name
        prefix = kwargs.pop("prefix", model.__name__)
        model = Model(model)
        param_names = model.param_names
        display_names = getattr(model, "display_names", model.param_names)
        units = getattr(model, "units", [""] * len(param_names))
    else:
        raise RuntimeError(f"model should be either an lmfit_mod.Model or a callable function, not a {type(model)}")

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

    try:  # if the model has an attribute display params then use these as the parameter names
        for k, display_name, unit in zip(param_names, display_names, units):
            if prefix:
                datafile[f"{prefix}{k} label"] = display_name
                datafile[f"{prefix}{k} units"] = unit

            else:
                datafile[f"{k} label"] = display_name
                datafile[f"{k} units"] = unit
    except (AttributeError, KeyError):
        pass

    text = "\n".join([datafile.format(f"{prefix}{k}", fmt="latex", mode=mode) for k in model.param_names])
    try:
        datafile[f"{prefix}chi^2 label"] = r"\chi^2_{red}"
        text += "\n" + datafile.format(f"{prefix}chi^2", fmt="latex", mode="eng")
    except KeyError:
        pass

    if not text_only:
        ax = kwargs.pop("ax", None)
        if ax is None:
            ax = datafile.plot.ax
        if ax is None:
            figure = datafile.__dict__.get("_figure")
            if figure is None or not figure.axes:
                raise ValueError("Plot data first or supply ax= to annotate_fit.")
            ax = figure.gca()
        if "zlim" in ax.properties():
            # 3D plot then
            if z is None:
                zb, zt = ax.properties()["zlim"]
                z = 0.5 * (zt - zb) + zb
            ax.text3D(x, y, z, text)
        elif "arrowprops" in kwargs:
            ax.annotate(text, xy=(x, y), **kwargs)
        else:
            kwargs.pop("xycoords", None)
            kwargs["transform"] = ax.transAxes
            ax.text(x, y, text, **kwargs)
        ret = datafile
    else:
        ret = text
    return ret


def curve_fit(datafile, func, xcol=None, ycol=None, sigma=None, **kwargs):
    """General curve fitting function passed through from scipy.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        func (callable, lmfit_mod.Model, odr.Model):
            The fitting function with the form def f(x,*p) where p is a list of fitting parameters
        xcol (index, Iterable):
            The index of the x-column data to fit. If list or other iterable sends a tuple of x columns to func
            for N-d fitting.
        ycol (index, list of indices or array):
            The index of the y-column data to fit. If an array, then should be 1D and
            the same length as the data. If ycol is a list of indices then the columns are iterated over in
            turn, fitting occurring for each one. In this case the return value is a list of what would be
            returned for a single column fit.

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try. See notes below.
        sigma (index):
            The index of the column with the y-error bars
        bounds (callable):
            A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
        where (Boolean array, pandas.Series or None):
            Rows to include, intersected with bounds and the finite, non-excluded
            inputs required by each y channel. Labelled selectors must match the
            complete row index exactly. None includes all rows.
        parameter_bounds (tuple):
            Lower and upper parameter limits forwarded to SciPy. The existing
            bounds argument continues to select observations.
        result (bool):
            Determines whether the fitted data should be added into the DataFile object. If result is True then
            the last column will be used. If result is a string or an integer then it is used as a column index.
            Default to None for not adding fitted data
        replace (bool):
            Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default
            False)
        header (string or None):
            If this is a string then it is used as the name of the fitted data. (default None)
        absolute_sigma (bool):
            If False, `sigma` denotes relative weights of the data points. The default True means that
            the sigma parameter contains absolute standard deviations when uncertainties are supplied.
        output (str, default "fit"):
            Specify what to return.
        **kwargs:
            Other arguments to pass to `curve_fit_result` or initial p0 values for curve_fit.

    Returns:
        (various):
            The return value is determined by the *output* parameter. Options are:
                * "fit"    (tuple of popt,pcov) Optimal values of the fitting parameters p, and the
                            variance-co-variance matrix for the fitting parameters.
                * "row"     just a one dimensional numpy array of the fit parameters interleaved with their
                            uncertainties
                * "full"    a tuple of (popt,pcov,dictionary of optional outputs, message, return code, row).
                * "data"   a copy of the :py:class:`Stoner.core.data.Data` object with the fit recorded in the
                            metadata and optionally as a new column.

    Notes:
        If the columns are not specified (or set to None) then the X and Y data are taken using the
        :py:attr:`Stoner.core.data.Data.setas` attribute.

        The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
        The x-column and y-column can be anything that :py:meth:`Stoner.core.data.Data.find_col` can use as an index
        but typucally either strings to be matched against column headings or integers.
        The initial parameter values and weightings default to None which corresponds to all parameters starting
        at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the
        point is to be used in the fit and false if not.


        The *absolute_sigma* keyword determines whether the returned covariance matrix `pcov` is based on
        *estimated* errors in the data, and is not affected by the overall magnitude of the values in `sigma`.
        Only the relative magnitudes of the *sigma* values matter.
        If True, `sigma` describes one standard deviation errors of the input data points. The estimated
        covariance in `pcov` is based on these values.

        The starting vector *p0* can be either a list, tuple or array, or a callable that will produce a list,
        tuple or array. If callable, it should take the form::

            def p0_func(ydata,x=xdata):
                ....

        and return a list of parameter values that is in the same order as the model function. If p0 is not
        given and a :py:class:`lmfit_mod.Model` or :py:class:`scipy.odr.Model` is supplied as the model function,
        then the model's estimates of the starting values will be used instead.


    See Also:
        *   :py:meth:`~Stoner.core.data.Data.lmfit`
        *   :py:meth:`~Stoner.core.data.Data.odr`
        *   :py:meth:`~Stoner.core.data.Data.differential_evolution`
        *   User guide section :ref:`curve_fit_guide`
    """
    settings = _Curve_Fit_Output()
    settings.result = kwargs.pop("result", None)
    settings.replace = kwargs.pop("replace", False)
    settings.header = kwargs.pop("header", None)
    settings.residuals = kwargs.pop("residuals", False)
    settings.prefix = kwargs.pop("prefix", None)

    # Support either scale_covar or absolute_sigma, the latter wins if both supplied
    # If neither are specified, then if sigma is not given, absolute sigma will be False.

    # Support both asrow and output, the latter wins if both supplied
    settings.output = kwargs.pop("output", "row" if kwargs.pop("asrow", False) else "fit")
    kwargs["full_output"] = True

    inputs = _curve_fit_inputs(datafile, xcol, ycol, sigma, kwargs.pop("where", None), kwargs.pop("bounds", None))
    if "parameter_bounds" in kwargs:
        kwargs["bounds"] = kwargs.pop("parameter_bounds")
    scale_covar = kwargs.pop("scale_covar", None)
    _func, initial = _get_curve_fit_func(func, kwargs)
    reports, retvals = [], []
    for xdat, ydat, uncertainty, columns, full_x, full_y, model_mask, residual_mask in inputs:
        p0 = initial
        if callable(p0):
            try:
                p0 = p0(ydat, xdat)
            except (ArithmeticError, RuntimeError, ValueError):
                p0 = None
        p0 = _curve_fit_p0_list(p0, func)
        parameter_count = len(p0) if p0 is not None else len(_get_model_parnames(func))
        if len(ydat) <= parameter_count:
            raise ValueError("A fit requires more eligible observations than free parameters")
        options = dict(kwargs)
        options.setdefault("absolute_sigma", not scale_covar if scale_covar is not None else uncertainty is not None)
        if p0 is not None:
            options["p0"] = p0
        if uncertainty is not None:
            options["sigma"] = uncertainty
        for var in ["xcol", "ycol", "zcol", "xerr", "yerr", "zerr"]:
            options.pop(var, None)
        popt, pcov, infodict, msg, ier = _curve_fit(_func, xdat, ydat, **options)
        report = _Curve_Fit_Result()
        report.settings = deepcopy(settings)
        report.settings.columns = columns
        report.func = _func
        report.results = {"popt": popt, "pcov": pcov, "mesg": msg, "ier": ier}
        report.infodict = infodict
        report.p0 = np.ones(len(report.popt)) if p0 is None else p0
        report.data = datafile
        report.residual_vals = ydat - _func(xdat, *popt)
        report.nfree = len(ydat) - len(popt)
        report.n_observations = len(ydat)
        weighted_residuals = report.residual_vals if uncertainty is None else report.residual_vals / uncertainty
        report.chisq = np.sum(weighted_residuals**2) / report.nfree
        report.full_x = full_x
        if settings.result is not None and settings.result is not False:
            report.model_values = _func(full_x, *popt)
            report.model_mask = model_mask | ~np.isfinite(report.model_values)
            report.full_residuals = full_y - report.model_values
            report.residual_mask = residual_mask | ~np.isfinite(report.full_residuals)
        reports.append(report)
        try:
            retvals.append(getattr(report, settings.output))
        except AttributeError as err:
            raise RuntimeError(f"Specified output: {settings.output}, from curve_fit not recognised") from err
    if settings.result is not None:
        owner = datafile._require_storage_owner()
        owner._check_writable()
        draft = datafile.clone
        for report in reports:
            _record_curve_fit_result(draft, func, report, report.settings)
        candidate = draft.export_storage()
        candidate.validate()
        owner._check_writable()
        owner._state = candidate
    return retvals[0] if len(retvals) == 1 else retvals


def differential_evolution(datafile, model, xcol=None, ycol=None, p0=None, sigma=None, **kwargs):
    """Fit model to the data using a differential evolution algorithm.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (lmfit_mod.Model):
            An instance of an lmfit_mod.Model that represents the model to be fitted to the data
        xcol (index or None):
            Columns to be used for the x  data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try.
            See the notes in :py:meth:`~Stoner.core.data.Data.curve_fit` for
            more details.
        sigma (index):
            The index of the column with the y-error bars
        bounds (callable):
            A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
        result (bool):
            Determines whether the fitted data should be added into the DataFile object. If result is True then
            the last column will be used. If result is a string or an integer then it is used as a column index.
            Default to None for not adding fitted data
        replace (bool):
            Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default
            False)
        header (string or None):
            If this is a string then it is used as the name of the fitted data. (default None)
        scale_covar (bool) :
            whether to automatically scale covariance matrix (leastsq only)
        output (str, default "fit"):
            Specify what to return.
        **kwargs:
            Other arguments to pass to `curve_fit_result` or initial p0 values for curve_fit.

    Returns:
        ( various ) :

            The return value is determined by the *output* parameter. Options are
                - "fit"    just the :py:class:`lmfit_mod.Model.ModelFit` instance that contains all relevant
                            information about the fit.
                - "row"     just a one dimensional numpy array of the fit parameters interleaved with their
                            uncertainties
                - "full"    a tuple of the fit instance and the row.
                - "data"    a copy of the :py:class:`Stoner.core.data.Data` object with the fit recorded in the
                            emtadata and optionally as a column of data.

    This function is essentially a wrapper around the :py:func:`scipy.optimize.differential_evolution` function
    that presents the same interface as the other Stoner package curve fitting functions. The parent function,
    however, does not provide the variance-covariance matrix to estimate the fitting errors. To work around this,
    this function does the initial fit with the differential evolution, but then uses that to give a starting
    vector to a call to :py:func:`scipy.optimize.curve_fit` to calculate the covariance matrix.

    See Also:
        -   :py:meth:`~Stoner.core.data.Data.curve_fit`
        -   :py:meth:`~Stoner.core.data.Data.lmfit`
        -   :py:meth:`~Stoner.core.data.Data.odr`
        -   User guide section :ref:`curve_fit_guide`

    Example:
        .. plot:: samples/differential_evolution_simple.py
            :include-source:
            :outname: diffev1
    """
    settings = _Curve_Fit_Output()
    bounds = kwargs.pop("bounds", lambda x, y: True)
    settings.result = kwargs.pop("result", None)
    settings.replace = kwargs.pop("replace", False)
    settings.residuals = kwargs.pop("residuals", False)
    settings.header = kwargs.pop("header", None)
    # Support both asrow and output, the latter wins if both supplied
    settings.asrow = kwargs.pop("asrow", False)
    settings.output = kwargs.pop("output", "row" if settings.asrow else "fit")

    data, kwargs, settings.columns = _assemnle_data_to_fit(
        datafile, xcol=xcol, ycol=ycol, sigma=sigma, bounds=bounds, **kwargs
    )
    model, settings.prefix = _prep_lmfit_model(model, kwargs)
    p0, single_fit = _prep_lmfit_p0(model, data.y, data.x, p0, kwargs)

    for k in model.param_names:
        kwargs.pop(k, None)

    diff_model = MimizerAdaptor(model, params=p0)

    kwargs.setdefault("polish", True)
    for arg in ["xcol", "ycol", "zcol", "xerr", "yerr", "zerr", "sigma", "sigma_x", "sigma_z"]:
        kwargs.pop(arg, [])
    abs_sigma = kwargs.pop("absolute_sigma", False)

    if not single_fit:
        raise NotImplementedError("Sorry chi^2 mapping not implemented for differential evolution yet.")
    fit = _differential_evolution(
        diff_model.minimize_func,
        diff_model.bounds,
        args=(
            data.x,
            data.y,
            data.e,
        ),
        **kwargs,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    kwargs.pop("polish", None)
    kwargs["full_output"] = True
    kwargs["absolute_sigma"] = abs_sigma
    popt, pcov, infodict, mesg, ier = _curve_fit(model.func, data.x, data.y[0], sigma=data.e[0], p0=fit.x, **kwargs)
    polish = _Curve_Fit_Result()
    polish.results = {"popt": popt, "pcov": pcov, "mesg": mesg, "ier": ier}
    polish.infodict = infodict
    polish.data = datafile
    polish.settings = settings

    polish.func = model.func
    polish.p0 = p0
    polish.residual_vals = data.y - polish.fvec
    polish.chisq = (polish.residual_vals**2).sum()
    polish.nfree = len(datafile) - len(polish.popt)
    polish.chisq /= polish.nfree

    model.popt = polish.popt
    fit.covar = polish.pcov
    fit.popt = polish.popt
    fit.perr = polish.perr
    fit.fit_report = polish.fit_report
    row = _record_curve_fit_result(datafile, model, fit, settings)
    ret_dict = {}
    for i, p in enumerate(model.param_names):
        ret_dict[p], ret_dict[f"d_{p}"] = row[2 * i], row[2 * i + 1]
    ret_dict["chi-square"] = polish.chisqr
    ret_dict["red. chi-sqr"] = polish.redchi
    ret_dict["nfev"] = fit.nfev
    fit.dict = ret_dict

    retval = {
        "fit": (row[::2], fit.covar),
        "report": fit,
        "row": row,
        "full": (fit, row),
        "data": datafile,
        "dict": fit.dict,
    }
    if settings.output not in retval:
        raise RuntimeError(f"Failed to recognise output format:{settings.output}")
    return retval[settings.output]


def lmfit(datafile, model, xcol=None, ycol=None, p0=None, sigma=None, **kwargs):
    r"""Wrap the lmfit module fitting.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (lmfit_mod.Model):
            An instance of an lmfit_mod.Model that represents the model to be fitted to the data
        xcol (index or None):
            Columns to be used for the x  data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try.
            See the notes in :py:meth:`~Stoner.core.data.Data.curve_fit` for
            more details.
        sigma (index):
            The index of the column with the y-error bars
        bounds (callable):
            A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
        result (bool):
            Determines whether the fitted data should be added into the DataFile object. If result is True then
            the last column will be used. If result is a string or an integer then it is used as a column index.
            Default to None for not adding fitted data
        replace (bool):
            Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default
            False)
        header (string or None):
            If this is a string then it is used as the name of the fitted data. (default None)
        scale_covar (bool) :
            whether to automatically scale covariance matrix (leastsq only)
        output (str, default "fit"):
            Specify what to return.
        **kwargs:
            Other arguments to pass to `curve_fit_result` or initial p0 values for curve_fit.

    Returns:
        ( various ) :
            The return value is determined by the *output* parameter. Options are
                - "fit"    just the :py:class:`lmfit_mod.Model.ModelFit` instance that contains all relevant
                            information about the fit.
                - "row"     just a one dimensional numpy array of the fit parameters interleaved with their
                            uncertainties
                - "full"    a tuple of the fit instance and the row.
                - "data"    a copy of the :py:class:`Stoner.core.data.Data` object with the fit recorded in the
                            emtadata and optionally as a column of data.

    See Also:
        -   :py:meth:`~Stoner.core.data.Data.curve_fit`
        -   :py:meth:`~Stoner.core.data.Data.odr`
        -   :py:meth:`~Stoner.core.data.Data.differential_evolution`
        -   User guide section :ref:`fitting_with_limits`

    .. note::

       If *p0* is fed a 2D array, then it assumed that you want to calculate :math:`\chi^2` for different
       starting parameters with some variables fixed. In this mode, fitting is carried out repeatedly with each
       row representing one attempt with different values of the parameters. In this mode the return value is
       a 2D array whose rows correspond to the inputs to the rows of p0, the columns are the fitted values of the
       parameters with an additional column for :math:`\chi^2`.

    Example:
        .. plot:: samples/lmfit_simple.py
            :include-source:
            :outname: lmfit2
    """
    settings = _Curve_Fit_Output()
    settings.result = kwargs.pop("result", None)
    settings.replace = kwargs.pop("replace", False)
    settings.residuals = kwargs.pop("residuals", False)
    settings.header = kwargs.pop("header", None)
    # Support both asrow and output, the latter wins if both supplied
    settings.asrow = kwargs.pop("asrow", False)
    settings.output = kwargs.pop("output", "row" if settings.asrow else "fit")

    data, kwargs, settings.columns = _assemnle_data_to_fit(datafile, xcol=xcol, ycol=ycol, yerr=sigma, **kwargs)
    model, settings.prefix = _prep_lmfit_model(model, kwargs)
    p0, single_fit = _prep_lmfit_p0(model, data.y, data.x, p0, kwargs)
    settings.nan_policy = kwargs.pop("nan_policy", getattr(model, "nan_policy", "omit"))
    settings.scale_covar = kwargs.get("scale_covar", not kwargs.get("absolute_sigma", False))

    if single_fit:
        return __lmfit_one(
            datafile,
            model,
            data,
            p0,
            settings,
        )
    # chi^2 mode
    pn = p0
    ret_val = np.zeros((pn.shape[0], pn.shape[1] * 2 + 1))
    output = settings.output
    settings.output = "row"
    for i, pn_i in enumerate(pn):  # iterate over every row in the supplied p0 values
        p0, single_fit = _prep_lmfit_p0(
            model, data.y, data.x, pn_i, kwargs
        )  # model, data, params, prefix, columns, scale_covar,**kwargs)
        ret_val[i, :] = __lmfit_one(
            datafile,
            model,
            data,
            p0,
            settings,
        )
    if output == "data":  # Create a data object and seet column headers etc correctly
        return _chi2_fit_to_data(datafile, ret_val, model)
    return ret_val


def polyfit(
    datafile,
    xcol=None,
    ycol=None,
    polynomial_order=2,
    bounds=lambda x, y: True,
    result=None,
    replace=False,
    header=None,
):
    """Pass through to numpy.polyfit.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        xcol (index):
            Index to the column in the data with the X data in it
        ycol (index):
            Index to the column int he data with the Y data in it
        polynomial_order (int):
            Order of polynomial to fit (default 2)
        bounds (callable):
            A function that evaluates True if the current row should be included in the fit
        result (index or None):
            Add the fitted data to the current data object in a new column (default don't add)
        replace (bool):
            Overwrite or insert new data if result is not None (default False)
        header (string or None):
            Name of column_header of replacement data. Default is construct a string from the y column
            headser and polynomial order.

    Returns:
        (numpy.poly):
            The best fit polynomial as a numpy.poly object.

    Notes:
        If the x or y columns are not specified (or are None) the the setas attribute is used instead.

        This method is deprecated and may be removed in a future version in favour of the more general
            curve_fit
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol, scalar=False)

    working = datafile.search(_.xcol, bounds)
    if not isiterable(_.ycol):
        _.ycol = [_.ycol]
    p = np.zeros((len(_.ycol), polynomial_order + 1))
    if isinstance(result, bool) and result:
        result = datafile.shape[1]
    for i, ycolumn in enumerate(_.ycol):
        p[i, :] = np.polyfit(
            working[:, datafile.find_col(_.xcol)], working[:, datafile.find_col(ycolumn)], polynomial_order
        )
        if result:
            if header is None:
                header = (
                    f"Fitted {datafile.column_headers[datafile.find_col(ycolumn)]} with "
                    + f"{ordinal(polynomial_order)} order polynomial"
                )
            datafile.add_column(
                np.polyval(p[i, :], x=datafile.column(_.xcol)), index=None if result is True else result, replace=replace, header=header
            )
    if len(_.ycol) == 1:
        p = p[0, :]
    datafile[f"{ordinal(polynomial_order)}-order polyfit coefficients"] = list(p)
    return p


def odr(datafile, model, xcol=None, ycol=None, **kwargs):
    """Wrap the scipy.odr orthogonal distance regression fitting.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (scipy.odr.Model, lmfit_mod.Models.Model or callable):
            The model that describes the data. See below for more details.
        xcol (index or None):
            Columns to be used for the x  data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not given defaults to the
            :py:attr:`Stoner.core.data.Data.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try.
            See the notes to :py:meth:`~Stoner.core.data.Data.curve_fit` for
            more details.
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
            Inidcatesa whether the fitted data replaces existing data or is inserted as a new column
            (default False)
        header (string or None):
            If this is a string then it is used as the name of the fitted data. (default None)
        output (str, default "fit"):
            Specify what to return.
        **kwargs:
            Other arguments to pass to `_odr_one` or initial p0 values for curve_fit.

    Returns:
        ( various ) :
            The return value is determined by the *output* parameter. Options are
                - "fit"    just the :py:class:`scipy.odr.Output` instance (default)
                - "row"     just a one dimensional numpy array of the fit parameters interleaved with their
                            uncertainties
                - "full"    a tuple of the fit instance and the row.
                - "data"    a copy of the :py:class:`Stoner.core.data.Data` object with the fit recorded in the
                            metadata and optionally as a column of data.

    Notes:
        The function tries to make use of whatever model you give it. Specifically, it accepts:

            -   A subclass or an instance of :py:class:`scipy.odr.Model` : this is the native model type for the
                underlying scipy odr package.
            -   A subclass or instance of an lmfit_mod.Models.Model: the :py:mod:`Stoner.analysis.fitting.models`
                package has a number of useful prebuilt lmfit models that can be used directly by this function.
            -   A callable function which should have a signature f(x,parameter1,parameter2...) and *not* the
                scip.odr standard f(beta,x)

        This function is designed to be as compatible as possible with :py:meth:`~Stoner.core.data.Data.curve_fit` and
        :py:meth:`~Stoner.core.data.Data.lmfit` to facilitate switching between them.

    See Also:
        -   :py:meth:`~Stoner.core.data.Data.curve_fit`
        -   :py:meth:`~Stoner.core.data.Data.lmfit`
        -   :py:meth:`~Stoner.core.data.Data.differential_evolution`
        -   User guide section :ref:`fitting_with_limits`

    Example:
        .. plot:: samples/odr_simple.py
             :include-source:
             :outname: odrfit1
    """
    # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
    # Support both asrow and output, the latter wins if both supplied
    settings = _Curve_Fit_Output()
    settings.result = kwargs.pop("result", None)
    settings.replace = kwargs.pop("replace", False)
    settings.residuals = kwargs.pop("residuals", False)
    settings.header = kwargs.pop("header", None)
    # Support both asrow and output, the latter wins if both supplied
    settings.asrow = kwargs.pop("asrow", False)
    settings.output = kwargs.pop("output", "row" if settings.asrow else "fit")
    sigma = kwargs.pop("sigma", None)
    sigma_x = kwargs.pop("sigma_x", None)
    bounds = kwargs.pop("bounds", lambda x, r: True)
    p0 = kwargs.pop("p0", None)
    data, kwargs, settings.columns = _assemnle_data_to_fit(
        datafile, xcol=xcol, ycol=ycol, yerr=sigma, bounds=bounds, sigma_x=sigma_x, **kwargs
    )
    if not isinstance(model, odrModel):
        model, settings.prefix = _prep_lmfit_model(model, kwargs)
    else:
        settings.prefix = kwargs.pop("prefix", getattr(model, "name", model.fcn.__name__))
    p0, single_fit = _prep_lmfit_p0(model, data.y, data.x, p0, kwargs)
    model = ODR_Model(model, p0=p0)
    if kwargs.get("scale_covar", True):
        if np.all(np.isclose(data.d, data.d.ravel()[0])):
            wd = None
        else:
            wd = 1 / data.d[0] ** 2
        if np.all(np.isclose(data.e, data.e.ravel()[0])):
            we = None
        else:
            we = 1 / data.e**2
        data = sp.odr.Data(data.x, data.y, wd=wd, we=we)
    else:
        data = sp.odr.RealData(data.x, data.y, sx=data.d, sy=data.e)

    if single_fit:
        ret_val = _odr_one(datafile, data, model, settings, p0, **kwargs)
    else:  # chi^2 mode
        raise NotImplementedError("Sorry cannot do chi^2 mode for orthogonal distance regression yet!")
    return ret_val
