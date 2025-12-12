#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fitting Functions and classes to mixin for :py:class:`Stoner.Data`."""

from collections.abc import Mapping
from inspect import getfullargspec, isclass

import lmfit as lmfit_mod
import numpy as np
from numpy import ma
import scipy as sp
from lmfit.model import Model
from scipy.odr import Model as odrModel
from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize import differential_evolution as _differential_evolution

from ...compat import index_types, string_types
from ...tools import isiterable, islistlike, isnone, ordinal
from .classes import (
    MimizerAdaptor,
    ODR_Model,
    _curve_fit_result,
    _prep_lmfit_model,
    _prep_lmfit_p0,
)

_lmfit = True


def _get_model_parnames(model):
    """Get a list of the model parameter names."""
    if isinstance(model, type) and (issubclass(model, Model) or issubclass(model, odrModel)):
        model = Model(func=model.func)

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


def _get_curve_fit_data(datafile, xcol, ycol, bounds, sigma):
    """Gather up the xdata and sigma columns for curve_fit."""
    working = datafile.search(xcol, bounds)
    working = ma.mask_rowcols(working, axis=0)
    if not isnone(sigma):
        sigma = working[:, datafile.find_col(sigma)]
    else:
        sigma = None
    if isinstance(xcol, string_types):
        xdat = working[:, datafile.find_col(xcol)]
    elif isiterable(xcol):
        for ix, c in enumerate(xcol):
            if ix == 0:
                xdat = working[:, datafile.find_col(c)]
            else:
                xdat = np.column_stack((xdat, working[:, datafile.find_col(c)]))
    else:
        xdat = working[:, datafile.find_col(xcol)]

    for i, yc in enumerate(ycol):
        if isinstance(yc, index_types):
            ydat = working[:, datafile.find_col(yc)]
        elif isinstance(yc, np.ndarray) and yc.ndim == 1 and len(yc) == len(datafile):
            ydat = yc
        else:
            raise RuntimeError(
                """Y-data for fitting not defined - should either be an index or a 1D numpy array of the same
                length as the dataset"""
            )
        if i == 0:
            ydata = np.atleast_2d(ydat)
        else:
            ydata = np.vstack([ydata, ydat])

    return xdat, ydata, sigma


def _assemnle_data_to_fit(datafile, xcol, ycol, sigma, bounds, scale_covar, sigma_x=None):
    """Marshall the data for doing a curve_fit or equivalent.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        xcol (index):
            Column with xdata in it
        ycol(index):
            Column with ydata in it
        sigma (index or array-like):
            column of y-errors or uncertainty values.
        bounds (callable):
            Used to select the data rows to fit
        scale_covar (bool,None):
            If set the flag to scale the covariance.
        sigma_x (index or array-like):
            column of x-errors or uncertainty values.

    Returns:
        (data,scale_covar,col_assignments):
            data is a tuple of (x,y,sigma). scale_covar is False if sigma is real errors.
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol)
    working = datafile.search(_.xcol, bounds)
    working = ma.mask_rowcols(working, axis=0)
    xdata = working[:, datafile.find_col(_.xcol)]
    ydata = working[:, datafile.find_col(_.ycol)]
    # Now check for sigma_y and sigma_x and have them default to sigma (which in turn defaults to None)

    if sigma is not None:
        if isinstance(sigma, index_types):
            _ = datafile._col(xcol=xcol, ycol=ycol, yerr=sigma)
            sigma = working[:, datafile.find_col(sigma)]
        elif isinstance(sigma, (list, tuple)):
            sigma = ma.array(sigma)
        elif isinstance(sigma, np.ndarray):
            sigma = ma.array(sigma)  # ensure masked
        else:
            raise RuntimeError("Sigma should have been a column index or list of values")
    elif not isnone(_.yerr):
        sigma = working[:, datafile.find_col(_.yerr)]
    else:
        sigma = 1.0
        scale_covar = False

    if sigma_x is None:
        if _.xerr is None:
            sigma_x = sigma
        else:
            sigma_x = working[:, datafile.find_col(_.xerr)]
    else:
        if isinstance(sigma_x, index_types):
            _ = datafile._col(xcol=xcol, ycol=ycol, yerr=_.yerr, xerr=sigma_x)
            sigma_x = working[:, datafile.find_col(sigma_x)]
        elif isinstance(sigma_x, (list, tuple)):
            sigma_x = ma.array(sigma_x)
        elif isinstance(sigma_x, np.ndarray):
            sigma_x = ma.array(sigma_x)  # ensure masked
        else:
            raise RuntimeError("Sigma_x should have been a column index or list of values")

    mask = np.invert(ydata.mask)
    if isinstance(sigma, np.ndarray):
        sigma = sigma[mask]
        sigma_x = sigma_x[mask]
    ydata = ydata[mask]
    xdata = xdata[mask]  # lmfit doesn't seem to work well with masked data - here we just delete masked points

    return (xdata, ydata, sigma, sigma_x), scale_covar, _


def __lmfit_one(
    datafile,
    model,
    data,
    params,
    prefix,
    columns,
    scale_covar,
    replace=False,
    result=False,
    header="",
    residuals=False,
    output="row",
    nan_policy="raise",
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
        prefix (str):
            Prefix for labels in metadata
        columns (attribute dict):
            Column assignments
        scale_covar (bool):
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
        residuals (bool):
            Whether to also include residuals (default False)
        nan_policy (str):
            What to do about nan's in the fit - passed to lmfit (default "raise")

    Returns:
        (various):
            Results froma  fit or raises and exception.
    """
    if not _lmfit:
        raise RuntimeError("lmfit module not available.")

    kwargs = {}
    kwargs[model.independent_vars[0]] = data[0]
    fit = model.fit(data[1], params, scale_covar=scale_covar, weights=1.0 / data[2], nan_policy=nan_policy, **kwargs)
    if fit.success:
        row = _record_curve_fit_result(
            datafile,
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
        if output not in retval:
            raise RuntimeError(f"Failed to recognise output format:{output}")
        else:
            return retval[output]
    else:
        raise RuntimeError(f"Failed to complete fit. Error was:\n{fit.lmdif_message}\n{fit.message}")


def _record_curve_fit_result(
    datafile, func, fit, xcol, header, result, replace, residuals=False, ycol=None, prefix=None
):
    """Annotate the DataFile object with the curve_fit result."""
    if isinstance(func, (lmfit_mod.Model)):
        f_name = type(func).__name__
        labels = getattr(type(func), "labels", None)
        units = getattr(type(func), "units", None)
        func = func.func
        args = getfullargspec(func)[0]  # pylint: disable=W1505
        if args[0] in ["datafile", "self"]:
            del args[0]
        if len(args) > 0:
            del args[0]

    elif isclass(func) and issubclass(func, lmfit_mod.Model):
        f_name = func.__name__
        labels = getattr(func, "labels", None)
        units = getattr(func, "units", None)
        args = getfullargspec(func)[0]  # pylint: disable=W1505
        func = func.func
        if args[0] == "datafile":
            del args[0]
        if len(args) > 0:
            del args[0]

    elif isinstance(func, (sp.odr.Model)):
        f_name = func.meta["name"]
        labels = getattr(func, "labels", None)
        units = getattr(func, "units", None)
        args = func.meta["param_names"]
        model = func

        def _func(x, *beta):
            return model.fcn(beta, x)

        func = _func
    else:
        f_name = func.__name__
        labels = getattr(func, "labels", None)
        units = getattr(func, "units", None)
        args = getfullargspec(func)[0]  # pylint: disable=W1505
        if args[0] in ["datafile", "self"]:
            del args[0]
        if len(args) > 0:
            del args[0]

    if prefix is not None:
        f_name = prefix

    if labels is None:
        labels = args
    if units is None:
        units = [""] * len(args)
    if isinstance(fit, _curve_fit_result):  # Come from curve_fit
        popt = fit.popt
        perr = fit.perr
        nfev = fit.nfev
        chisq = fit.chisq
    elif isinstance(fit, lmfit_mod.model.ModelResult):  # Come form an lmfit operation
        popt = [fit.params[x].value for x in args]
        perr = [fit.params[x].stderr for x in args]
        nfev = fit.nfev
        chisq = fit.redchi
    elif isinstance(fit, sp.odr.Output):
        popt = fit.beta
        perr = fit.sd_beta
        delta, eps = fit.delta, fit.eps
        nfree = len(delta) - len(popt)
        chisq = np.sum((delta**2 + eps**2)) / nfree
        nfev = None
    elif isinstance(fit, sp.optimize.OptimizeResult):
        popt = fit.popt
        perr = fit.perr
        nfev = fit.nfev
        nfree = len(datafile) - len(popt)
        data = datafile.data[:, ycol]
        fit_data = func(datafile.data[:, xcol], *popt)
        chisq = np.sum((data - fit_data) ** 2) / nfree
    else:
        raise RuntimeError("Unable to understand {type(fit)} as a fitting result")

    for val, err, name, label, unit in zip(popt, perr, args, labels, units):
        datafile[f"{f_name}:{name}"] = val
        datafile[f"{f_name}:{name} err"] = err
        datafile[f"{f_name}:{name} label"] = datafile.metadata.get(f"{f_name}:{name} label", label)
        datafile[f"{f_name}:{name} unit"] = datafile.metadata.get(f"{f_name}:{name} unit", unit)

    if not isinstance(header, string_types):
        header = "Fitted with " + func.__name__

    # Store our current mask, calculate new column's mask and turn off mask
    tmp_mask = datafile.mask
    col_mask = np.any(tmp_mask, axis=1)
    datafile.mask = False

    if isinstance(result, bool) and result:  # Appending data to end of data
        result = datafile.shape[1]
        tmp_mask = np.column_stack((tmp_mask, col_mask))
    else:  # Inserting data
        tmp_mask = np.column_stack((tmp_mask[:, 0:result], col_mask, tmp_mask[:, result:]))
    if islistlike(xcol):
        new_col = func(datafile[:, xcol], *popt)
    else:
        new_col = func(datafile.column(xcol), *popt)
    if result:
        datafile.add_column(new_col, index=result, replace=replace, header=header)
    if residuals and result:
        if not islistlike(ycol):
            ycol = [ycol]
        for yc in ycol:
            residual_vals = datafile.column(yc) - new_col
            if isinstance(residuals, bool) and residuals:
                if result is None:
                    residuals_idx = None
                else:
                    residuals_idx = datafile.find_col(result) + 1
            else:
                residuals_idx = residuals
            datafile.add_column(residual_vals, index=residuals_idx, replace=False, header=header + ":residuals")
            datafile[f"{f_name}:mean residual"] = np.mean(residual_vals)
            datafile[f"{f_name}:std residual"] = np.std(residual_vals)
            datafile[f"{f_name}:chi^2"] = chisq
            datafile[f"{f_name}:chi^2 err"] = np.sqrt(2 / len(residual_vals)) * chisq
    if nfev is not None:
        datafile[f"{f_name}:nfev"] = nfev

    datafile.mask = tmp_mask
    # Make row object
    row = []
    ch = []
    for v, e, a in zip(popt, perr, args):
        row.extend([v, e])
        ch.extend([a, f"{a} stderr"])
    row.append(chisq)
    ch.append("$\\chi^2$")
    cls = type(datafile.data)
    row = cls(row)
    row.column_headers = ch
    return row


def _odr_one(
    datafile,
    data,
    model,
    prefix,
    _,
    result=None,
    replace=False,
    header=None,
    residuals=False,
    asrow=False,
    output=None,
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
        prefix (str):
            Prefix for labels in metadata

    Keyword Arguments:
        result (bool,str):
            Where the result goes
        header (str):
            Name of new data column if used
        replace (bool):
            whether to add new dataa
        residuals (bool):
            Also calcualte the residuals from the fit - default False.
        asrow (bool):
            Return the fit results as a single 1D row array - default False.
        output (str):
            What to return - default depends on asrow for either a row or the fit.
        p0 (Iterable):
            Estimated values for model. If not given, model.estimate is used.
        **kwargs:
            Other parameters to try to create a p0 with.

    Returns:
        (various):
            Results froma  fit or raises and exception.
    """
    if output is None:
        output = "row" if asrow else "fit"

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
        fit_result.redchi = fit_result.sum_square / (
            len(fit_result.y) - len(fit_result.beta)
        )  # pylint: disable=no-member
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
    _record_curve_fit_result(
        datafile, model, fit_result, _.xcol, header, result, replace, residuals, ycol=_.ycol, prefix=prefix
    )

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
    if output not in retval:
        raise RuntimeError(f"Failed to recognise output format:{output}")
    return retval[output]


def annotate_fit(datafile, model, x=None, y=None, z=None, text_only=False, mode="float", **kwargs):
    """Annotate a plot with some information about a fit.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (callable or lmfit_mod.Model):
            The function/model used to describe the fit to be annotated.

    Keyword Parameters:
        x (float):
            x coordinate of the label
        y (float):
            y coordinate of the label
        z (float):
            z co-ordinbate of the label if the current axes are 3D
        prefix (str):
            The prefix placed ahead of the model parameters in the metadata.
        text_only (bool):
            If False (default), add the text to the plot and return the current object, otherwise,
            return just the text and don't add to a plot.
        prefix(str):
            If given  overridges the prefix from the model to determine a prefix to the parameter names in the
            metadata
        mode (str):
            Formatting mode to use for numbers in fit - see :py:func:`Stoner.core.functions.format`.
        **kwargs:
            Other keyword arguments, such as `prefix`, `arrowprops`.

    Returns:
        (Datam, str):
            A copy of the current Data instance if text_only is False, otherwise returns the text.

    If *prefix* is not given, then the first prefix in the metadata lmfit.prefix is used if present,
    otherwise a prefix is generated from the model.prefix attribute. If *x* and *y* are not specified then they
    are set to be 0.75 * maximum x and y limit of the plot.
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

    try:  # if the model has an attribute display params then use these as the parameter anmes
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
        datafile[f"{prefix}chi^2 label"] = r"\chi^2"
        text += "\n" + datafile.format(f"{prefix}chi^2", fmt="latex", mode=mode)
    except KeyError:
        pass

    if not text_only:
        ax = datafile.fig.gca()
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
            the sigma parameter is the reciprocal of the absolute standard deviation.
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
                * "data"   a copy of the :py:class:`Stoner.Core.DataFile` object with fit recorded in the
                            metadata and optionally as a new column.

    Note:
        If the columns are not specified (or set to None) then the X and Y data are taken using the
        :py:attr:`Stoner.Core.DataFile.setas` attribute.

        The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
        The x-column and y-column can be anything that :py:meth:`Stoner.Core.DataFile.find_col` can use as an index
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
        tuple or array. IF callable, it should take the form:

            def p0_func(ydata,x=xdata):
                ....

        and return a list of parameter values that is in the same order as the model function. If p0 is not
        given and a :py:class:`lmfit_mod.Model` or :py:class:`scipy.odr.Model` is supplied as the model function,
        then the model's estimates of the starting values will be used instead.


    See Also:
        *   :py:meth:`Stoner.Data.lmfit`
        *   :py:meth:`Stoner.Data.odr`
        *   :py:meth:`Stoner.Data.differential_evolution`
        *   User guide section :ref:`curve_fit_guide`
    """
    _ = datafile._col_args(scalar=False, xcol=xcol, ycol=ycol, yerr=sigma)
    xcol, ycol, sigma = _.xcol, _.ycol, _.yerr

    bounds = kwargs.pop("bounds", lambda x, y: True)
    result = kwargs.pop("result", None)
    replace = kwargs.pop("replace", False)
    header = kwargs.pop("header", None)
    residuals = kwargs.pop("residuals", False)
    prefix = kwargs.pop("prefix", None)

    # Support either scale_covar or absolute_sigma, the latter wins if both supplied
    # If neither are specified, then if sigma is not given, absolute sigma will be False.

    scale_covar = kwargs.pop("scale_covar", sigma is not None)
    absolute_sigma = kwargs.pop("absolute_sigma", not scale_covar)
    # Support both asrow and output, the latter wins if both supplied
    asrow = kwargs.pop("asrow", False)
    output = kwargs.pop("output", "row" if asrow else "fit")
    kwargs["full_output"] = True

    if not isinstance(ycol, list):
        ycol = [ycol]

    xdat, ydata, sigma = _get_curve_fit_data(datafile, xcol, ycol, bounds, sigma)

    # Support any of our alternatives for the fitting function
    if isinstance(func, type) and issubclass(func, (Model, sp.odr.Model)):
        func = func()
    if isinstance(func, sp.odr.Model):  # scipy othrothogonal model hack

        def _func(x, *beta):
            return func.fcn(beta, x)

        p0 = kwargs.pop("p0", func.estimate)
    elif isinstance(func, Model):
        _func = func.func
        try:
            if "p0" not in kwargs:  # Avoid expensive guess if we have a p0
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
        p0 = kwargs.pop("p0", pguess)
    elif callable(func):
        _func = func
        p0 = kwargs.pop("p0", None)
    else:
        raise TypeError(
            "".join(
                [
                    "curve_fit parameter 1 must be either a Model class from",
                    f" lmfit or scipy.odr, or a callable, not a {type(func)}",
                ]
            )
        )

    if callable(p0):  # Allow the user to supply p0 as a callanble function
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
        if isinstance(sigma, np.ndarray) and sigma.shape[0] > 1:
            if sigma.shape[0] == len(ycol):
                s = sigma[i]
            elif sigma.ndim == 2 and sigma.shape[1] == len(ycol):
                s = sigma[:, i]
            else:
                s = sigma  # probably this will fail!
        else:
            s = sigma
        report = _curve_fit_result(
            *_curve_fit(_func, xdat, ydat, p0=p0, sigma=s, absolute_sigma=absolute_sigma, **kwargs)
        )
        report.func = func
        if p0 is None:
            report.p0 = np.ones(len(report.popt))
        else:
            report.p0 = p0
        report.data = datafile
        report.residual_vals = ydata - report.fvec
        report.chisq = (report.residual_vals**2).sum()
        report.nfree = len(datafile) - len(report.popt)
        report.chisq /= report.nfree

        if result is not None:
            _record_curve_fit_result(
                datafile, func, report, xcol, header, result, replace, residuals=residuals, ycol=ycol, prefix=prefix
            )
        try:
            retvals.append(getattr(report, output))
        except AttributeError as err:
            raise RuntimeError(f"Specified output: {kwargs['output']}, from curve_fit not recognised") from err
    if i == 0:
        retvals = retvals[0]
    return retvals


def differential_evolution(datafile, model, xcol=None, ycol=None, p0=None, sigma=None, **kwargs):
    """Fit model to the data using a differential evolution algorithm.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (lmfit_mod.Model):
            An instance of an lmfit_mod.Model that represents the model to be fitted to the data
        xcol (index or None):
            Columns to be used for the x  data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try. See the notes in :py:meth:`Stoner.Data.curve_fit` for
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
                - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the
                            emtadata and optionally as a column of data.

    This function is essentially a wrapper around the :py:func:`scipy.optimize.differential_evolution` function
    that presents the same interface as the other Stoner package curve fitting functions. The parent function,
    however, does not provide the variance-covariance matrix to estimate the fitting errors. To work around this,
    this function does the initial fit with the differential evolution, but then uses that to give a starting
    vector to a call to :py:func:`scipy.optimize.curve_fit` to calculate the covariance matrix.

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
    bounds = kwargs.pop("bounds", lambda x, y: True)
    result = kwargs.pop("result", None)
    replace = kwargs.pop("replace", False)
    residuals = kwargs.pop("residuals", False)
    header = kwargs.pop("header", None)
    # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
    absolute_sigma = kwargs.pop("absolute_sigma", True)
    scale_covar = kwargs.pop("scale_covar", not absolute_sigma)
    # Support both asrow and output, the latter wins if both supplied
    asrow = kwargs.pop("asrow", False)
    output = kwargs.pop("output", "row" if asrow else "fit")

    data, scale_covar, _ = _assemnle_data_to_fit(datafile, xcol, ycol, sigma, bounds, scale_covar)
    data = data[0:3]
    model, prefix = _prep_lmfit_model(model, kwargs)
    p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kwargs)

    for k in model.param_names:
        kwargs.pop(k, None)

    diff_model = MimizerAdaptor(model, params=p0)

    kwargs["polish"] = kwargs.get("polish", True)

    if not single_fit:
        raise NotImplementedError("Sorry chi^2 mapping not implemented for differential evolution yet.")
    fit = _differential_evolution(diff_model.minimize_func, diff_model.bounds, data, **kwargs)
    if not fit.success:
        raise RuntimeError(fit.message)
    kwargs.pop("polish", None)
    kwargs["full_output"] = True
    polish = _curve_fit_result(
        *_curve_fit(model.func, data[0], data[1], sigma=data[2], p0=fit.x, absolute_sigma=not scale_covar, **kwargs)
    )

    polish.func = model.func
    polish.p0 = p0
    polish.data = datafile
    polish.residual_vals = data[1] - polish.fvec
    polish.chisq = (polish.residual_vals**2).sum()
    polish.nfree = len(datafile) - len(polish.popt)
    polish.chisq /= polish.nfree

    model.popt = polish.popt
    fit.covar = polish.pcov
    fit.popt = polish.popt
    fit.perr = polish.perr
    fit.fit_report = polish.fit_report
    row = _record_curve_fit_result(
        datafile, model, fit, _.xcol, header, result, replace, residuals=residuals, prefix=prefix, ycol=_.ycol
    )
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
    if output not in retval:
        raise RuntimeError(f"Failed to recognise output format:{output}")
    return retval[output]


def lmfit(datafile, model, xcol=None, ycol=None, p0=None, sigma=None, **kwargs):
    r"""Wrap the lmfit module fitting.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        model (lmfit_mod.Model):
            An instance of an lmfit_mod.Model that represents the model to be fitted to the data
        xcol (index or None):
            Columns to be used for the x  data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try. See the notes in :py:meth:`Stoner.Data.curve_fit` for
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
                - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the
                            emtadata and optionally as a column of data.

    See Also:
        -   :py:meth:`Stoner.Data.curve_fit`
        -   :py:meth:`Stoner.Data.odr`
        -   :py:meth:`Stoner.Data.differential_evolution`
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
    bounds = kwargs.pop("bounds", lambda x, y: True)
    result = kwargs.pop("result", None)
    replace = kwargs.pop("replace", False)
    residuals = kwargs.pop("residuals", False)
    header = kwargs.pop("header", None)
    # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
    absolute_sigma = kwargs.pop("absolute_sigma", True)
    scale_covar = kwargs.pop("scale_covar", not absolute_sigma)
    # Support both asrow and output, the latter wins if both supplied
    asrow = kwargs.pop("asrow", False)
    output = kwargs.pop("output", "row" if asrow else "fit")

    data, scale_covar, _ = _assemnle_data_to_fit(datafile, xcol, ycol, sigma, bounds, scale_covar)
    model, prefix = _prep_lmfit_model(model, kwargs)
    p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kwargs)
    nan_policy = kwargs.pop("nan_policy", getattr(model, "nan_policy", "omit"))

    if single_fit:
        ret_val = __lmfit_one(
            datafile,
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
            nan_policy=nan_policy,
        )
    else:  # chi^2 mode
        pn = p0
        ret_val = np.zeros((pn.shape[0], pn.shape[1] * 2 + 1))
        for i, pn_i in enumerate(pn):  # iterate over every row in the supplied p0 values
            p0, single_fit = _prep_lmfit_p0(
                model, data[1], data[0], pn_i, kwargs
            )  # model, data, params, prefix, columns, scale_covar,**kwargs)
            ret_val[i, :] = __lmfit_one(
                datafile, model, data, p0, prefix, _, scale_covar, nan_policy=nan_policy, output="row"
            )
        if output == "data":  # Create a data object and seet column headers etc correctly
            ret = datafile.clone
            ret.data = ret_val
            ret.column_headers = []
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

    Note:
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
                np.polyval(p[i, :], x=datafile.column(_.xcol)), index=result, replace=replace, header=header
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
            Columns to be used for the x  data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` x column
        ycol (index or None):
            Columns to be used for the  y data for the fitting. If not givem defaults to the
            :py:attr:`Stoner.Core.DataFile.setas` y column

    Keyword Arguments:
        p0 (list, tuple, array or callable):
            A vector of initial parameter values to try. See the notes to :py:meth:`Stoner.Data.curve_fit` for
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
                - "data"    a copy of the :py:class:`Stoner.Core.DataFile` object with the fit recorded in the
                            emtadata and optionally
                    as a column of data.

    Notes:
        The function tries to make use of whatever model you give it. Specifically, it accepts:

            -   A subclass or an instance of :py:class:`scipy.odr.Model` : this is the native model type for the
                underlying scipy odr package.
            -   A subclass or instance of an lmfit_mod.Models.Model: the :py:mod:`Stoner.analysis.fitting.models`
                package has a number of useful prebuilt lmfit models that can be used directly by this function.
            -   A callable function which should have a signature f(x,parameter1,parameter2...) and *not* the
                scip.odr standard f(beta,x)

        This function is designed to be as compatible as possible with :py:meth:`Stoner.Data.curve_fit` and
            :py:meth:`Stoner.Data.lmfit` to facilitate easy of switching between them.

    See Also:
        -   :py:meth:`Stoner.Data.curve_fit`
        -   :py:meth:`Stoner.Data.lmfit`
        -   :py:meth:`Stoner.Data.differential_evolution`
        -   User guide section :ref:`fitting_with_limits`

    Example:
        .. plot:: samples/odr_simple.py
             :include-source:
             :outname: odrfit1
    """
    # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
    # Support both asrow and output, the latter wins if both supplied
    sigma = kwargs.pop("sigma", None)
    sigma_x = kwargs.pop("sigma_x", None)
    absolute_sigma = kwargs.pop("absolute_sigma", sigma is not None)
    scale_covar = kwargs.pop("scale_covar", not absolute_sigma)
    bounds = kwargs.pop("bounds", lambda x, r: True)
    p0 = kwargs.pop("p0", None)
    data, scale_covar, _ = _assemnle_data_to_fit(datafile, xcol, ycol, sigma, bounds, scale_covar, sigma_x=sigma_x)
    if not isinstance(model, odrModel):
        model, prefix = _prep_lmfit_model(model, kwargs)
    else:
        prefix = kwargs.pop("prefix", getattr(model, "name", model.fcn.__name__))
    p0, single_fit = _prep_lmfit_p0(model, data[1], data[0], p0, kwargs)
    kwargs["p0"] = p0
    model = ODR_Model(model, p0=p0)
    if not absolute_sigma:
        data = sp.odr.Data(data[0], data[1], wd=1 / data[3] ** 2, we=1 / data[2] ** 2)
    else:
        data = sp.odr.RealData(data[0], data[1], sx=data[3], sy=data[2])

    if single_fit:
        ret_val = _odr_one(datafile, data, model, prefix, _, **kwargs)
    else:  # chi^2 mode
        raise NotImplementedError("Sorry cannot do chi^2 mode for orthogonal distance regression yet!")
    return ret_val
