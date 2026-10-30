Fit outputs, metadata and observation selection
***********************************************

.. currentmodule:: Stoner

After getting a fit to work, decide what you want to keep: fitted parameters,
a report, a fitted-curve column, residuals, or a Data object for further work.
Start with :doc:`curve_fitting` for a complete first example. This page explains
the common options used by the general fitting methods.

A residual is the measured value minus the model's prediction. Examining
residuals can help you see where a model does not describe the measurements.
The covariance matrix describes parameter variances and covariances; its
diagonal entries are parameter variances rather than standard errors.

.. contents:: On this page
   :local:
   :depth: 1

Return values, stored curves and observation selection
======================================================

The output of the methods used to fit arbitrary functions depends on the keyword parameters *output*, *result*, *replace* and *header* in the method call.

    -   *output="fit"*
        The optimal parameters and a variance-covariance matrix are returned
    -   *output="row"*
        A 1D numpy array of optimal values and standard errors interleaved (i.e. p_opt[0],p_error[0],p_opt[1],p_error[1]....)
        is returned. This is useful when analysing a large number of similar data sets in order to build a table of fitting results.
    -   *output="report"*
        A python object representing the fitting result is returned.
    -   *output="data"*
        The modified data object itself is returned - this is most useful when used in conjunction with :py:class:`~Stoner.folders.mixins.DataFolder`
    -   *output="full"*
        As much information about the fit as can be extracted from the fitting algorithm is returned.

The fit parameters, uncertainties and goodness of fit are stored in metadata.
Use *prefix* to distinguish successive fits; otherwise the model name supplies it.
To store the fitted curve too, set ``result=True`` to append a column, or give
*result* a destination column. At a destination, ``replace=True`` overwrites the
column and ``replace=False`` inserts a column. ``result=None`` leaves the
measurement columns unchanged. *header* names the fitted column; its default is
``"Fitted with {model name}"``.

To store residuals (measured values minus fitted values), also set
``residuals=True``. They are inserted after the fitted column. Alternatively,
give *residuals* a destination column. Residual columns are only stored when a
fitted column is requested.

The default return is ``output="fit"``: fitted parameters and their covariance
matrix. Request ``output="report"`` when you want the fitting library's result
object, for example to call an lmfit result's ``fit_report()`` method. See the
individual method reference for the precise contents of ``output="full"``.

Use :py:class:`~Stoner.enums.FitOutput` for named choices such as
``output=FitOutput.REPORT``; the corresponding strings remain supported.
See :doc:`fitting_registry` for enum choices and model registration.

The plotting accessor method :py:meth:`~Stoner.plot.accessor.DataPlotter.annotate_fit` is useful
for adding formatted fit details to a plot with ``data.plot.annotate_fit(model)``
(as in :doc:`fitting_simple`). It returns the same ``Data`` object for chaining.
Use ``text_only=True`` to obtain formatted text without drawing or creating a figure.

.. plot:: samples/curve_fit_line.py
    :include-source:
    :outname: curve_fit_line

The *bounds* function can be used to restrict the fitting to only a subset of the rows
of data. Any callable object which will take a float and an array of floats, representing the one x-value and one complete row and return
True if the row is to be included in the fit and False if not. e.g::

    def bounds_func(x, row):
        """Keep 100 < x < 200 and 5 < Y < 20."""
        y = row[a.find_col("Y")]
        return 100 < x < 200 and 5 < y < 20

    parameters, covariance = a.fit.curve_fit(func, bounds=bounds_func)

What the fitting wrappers share
===============================

There are a number of advantages to using the Stoner package wrappers around the various fitting algorithms rather than using them as
standalone fitting functions:

    #.  They provide a consistent way of defining the model to be fitted. All of the Stoner package functions accept a model function of the form:
        ``f(x, p1, p2, p3)``, constructing the necessary intermediate model class as required. Similarly, they can all take an :py:class:`lmfit.model.Model`
        class or instance and adapt that as necessary.
    #.  They provide a consistent parameter order and keyword argument names as far as possible within the limits of the underlying algorithms.
        Generally these follow the :py:func:`scipy.optimize.curve_fit` conventions.
    #.  They make use of the :py:attr:`~Stoner.core.data.Data.setas` attribute to identify data columns containing *x*, *y* and associated uncertainties. They
        also provide a common way to select a subset of data to use for fitting through the *bounds* keyword argument.
    #.  They provide a consistent way to add the best-fit data as one or more columns to the :py:class:`~Stoner.core.data.Data` object and to store the best-fit
        parameters in the metadata for later retrieval. The package can therefore use
        :py:meth:`~Stoner.plot.accessor.DataPlotter.annotate_fit` to display the fitting parameters on a plot of the data.

Next steps
==========

See :doc:`fitting_simple` for ordinary fits, :doc:`fitting_methods` for
other algorithms, or :doc:`curve_fitting` for the learning path.
