Polynomial and function fits
****************************

.. currentmodule:: Stoner

Use ``polyfit`` when your model is a polynomial, such as a straight line or a
quadratic. Use ``curve_fit`` when you want to write the model as a Python
function. The runnable example in :doc:`curve_fitting` shows a straight-line
function fit from start to finish.

Examples below use a Data object called ``a`` and illustrative column names.
Load your own measurement with :doc:`data_loading` and replace those names with
the ones in your file. ``p0`` supplies starting values for the model parameters.
For fitting methods that append result columns, work on ``a.clone`` if you need
to preserve the original table.

.. contents:: On this page
   :local:
   :depth: 1

Fit a polynomial
================

Simple least squares fitting of polynomial functions is handled by the
:py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.polyfit` method::

   a.fit.polyfit(column_x,column_y,polynomial_order, bounds=lambda x, y:True,result="New Column")

This is a simple pass through to the numpy routine of the same name. The x and y
columns are specified in the first two arguments using the usual index rules for
the Stoner package. The routine will fit multiple columns if *column_y*
is a list or slice. The *polynomial_order* parameter should be a simple integer
greater or equal to 1 to define the degree of polynomial to fit. The *bounds*
function follows the same rules as the *bounds* function in
:py:meth:`~Stoner.core.data.Data.search` to restrict the fitting to a limited range of rows. The
method returns a list of coefficients with the highest power first. If
*column_y* was a list, then a 2D array of coefficients is returned.

If *result* is specified then a new column with the header given by the *result* parameter will be
created and the fitted polynomial evaluated at each point.

Fit a Python function
=====================

For more general curve fitting operations the :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.curve_fit`
method wraps SciPy's curve-fitting routine. For example::

   parameters, covariance = a.fit.curve_fit(
       func, x="X", y="Y", p0=[1.0, 0.0],
       sigma="dY", absolute_sigma=True, result=True, header="Fit",
   )

For ``curve_fit``, use ``where`` to choose observations without changing the
measurement or its persistent mask. True includes a row. For example, fit two
background regions and evaluate the model over the complete measurement::

    import numpy as np

    def background(x, slope, offset):
        return slope * x + offset

    x = a.to_numpy(masked=False)[:, a.find_col("field")]
    region = (x < -5) | (x > 5)
    parameters, covariance = a.fit.curve_fit(
        background, x="field", y="signal", sigma="uncertainty",
        where=region, result=None,
    )
    corrected = a.to_numpy(masked=False)[:, a.find_col("signal")] - background(x, *parameters)

``where`` accepts a full-length Boolean array or a Boolean pandas Series whose
unique index exactly matches the row index in order. Missing Boolean values,
numeric selectors and broadcasting are rejected. If ``bounds`` is also supplied,
the two regions are intersected. ``bounds`` retains its row-selection meaning;
use ``parameter_bounds=(lower, upper)`` for SciPy optimiser limits.

Each fitted y channel independently excludes masked or non-finite required x, y
and uncertainty inputs. Masks on unrelated channels do not remove observations.
External uncertainty arrays must have the full row count; eligible uncertainties
must be positive. Each fit requires more eligible observations than parameters.
If any channel fails, no fit columns or metadata are published.

Stored model columns evaluate the full x axis, including unselected rows, and
exclude invalid or excluded x inputs. Residual columns additionally exclude
invalid or excluded measured y values. The original values and exclusions remain
unchanged. Other fitting methods retain their existing selection interfaces.

The first parameter is the fitting function. This should have prototype
``y=func(x,p[0],p[1],p[2]...)``: where *p* is a list of fitting parameters.

Alternatively a subclass of, or instance of, a :py:class:`lmfit.model.Model` can also be passed and it's function will be used to provide information to
:py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.curve_fit`.

The *p0* parameter supplies initial guesses for the fitting parameters. When
it is omitted, a bare callable uses initial values of one; an lmfit model can
provide parameter hints or a guess from the data.
*x* and *y* are the x and y columns to fit. If *x* and *y* are not given, then the :py:attr:`~Stoner.core.data.Data.setas` attribute is used to determine
which columns to fit.

*sigma*, *absolute_sigma* and *scale_covar* determine how the fitting process takes account of uncertainties in the *y* data.
*sigma*, if present, provides the weightings or standard deviations for each datapoint and so
should also be an array of the same length as the x and y data. If *sigma* is not given and a column is
identified in the :py:attr:`~Stoner.core.data.Data.setas` attribute as containing *e* values, then that is used instead.
If *absolute_sigma* is given then if this is True, the *sigma* values are interpreted as absolute uncertainties in the data points,
if it is False, then they are relative weightings. If *absolute_sigma* is not given, a *scale_covar* parameter will have the same effect,
but inverted, so that True equates to relative weights and False to absolute uncertainties. Finally if neither is given then any *sigma* values
are assumed to be absolute.

If a variance-co-variance matrix is returned from the fit then this can be used to calculate the estimated standard errors in the fitting parameters::

    p_error=np.sqrt(np.diag(p_cov))

The off-diagonal terms of the variance-co-variance matrix give the co-variance between the fitting parameters. Large absolute values of these terms indicate
that the parameters are not properly independent in the model.

Next steps
==========

Use :doc:`fitting_options` for result/report options and
:doc:`fitting_methods` for constraints or uncertainties in both coordinates.
Return to :doc:`curve_fitting` for the learning path.
