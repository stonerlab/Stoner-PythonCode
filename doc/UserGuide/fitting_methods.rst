Parameter limits and other fitting methods
******************************************

.. currentmodule:: Stoner

Choose an algorithm to match your measurement and model:

* Use ``lmfit`` for parameters with limits or other parameter settings.
* Use ``odr`` for measurements with uncertainty in both coordinates.
* Consider ``differential_evolution`` when a population-based search is useful.

Start with :doc:`curve_fitting` or :doc:`fitting_simple` if this is your first
fit. The examples below use their own measurements and models; each complete
script shows its required imports and data preparation.

.. contents:: On this page
   :local:
   :depth: 1

.. _fitting_with_limits:

Fit with parameter limits
=========================

The Stoner package has a number of alternative fitting mechanisms that supplement the standard
:py:func:`scipy.optimize.curve_fit` function. New in version 0.2 onwards of the Package is an interface to the
*lmfit* module.


The `lmfit package <https://lmfit.github.io/lmfit-py/>`_ provides a flexible, object-oriented way to fit complex models to experimental data.
A full description is given in the `lmfit documentation <https://lmfit.github.io/lmfit-py/>`_. The
:py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` method is used to interact with lmfit.

In order to use :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit`, one requires a :py:class:`lmfit.model.Model` instance. This describes a function
and its independent and fittable parameters, whether they have limits and what the limits are. The :py:mod:`Stoner.analysis.fitting` package contains
a series of :py:class:`lmfit.model.Model` subclasses that represent various models used in condensed matter physics.

The operation of :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` is very similar to that of :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.curve_fit`::

    from Stoner.analysis.fitting.models.thermal import Arrhenius
    model=Arrhenius()
    fit=a.fit.lmfit(model,x="Temp",y="Cond",p0=[1E7, 0.01],
                result=True,header="Fit",output="report")
    print(fit.fit_report())
    print(fit.params["A"].value, fit.params["A"].stderr)

In this example we fit an Arrhenius model to data contained in the 'Temp' and 'Cond' columns. The resulting
fit would be added as an additional column called ``Fit``. In addition, details of the fit are added as metadata to the current :py:class:`~Stoner.core.data.Data`.

The *model* argument to :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` can be either an instance of the model class, or just the class itself (in which case it will be
instantiated as required), or just a bare callable, in which case a model class will be created around it. The latter is approximately equivalent to
a simple call to :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.curve_fit`.

The return value from :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` is controlled by the *output* keyword parameter. With ``output="report"`` it is an :py:class:`lmfit.model.ModelResult`
instance; the default ``output="fit"`` returns parameters and covariance. This contains all the information about the fit and fitting process.

You can pass the model as a subclass of model, if you don't pass initial values either via the *p0* parameter or as keyword arguments, then the model's
*guess* method is called (e.g. :py:meth:`Stoner.analysis.fitting.models.thermal.Arrhenius.guess`) to determine parameters from the data. For example:

.. plot:: samples/lmfit_example.py
    :include-source:
    :outname: lmfit_example

Orthogonal distance regression
==============================

:py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.curve_fit` and :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` are least-squares fitting interfaces. The selected optimiser depends on the options and constraints. The essential point is that it seeks to minimise the **vertical** distance between the model and the data points, taking into account the uncertainty
in the vertical position (i.e. *y* coordinate) only. If your data has peaks that may change position and/or uncertainties in the horizontal (*x*) position of
the data points, you may be better off using an orthogonal distance regression.

The :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.odr` method wraps the :py:mod:`scipy.odr` module and follows the ``lmfit`` interface where possible. In fact, in most
cases it can be used as a drop-in replacement:

.. plot:: samples/odr_simple.py
     :include-source:
     :outname: odrfit2

The :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.odr` method allows uncertainties in *x* and *y* to be specified via the *sigma_x* and *sigma_y*  parameters. If either are not specified, and
a *sigma* parameter is given, then that is used instead. If they are either explicitly set to **None** or not given, then the :py:attr:`~Stoner.core.data.Data.setas` attribute is
used instead.

Differential evolution
======================

Differential evolution evaluates a population of possible parameter sets and
combines promising candidates to generate new ones. Random variation helps the
search explore beyond local minima. This can be useful when there are many
parameters or a cost function that is not smoothly differentiable. Stoner uses
a weighted-variance cost, as in its ordinary function-fitting methods.

When the number of parameters gets large it can get increasingly difficult to get fits using the techniques above. In these situations, the differential evolution
approach may be valuable. The :py:meth:`Stoner.analysis.fitting.accessor.DataFitter.differential_evolution` method provides a wrapper around the :py:func:`scipy.optimize.differential_evolution`
minimiser with the advantage that the model specification, and calling signatures are essentially the same as for the other fitting functions and thus there is
little programmer overhead to switching to it:

.. plot:: samples/differential_evolution_simple.py
    :include-source:
    :outname: diffev2

Intrinsically, the differential evolution algorithm does not calculate a variance-covariance matrix since it never needs to find the gradient of the :math:`\chi^2`
of the fit with the parameters. In order to provide such an estimate, the :py:meth:`Stoner.analysis.fitting.accessor.DataFitter.differential_evolution` method carries out a standard least-squares
non-linear fit (using :py:func:`scipy.optimize.curve_fit`) as a second stage once :py:func:`scipy.optimize.differential_evolution` has found a likely good fitting
set of parameters. This hybrid approach allows a good fit location to be identified, but also the physically useful fitting errors to be estimated.

Next steps
==========

See :doc:`fitting_models` to choose or build a model,
:doc:`fitting_options` for outputs, or :doc:`curve_fitting` for the learning path.
