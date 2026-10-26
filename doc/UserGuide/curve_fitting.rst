.. _curve_fit_guide:

Fit your first curve
********************

Curve fitting finds model parameters that describe your measured points.
For example, this fits a straight line ``slope * x + offset`` to five points::

    import numpy as np
    from Stoner import Data

    def line(x, slope, offset):
        return slope * x + offset

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = np.array([1.0, 3.1, 4.9, 7.1, 8.9])
    d = Data(x, signal, column_headers=["X", "Signal"], setas="xy")
    parameters, covariance = d.curve_fit(
        line, p0=[1.0, 0.0], result=True, header="Fit"
    )
    slope, offset = parameters
    print(slope, offset)

The fitted slope is about 1.98 and the offset about 1.04. ``p0`` supplies
starting values in the same order as the parameters in ``line``. ``result=True``
adds the fitted curve as a column called Fit; fit parameters are also stored
in the measurement's metadata. The returned covariance matrix describes the
parameter variances and covariances.

Compare the fit with the measured points::

    d.plot.line(y=["Signal", "Fit"])
    d.plot.show()

For your own experiment, replace the example Data object with a loaded
measurement and choose a function that describes its physics. See
:doc:`data_loading` for loading and :doc:`data_columns` for coordinate and
uncertainty roles. A successful fit is a starting point: inspect the result
and residuals before drawing scientific conclusions.

Choose your next task
=====================

.. toctree::
   :maxdepth: 1

   Polynomial and function fits <fitting_simple>
   Fit outputs, metadata and observation selection <fitting_options>
   Parameter limits and other fitting methods <fitting_methods>
   Choose or create fitting models <fitting_models>
   Fit planes and multidimensional models <fitting_multidimensional>
   Configure fitting workflows with INI files <fitting_configuration>

For an unconstrained polynomial, use ``polyfit``. For a general function with
uncertainty in Y, start with ``curve_fit``. Use ``lmfit`` for parameter limits
or ``odr`` when both X and Y have uncertainties. Differential evolution offers
a population-based alternative when that search strategy suits the problem.

The later guides keep the detailed method options and worked examples separate
from this first recipe. For exact signatures, see the :doc:`module reference <../Stoner>`.
