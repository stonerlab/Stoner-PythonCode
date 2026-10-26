Fitting planes and other multidimensional models
************************************************

.. currentmodule:: Stoner

These examples extend function fitting to several coordinates. Each script
prepares the coordinates, defines the model, performs the fit and plots its
result. Start with :doc:`fitting_simple` if you have not yet fitted a function
of one coordinate.

.. contents:: On this page
   :local:
   :depth: 1

Fit more than one coordinate
============================

:py:meth:`~Stoner.core.data.Data.curve_fit` can also be used to fit more complex problems. In the example below, a set of
points in x,y,z space are fitted to a plane.

.. plot:: samples/curve_fit_plane.py
    :include-source:
    :outname: curvefit_plane

You can also specify the *y-data* to fit to as a NumPy array. This can be used to fit functions that
don't themselves return values that can be matched up to existing data. An example of doing this is fitting a
sphere to a set of :math:`(x,y,z)` data points. In this case the fitting parameters are :math:`(x_0,y_0,z_0)` for the centre of
the sphere, :math:`r` for the radius and the fitting equation is :math:`(x-x_0)^2+(y-y_0)^2+(z-z_0)^2-r^2=0` and so we pass an array
of zeros as the y-data.

.. plot:: samples/sphere_fit.py
    :include-source:
    :outname: curvefit_sphere

See also :ref:`fitting_tricks`

Next steps
==========

See :doc:`fitting_options` for fit outputs and observation selection,
or return to :doc:`curve_fitting` for the learning path.
