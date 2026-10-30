Detailed fitting algorithm options
**********************************

Use :py:attr:`~Stoner.core.data.Data.fit` for ordinary fitting calls. Its
:py:class:`~Stoner.analysis.fitting.accessor.DataFitter` methods delegate to
the numerical functions below, preserving their algorithm-specific options,
result storage and return values.

On the accessor, supply ``x`` and ``y`` as coordinate selectors. These are
translated to the ``xcol`` and ``ycol`` arguments documented below. The explicit
``datafile`` argument is supplied by the accessor's owner. For example,
``data.fit.curve_fit(model, x=0, y=1)`` delegates to
``curve_fit(data, model, xcol=0, ycol=1)``. Direct function calls retain their
original argument names.

.. automodule:: Stoner.analysis.fitting.functions
   :members: lmfit, curve_fit, odr, differential_evolution, polyfit
