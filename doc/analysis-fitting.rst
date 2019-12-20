.. _fitting-models:

-------------------------
Additional Fitting models
-------------------------

The Stoner package contains several pre-build fitting models that are provided as individual
functions for use with :py:meth:`Stoner.Data.curve_fit` and :py:class:`lmfit.Model` classes.Additional
The latter also support the ability to determine an initial value of the parameters from the Data
and so can simplify the fitting code considerably. Many of the models come with an example function.

Generic Modles
^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.generic
    :no-inheritance-diagram:
    :headings: -~

Thermal Physics models
^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.thermal
    :no-inheritance-diagram:
    :headings: -~

Electron Tunnelling models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.tunnelling
    :no-inheritance-diagram:
    :headings: -~

Other Electron Transpoort models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.e_transport
    :no-inheritance-diagram:
    :headings: -~

Magnetism and Magnetic Materials models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.magnetism
    :no-inheritance-diagram:
    :headings: -~

superconductivity models
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis.fitting.models.superconductivity
    :no-inheritance-diagram:
    :headings: -~

