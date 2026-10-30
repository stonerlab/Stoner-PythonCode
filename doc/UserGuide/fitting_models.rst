Choosing and creating fitting models
************************************

.. currentmodule:: Stoner

A model is the mathematical function you expect the measurement to follow.
You can pass a Python function directly, use one of Stoner's physical models,
or create a model class with parameter guesses and limits.

Start with the included-model list when a standard physical model describes
your experiment. The final section shows how to turn your own function into
a reusable model. See :doc:`fitting_simple` for using a plain function first.

.. contents:: On this page
   :local:
   :depth: 1

Included models
===============

The :py:mod:`Stoner.analysis.fitting.models` module provides a number of standard fitting models suitable for solid-state physics. Each model is provided as either an
``lmfit.Model`` or a callable function. The former often also provides a means to guess initial values from the data.

Elementary Models
-----------------

Amongst the included models are very generic model functions (in :py:mod:`Stoner.analysis.fitting.models.generic`) including:
.. currentmodule:: Stoner.analysis.fitting.models.generic

    -   :py:class:`Stoner.analysis.fitting.models.generic.Linear` - straight-line fit :math:`y=mx+c`
    -   :py:class:`Stoner.analysis.fitting.models.generic.Quadratic` - second-order polynomial :math:`y=ax^2+bx+c`
    -   :py:class:`Stoner.analysis.fitting.models.generic.PowerLaw` - a general power-law expression :math:`y=Ax^k`
    -   :py:class:`Stoner.analysis.fitting.models.generic.StretchedExp` is a standard exponential decay with an additional power :math:`\beta` -
        :math:`y=A\exp\left[\left(\frac{-x}{x_0}\right)^\beta\right]`

Thermal Physics Models
----------------------

The :py:mod:`Stoner.analysis.fitting.models.thermal` module supports a range of models suitable for various thermal physics related expressions:
.. currentmodule:: Stoner.analysis.fitting.models.thermal

    -   :py:class:`Stoner.analysis.fitting.models.thermal.Arrhenius` - The Arrhenius expression is used to describe processes whose rate is controlled by a thermal distribution,
        it is essentially an exponential decay :math:`y=A\exp\left(\frac{-\Delta E}{k_Bx}\right)`.
    -   :py:class:`Stoner.analysis.fitting.models.thermal.ModArrhenius` - The modified Arrhenius expression :math:`\tau=Ax^n\exp\left(\frac{-\Delta E}{k_B x}\right)` is used when the
        prefactor in the regular Arrhenius theory has a temperature dependence. Typically the prefactor exponent ranges for :math:`-1<n<1`.
    -   :py:class:`Stoner.analysis.fitting.models.thermal.NDimArrhenius` is suitable for thermally activated transport in more than one dimension, or in variable-range hopping processes -
        :math:`\tau=A\exp\left(\frac{-\Delta E}{k_B x^n}\right)`
    -   :py:class:`Stoner.analysis.fitting.models.thermal.VFTEquation` - the Vogel-Fulcher-Tammann equation is another modification of the Arrhenius equation which can apply when a process
        has both an activation energy and a threshold temperature (due to a phase transition, for example) and is given by
        :math:`\tau = A\exp\left(-\frac{\Delta E}{k_B(x-x_0)}\right)`

Tunnelling Electron Transport Models
------------------------------------

We also have a number of models for electron tunnelling processes built into the library in :py:mod:`Stoner.analysis.fitting.models.tunnelling`:
.. currentmodule:: Stoner.analysis.fitting.models.tunnelling

    -   :py:class:`Stoner.analysis.fitting.models.tunnelling.Simmons` - the Simmons model describes tunnelling through a square barrier potential for the limit where the barrier height
        is comparable to the junction bias.
    -   :py:class:`Stoner.analysis.fitting.models.tunnelling.BDR` - this model introduces a trapezoidal barrier where the barrier height is different between the two electrodes - e.g. where the
        electrodes are composed of different materials.
    -   :py:class:`Stoner.analysis.fitting.models.tunnelling.FowlerNordheim` - this is another simplified model of electron tunnelling that has single barrier-height and width parameters.
    -   :py:class:`Stoner.analysis.fitting.models.tunnelling.TersoffHammann` - this model just treats tunnelling as a linear I-V process and is applicable when the barrier height is large compared
        to the bias across the tunnel barrier.

Magnetism Related Models
------------------------

The :py:mod:`Stoner.analysis.fitting.models.magnetism` includes models related to magnetism and magnetic materials.

.. currentmodule:: Stoner.analysis.fitting.models.magnetism

-   :py:class:`Stoner.analysis.fitting.models.magnetism.Langevin` models the magnetic moment versus field of a paramagnet.
-   :py:class:`Stoner.analysis.fitting.models.magnetism.KittelEquation` and :py:class:`Stoner.analysis.fitting.models.magnetism.Inverse_Kittel` describe the magnetic field and frequency
    response of the ferromagnetic resonance peak.

Peak Models
-----------

The `lmfit <https://lmfit.github.io/lmfit-py/>`_ package comes with several common peak-function models which can be used directly. The Stoner package adds a couple more to the
selection - these are particularly useful for fitting ferromagnetic resonance data:

    -   :py:class:`Stoner.analysis.fitting.models.generic.Lorentzian_diff` - lmfit includes built-in classes for Lorentzian peaks, but this model is the differential
        of a Lorentzian peak.
    -   :py:class:`Stoner.analysis.fitting.models.magnetism.FMR_Power` - although this model is usually used specifically for calculating the absorption spectrum for a ferromagnetic resonance
        process, it is in fact a generic combination of both Lorentzian peak and differential forms.

Superconductivity Related Models
--------------------------------

The :py:mod:`Stoner.analysis.fitting.models.superconductivity` offers some models related to superconducting materials and devices.
.. currentmodule:: Stoner.analysis.fitting.models.superconductivity


    -   :py:class:`Stoner.analysis.fitting.models.superconductivity.Strijkers` - this tunnel model describes electrons passing from a superconductor to a non-superconductor with a potential step or
        barrier between the two. In one limit it describes Andreev reflection in a clean interface and in the other, a Giaever Junction (S-I-N).



Other Electrical Transport Models
---------------------------------

Finally, we include some other common electrical-transport models for solid-state physics in the :py:mod:`Stoner.analysis.fitting.models.e_transport` module.
.. currentmodule:: Stoner.analysis.fitting.models.e_transport

    -   :py:class:`Stoner.analysis.fitting.models.e_transport.BlochGrueneisen` - this model describes electrical resistivity of a metal as a function of temperature and can be used to
        extract the Debye temperature :math:`\Theta_D`.
    -   :py:class:`Stoner.analysis.fitting.models.e_transport.FluchsSondheimer` - this model describes the electrical resistivity of a thin film as a function of its thickness and can
        be used to extract a mean free path :math:`\lambda_{mfp}` and intrinsic conductivity.
    -   :py:class:`Stoner.analysis.fitting.models.e_transport.WLfit` - the weak-localisation fit can be used to describe the magnetoconductance of a system with sufficient scattering that
        weak-localisation effects appear,

Create a reusable model
=======================

You can simply pass a bare function to any of the general fitting methods, but there can be advantages in using an ``lmfit.Model`` class - such as the
ability to combine several models together, the ability to guess parameters or just for the readability. The Stoner package provides some tools to help make
suitable model classes.

:py:func:`Stoner.analysis.fitting.models.make_model` is a Python decorator function that can do the leg-work of transforming a simple model function of the form::

    from Stoner.analysis.fitting.models import make_model

    @make_model
    def model_func(x_data,param1,param2):
        return param1 * x_data + param2

into a suitable model class of the same name. This model class then provides a decorator function itself to mark a second function as something that can be used
to guess parameter values::

    import numpy as np

    @model_func.guesser
    def model_guess(y_data, x=None):
        return [1.0, float(np.mean(y_data))]

In the same vein, the class provides a decorator to use a function to generate hints about the parameter, such as bounding values::

    @model_func.hinter
    def model_parameter_hints():
        return {"param1":{"max":10.0, "min":1.0},"param2":{"max":0.0}}

The newly created ``model_func`` class is registered automatically. Fit through
``d.fit.model_func(...)`` or pass the class explicitly to
``d.fit.lmfit(model_func, ...)``. The guesser and hinter remain attached to the
same registered class. See :doc:`fitting_registry` for explicit names, collision
handling and read-only model discovery. The following runnable example uses the
shortcut created by ``@make_model``.

.. plot:: samples/make_model.py
    :include-source:
    :outname: make_model

Next steps
==========

See :doc:`fitting_methods` to fit with parameter limits, or
:doc:`fitting_configuration` to configure repeated fits from a file.
Return to :doc:`curve_fitting` for the learning path.
