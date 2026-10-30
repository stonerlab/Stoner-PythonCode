Registered fitting shortcuts
*****************************

.. currentmodule:: Stoner.analysis.fitting.accessor

Each method below uses lmfit with its registered model already selected.
``x`` and ``y`` select measurement columns. Omitted coordinates follow the
existing role defaults. ``p0``, ``sigma`` and other options have the same
meaning as :py:meth:`DataFitter.lmfit`; return values follow its output option.

Model classes are instantiated afresh through the existing lmfit path.
Custom registered names appear in ``dir(data.fit)`` and use the same signature.
See :doc:`UserGuide/fitting_registry` for registration, factories and migration.

The entries document the built-in catalogue. Their signatures describe the
dynamic methods on an accessor instance; these are not static class attributes.

.. py:method:: DataFitter.linear(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.generic.Linear` using lmfit.

.. py:method:: DataFitter.quadratic(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.generic.Quadratic` using lmfit.

.. py:method:: DataFitter.power_law(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.generic.PowerLaw` using lmfit.

.. py:method:: DataFitter.stretched_exp(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.generic.StretchedExp` using lmfit.

.. py:method:: DataFitter.lorentzian_diff(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.generic.Lorentzian_diff` using lmfit.

.. py:method:: DataFitter.arrhenius(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.thermal.Arrhenius` using lmfit.

.. py:method:: DataFitter.n_dim_arrhenius(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.thermal.NDimArrhenius` using lmfit.

.. py:method:: DataFitter.mod_arrhenius(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.thermal.ModArrhenius` using lmfit.

.. py:method:: DataFitter.vft_equation(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.thermal.VFTEquation` using lmfit.

.. py:method:: DataFitter.bloch_law(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.BlochLaw` using lmfit.

.. py:method:: DataFitter.bloch_law_thin(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.BlochLawThin` using lmfit.

.. py:method:: DataFitter.langevin(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.Langevin` using lmfit.

.. py:method:: DataFitter.kittel_equation(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.KittelEquation` using lmfit.

.. py:method:: DataFitter.inverse_kittel(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.Inverse_Kittel` using lmfit.

.. py:method:: DataFitter.fmr_power(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.magnetism.FMR_Power` using lmfit.

.. py:method:: DataFitter.simmons(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.tunnelling.Simmons` using lmfit.

.. py:method:: DataFitter.bdr(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.tunnelling.BDR` using lmfit.

.. py:method:: DataFitter.fowler_nordheim(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.tunnelling.FowlerNordheim` using lmfit.

.. py:method:: DataFitter.tersoff_hammann(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.tunnelling.TersoffHammann` using lmfit.

.. py:method:: DataFitter.wl_fit(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.e_transport.WLfit` using lmfit.

.. py:method:: DataFitter.fuchs_sondheimer(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.e_transport.FluchsSondheimer` using lmfit.

.. py:method:: DataFitter.bloch_grueneisen(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.e_transport.BlochGrueneisen` using lmfit.

.. py:method:: DataFitter.strijkers(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.Strijkers` using lmfit.

.. py:method:: DataFitter.woods_diffusive(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.Woods_Diffusive` using lmfit.

.. py:method:: DataFitter.woods_ballistic(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.Woods_Ballistic` using lmfit.

.. py:method:: DataFitter.rsj_noiseless(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.RSJ_Noiseless` using lmfit.

.. py:method:: DataFitter.rsj_simple(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.RSJ_Simple` using lmfit.

.. py:method:: DataFitter.ic_b_airy(x=None, y=None, p0=None, sigma=None, **kwargs)

   Fit with :py:class:`~Stoner.analysis.fitting.models.superconductivity.Ic_B_Airy` using lmfit.

Additional registered model references
======================================

These existing classes are registered alongside the module-exported models.

.. currentmodule:: Stoner.analysis.fitting.models.magnetism

.. py:class:: BlochLawThin(*args, **kwargs)

   Existing lmfit model class registered as ``bloch_law_thin``. Constructor
   arguments follow the class's existing lmfit configuration interface.

.. currentmodule:: Stoner.analysis.fitting.models.superconductivity

.. py:class:: Woods_Diffusive(*args, **kwargs)

   Existing diffusive-interface lmfit model registered as ``woods_diffusive``.
   Constructor arguments follow the class's existing lmfit configuration interface.

.. py:class:: Woods_Ballistic(*args, **kwargs)

   Existing ballistic-interface lmfit model registered as ``woods_ballistic``.
   Constructor arguments follow the class's existing lmfit configuration interface.
