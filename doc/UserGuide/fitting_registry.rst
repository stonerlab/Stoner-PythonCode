Fit registered models and select outputs
****************************************

Use :py:attr:`~Stoner.core.data.Data.fit` to fit with a registered model name::

    report = d.fit.linear(x="Field", y="Signal", output="report")

``x`` and ``y`` select columns, consistently with plotting. Omit them to use the
existing column roles and fitting defaults. These are column selectors, not new
raw x/y arrays. The registered ``linear`` shortcut uses lmfit. A complete example:

.. plot:: samples/fit_accessor.py
   :include-source:
   :outname: fit_accessor

Choose the fitting algorithm
============================

Calling the accessor itself defaults to lmfit::

    from Stoner.analysis.fitting.models.generic import Linear

    report = d.fit(Linear, x="Field", y="Signal", output="report")
    report = d.fit.lmfit(Linear, x="Field", y="Signal", output="report")
    parameters, covariance = d.fit.curve_fit(Linear, x="Field", y="Signal")
    report = d.fit.odr(Linear, x="Field", y="Signal", output="report")

``d.fit.differential_evolution`` and ``d.fit.polyfit`` expose their existing
algorithm contracts too. ``method`` remains an algorithm's own option where
supported; it is not an accessor-level algorithm switch. Configured and composite
model instances work with the explicit algorithm methods. Return values follow
the selected algorithm and output choice; fitting does not automatically draw.

The :doc:`/fit-shortcuts` catalogue lists the built-in model shortcuts. Registered
names also appear in ``dir(d.fit)``. Resolving a shortcut constructs no model and
reads no measurement data. Each call uses current owner state and the current
registration, including for a shortcut saved before an explicit replacement.

Migration from direct methods
==============================

Replace ``d.lmfit(Linear, xcol=0, ycol=1)`` with
``d.fit.lmfit(Linear, x=0, y=1)`` or ``d.fit.linear(x=0, y=1)``.
The accessor rejects ``xcol`` and ``ycol``, including mixed old/new keywords.
Positional coordinate order is unchanged. Existing direct fitting methods and
their ``xcol``/``ycol`` keywords remain available during migration.

``d.fit`` now takes precedence over attribute-based lookup of a column named
``fit``. Use ``d.column("fit")`` or explicit column selectors for that column.
Methods returning the owner still support chaining, for example
``d.fit.linear(output="data").plot()``. Other output modes return their existing
reports, arrays or tuples.

Discover and construct models
=============================

Inspect the live, read-only mapping of built-in and user registrations::

    from Stoner.analysis.fitting.models import model_registry

    print(list(d.fit.models))
    Linear = d.fit.models["linear"]
    model = model_registry.create("linear", prefix="background_")
    report = d.fit(model, output="report")

Registry keys use ASCII snake_case while classes retain their existing names.
For example, ``bloch_grueneisen`` maps to ``BlochGrueneisen`` and
``rsj_noiseless`` maps to ``RSJ_Noiseless``. The physics-correct key
``fuchs_sondheimer`` maps to the legacy class ``FluchsSondheimer``; the existing
class/function imports have not been renamed. The spelling is supported by
`Kästle et al., Physical Review B 70, 165414 (2004)
<https://doi.org/10.1103/PhysRevB.70.165414>`_. For Bloch–Grüneisen see
`Vedeneev et al., Physical Review B 51, 16380 (1995)
<https://doi.org/10.1103/PhysRevB.51.16380>`_.

The explicit built-in catalogue includes 28 concrete classes from the generic,
thermal, magnetic, tunnelling, electron-transport and superconductivity modules.
It excludes raw model functions, imported base classes and helper functions.
Use ``model_registry.registration(name)`` for immutable registration provenance.
Looking up a class never instantiates it or guesses parameters. Each ``create``
call invokes its class/factory separately; model state is not cached.

Register a decorated function
=============================

The existing :py:func:`~Stoner.analysis.fitting.models.make_model` decorator now
registers its generated class automatically::

    from Stoner.analysis.fitting.models import make_model, model_registry

    @make_model
    def custom_response(x, slope, intercept):
        return slope * x + intercept

    @custom_response.guesser
    def initial_values(y, x=None):
        return [1.0, 0.0]

    @custom_response.hinter
    def parameter_hints():
        return {"slope": {"min": 0.0}}

    assert model_registry["custom_response"] is custom_response
    report = d.fit.custom_response(output="report")

Registration uses the decorated function's name, not the generated internal
class name. Existing guesser/hinter decorators modify that same registered
class. The generated class's existing fitting metadata prefix is retained.
Use ``@make_model(name="other_response")`` for an explicit registry key.

Register a class or factory
===========================

Use :py:func:`~Stoner.analysis.fitting.models.register_model` for a handwritten
``lmfit.Model`` subclass::

    from Stoner.analysis.fitting.models import register_model
    from Stoner.analysis.fitting.models.generic import Linear

    @register_model
    class BackgroundResponse(Linear):
        pass

    assert model_registry["background_response"] is BackgroundResponse

A factory requires an explicit name and must return a fresh ``lmfit.Model`` on
each call. It is not called during registration; its return type is checked by
``create``. Registered instances and unrelated classes are rejected.

Names must be public ASCII snake_case identifiers. Algorithm names such as
``lmfit`` and ``curve_fit``, and the name ``models``, are reserved for the accessor.
A different object cannot silently replace an existing registration.
Use ``replace=True`` explicitly when redefining a model interactively, for
example ``@make_model(name="custom_response", replace=True)``. Registering the
same class/factory again is harmless. Invalid or duplicate names raise
``ValueError``; unsupported objects raise ``TypeError``. Unknown mapping keys
raise ``KeyError``.

The registry is process-local. Bulk shortcuts through ``folder.each.fit`` send
the selected registration to each worker without installing it in that worker's
global catalogue. Custom decorated classes and factories can therefore be used
when the existing serializer can transfer their definitions. See
:doc:`fitting_folders` for execution controls, results and worker errors.

Factory shortcuts invoke the named factory without constructor arguments. Pass
constructor options explicitly through ``model_registry.create(name, **options)``
and supply that configured model to ``d.fit.lmfit``. This keeps constructor
configuration distinct from fitting options such as ``prefix`` and ``nan_policy``.

Use named fitting outputs
=========================

The :py:class:`~Stoner.enums.FitOutput` string enum supports completion and named
choices while retaining the existing strings::

    from Stoner.enums import FitOutput

    report = d.fit.linear(output=FitOutput.REPORT)
    report = d.fit.linear(output="report")  # Equally supported.

.. list-table:: Common output choices
   :header-rows: 1
   :widths: 20 15 65

   * - Member
     - String
     - Result
   * - ``FIT``
     - ``"fit"``
     - Parameters and covariance.
   * - ``REPORT``
     - ``"report"``
     - Algorithm-specific report object.
   * - ``ROW``
     - ``"row"``
     - Interleaved parameter/error row with goodness of fit.
   * - ``FULL``
     - ``"full"``
     - Algorithm-specific full-result tuple.
   * - ``DATA``
     - ``"data"``
     - The measurement object for a single fit.
   * - ``DICT``
     - ``"dict"``
     - Parameters, errors and fit statistics as a dictionary.

All six choices work with ordinary single-fit calls to ``curve_fit``, ``lmfit``,
``odr`` and ``differential_evolution``. They standardise the selector, not the
report type or detailed tuple contents. Multi-channel ``curve_fit`` retains its
existing collection rules. For repeated-initial-guess ``lmfit`` calls, ``DATA``
returns a results table and other selections retain the existing numerical
results array. ``polyfit`` has no output selector.

Defaults, legacy string dispatch and existing return values remain unchanged.
An enum belonging to another domain raises ``TypeError``. Enum values remain
ordinary string tokens for JSON/configuration interchange; see
:doc:`fitting_options` for the other result-storage controls.
