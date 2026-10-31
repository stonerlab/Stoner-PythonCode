Fit and flatten image backgrounds
*********************************

:py:attr:`~Stoner.Image.core.ImageFile.fit` follows the callable and explicit
algorithm conventions of ``Data.fit``. Its first workflow fits reference patches
that should be flat, then corrects the background across the entire image::

    from Stoner.enums import FitOutput

    image.roi.box(0, 20, 0, 100)
    report = image.fit.plane(output=FitOutput.REPORT)
    background = image.fit.evaluate(report)
    corrected = image.fit.flatten(report)

The plane is ``offset + slope_x*x + slope_y*y``. A full quadratic, including
the cross-term, is available as ``image.fit.quadratic_surface()``:
``offset + slope_x*x + slope_y*y + quadratic_x*x**2 + cross_xy*x*y + quadratic_y*y**2``.
The existing one-dimensional ``quadratic`` registry entry is unchanged.

``image.fit(model, ...)`` defaults to lmfit, just like ``image.fit.lmfit(model, ...)``.
Models can be registered snake_case keys, Model classes/instances or functions
with two independent arguments. There is no image ODR/curve_fit accessor in this
alpha. Lookup, completion and ``models`` discovery do not construct models, read
pixels or create figures. Rendering remains on :doc:`image_plotting`.

An executable background-correction workflow
============================================

Both rows below use disjoint reference patches outside a compact foreground
feature. The model is estimated on those patches and evaluated across the whole
grid, including underneath the foreground feature:

.. plot:: samples/image_fit_accessor.py
   :include-source:
   :outname: image_fit_accessor

Selection, coordinates and uncertainties
========================================

Selection uses :doc:`image_roi`: omitted ``where`` uses the current ROI, explicit
``where=region`` or an exact Boolean/labelled array overrides it, and
``where=ROI.all()`` bypasses current state. True includes pixels. Persistent
exclusions and non-finite intensities never participate in estimation. The
optional ``box`` intersects this selection using the legacy pixel crop parser;
it never invokes a selector. Native physical boxes should be built with ROI.
Empty regions, insufficient samples and rank-deficient geometry fail before
metadata publication. A line of pixels cannot determine an unconstrained plane.

``x`` and ``y`` name existing separable spatial coordinates; intensity is the
dependent variable. ``coordinates="native"`` uses stored coordinates, preserving
crop offsets, decreasing/non-uniform axes and unequal spatial steps. Explicit
``"pixel"`` uses local column/row indices and rejects named coordinates.
Metadata such as MicronsPerPixel does not implicitly override stored calibration.
Two-dimensional coordinate maps are not supported. For a region/frame handle,
the whole image means that handle's domain, without expanding into its parent.

``sigma`` is a positive scalar, exact-shape real array or exactly aligned
DataArray of standard deviations. The residual weights are ``1/sigma``.
Invalid participating uncertainties are errors; values outside the eligible
region do not remove valid observations. Without sigma the fit is unweighted.
No broadcasting, reindexing, clipping or automatic outlier rejection occurs.

The default covariance uses residual scaling, as Data.fit.lmfit does.
``absolute_sigma=True`` disables that scaling, interpreting supplied sigma as
absolute standard deviations. ``scale_covar`` is the inverse control; conflicting
explicit settings are rejected. Native lmfit ``method``, ``fit_kws``, ``max_nfev``
and callback controls are accepted. Use full optimiser names; unknown methods are rejected rather than
silently selecting another optimiser. A successful fit must estimate finite
covariance; methods/configurations that cannot do so fail explicitly. Fixed
coefficients, physical bounds and lmfit parameter expressions are supported
through configured models or ``p0=Parameters``.

Conditioning and native reports
===============================

Unconstrained built-in surfaces fit centred/scaled coordinates, using an exact
residual Jacobian with the default leastsq method. This avoids cancellation in
background prediction when native coordinates have large offsets and small
spacing. Physical coefficients remain in their original basis as native lmfit
parameter expressions. They may be highly correlated; a large intercept far
outside the sampled region is not necessarily a well-determined observable.

The returned REPORT is a genuine lmfit ModelResult. ``params``, ``eval`` and
``fit_report()`` retain lmfit behaviour. Its free ``image_normal_*`` parameters
describe the conditioned basis, so **native report.covar follows report.var_names**.
Do not interpret it as physical coefficient covariance. Physical covariance,
including all transformed correlations, is returned by FIT and is recorded in::

    record = report.image_fit.to_dict()
    names = record["parameter_names"]
    covariance = record["covariance"]

The frozen :py:class:`~Stoner.Image.fit_support.ImageFitContext` stores version-1
provenance without live owners: selection and eligibility descriptions, sigma,
coordinates/units/grid identity, conditioning, coefficients, physical covariance,
sample count/span, free rank/condition, degrees of freedom and fit statistics.
The condition diagnostic refers to the normalised surface design or column-scaled
native Jacobian, as identified by ``condition_basis``. Fixed coefficients have
zero covariance. Physical constraints use a physical-basis fit; an ill-conditioned
constrained system is rejected rather than producing an unreliable estimate.

Outputs and metadata
=====================

Use :py:class:`~Stoner.enums.FitOutput` or its corresponding string:

.. list-table:: Image fitting results
   :header-rows: 1
   :widths: 20 80

   * - Choice
     - Result
   * - FIT (default)
     - Physical coefficients and full physical covariance, in model parameter order.
   * - REPORT
     - Native ModelResult with frozen image_fit context.
   * - ROW
     - Interleaved physical values/errors, followed by reduced chi-square.
   * - FULL
     - Tuple (native report, row).
   * - DICT
     - Named values, d_name errors, chi-square, red. chi-sqr and nfev.
   * - DATA
     - The same ImageFile, with fit metadata; pixels are not corrected.

Successful ordinary fits record ``prefix:image_fit``. ``prefix`` defaults to the
registered key or explicit model class/function name. Failures publish nothing.
Fitting-only calls never alter pixels, exclusions, coordinates or current ROI.
The combined ``flatten(model=...)`` operation also preserves source metadata.

Correction and captured-fit replay
==================================

Correction is ``measured - fitted_background + reference``. The default
:py:class:`~Stoner.enums.BackgroundReference` ``ZERO`` (``"zero"``) removes the
whole model. ``REGION_MEAN`` (``"region_mean"``) retains the captured fitted
background mean on eligible reference pixels, with inverse-variance weighting
when sigma was supplied. A finite number sets an explicit intensity level::

    corrected = image.fit.flatten(report, reference="region_mean")
    at_level_seven = image.fit.flatten(report, reference=7.0)

Both ``evaluate`` and ``flatten`` return independent float64 ImageFiles with
preserved geometry and exclusions. Integer/bool values remain in raw intensity
units; there is no dtype-range normalisation, clipping or truncation. Correction
also changes excluded finite raw values while retaining their exclusion flags.
Original NaNs/infinities remain non-finite. Foreground features outside the ROI
receive the background correction but never influence its estimate. Evaluation
beyond the reference patches is extrapolation; inspect the recorded sampled span.

Accessor replay uses frozen fitted parameters, independently of later source
value edits, ROI changes or registry replacement. Native ``report.eval()`` still
uses the report's current mutable params. Accessor evaluation requires matching
coordinate labels/units and grid geometry, as well as intensity units. It never
silently re-zeroes or resamples another image's coordinates. Built-in reports can be
copied/pickled with their context; custom callables retain lmfit/Python serialisation
limitations. ``flatten(report)`` rejects fitting options;
it does not silently refit. Output records use separate ``prefix:background`` or
``prefix:correction`` keys, preserving unrelated typed metadata.

Legacy metadata entries ``ax``/``fig`` are transient display handles and are
omitted from independent results; fitting retains their source identities without
copying figures. New accessor plot state is likewise not copied into a result.

In-place correction is deferred: ``inplace=True`` raises NotImplementedError.
There is no folder/stack-wide fit dispatcher in this alpha. Individual region
and frame handles can fit and produce independent corrections. Existing
``level_image`` keeps its directional algorithm, options and return behaviour;
it is not replaced by this joint surface solver.

Custom surface models
======================

The existing decorator now declares multiple independent variables::

    from Stoner.analysis.fitting.models import make_model

    @make_model(independent_vars=("x", "y"))
    def tilted_background(x, y, offset, slope_x, slope_y):
        return offset + slope_x*x + slope_y*y

    @tilted_background.guesser
    def guess(data, x=None, y=None):
        return [data.mean(), 0., 0.]

    report = image.fit.tilted_background(output="report")

Both selected coordinate arrays reach the guesser. Bare/named decorators,
hinters and existing Data fitting registrations retain their behaviour. The
shared registry remains read-only except through registration APIs. Optional
``independent_vars`` metadata supports contextual completion without constructing
factories; unknown factory capabilities are validated on call. Image completion
excludes declared 1D models, and Data completion excludes declared 2D models.
Data's explicit multidimensional curve_fit path is unchanged. Existing registrations
named evaluate/flatten are not removed; accessor members take precedence, and the
registered key/model can be supplied explicitly to a fitting call.
