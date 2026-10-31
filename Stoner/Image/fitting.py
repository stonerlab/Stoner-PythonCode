"""Fit selected image backgrounds and apply captured surfaces over the whole grid."""

import json
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum

import numpy as np
from lmfit import Parameters
from lmfit.model import ModelResult

from ..analysis.fitting.models import model_registry
from ..analysis.fitting.models.surfaces import _SurfaceModel
from ..core.roi import ROI, geometry_signature
from ..core.storage import _copy_metadata
from ..enums import (
    BackgroundReference,
    CoordinateMode,
    FitOutput,
    _normalise_fit_output,
)
from .coordinates import resolve_grid
from .fit_support import (
    ImageFitContext,
    construct_model,
    parameters,
    physical_covariance,
    prepare,
)
from .roi import copy_roi
from .storage_bridge import owner_of

__all__ = ["ImageFitContext", "ImageFitter"]


class _ImageModels(Mapping):
    """Expose known two-variable and undeclared models without constructing factories."""

    def __iter__(self):
        return (key for key in model_registry if model_registry.registration(key).independent_vars is None
                or len(model_registry.registration(key).independent_vars) == 2)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, key):
        variables = model_registry.registration(key).independent_vars
        if variables is not None and len(variables) != 2:
            raise KeyError(key)
        return model_registry[key]


class _ImageModelFit:
    """Resolve a registered model afresh when an image shortcut is invoked."""

    def __init__(self, fitter, name):
        self._fitter, self._name = fitter, name

    @property
    def __name__(self):
        return self._name

    @property
    def __doc__(self):
        return f"Fit registered image model {self._name}; arguments follow ImageFitter.lmfit."

    def __call__(self, x=None, y=None, p0=None, sigma=None, **kwargs):
        return self._fitter.lmfit(self._name, x, y, p0, sigma, **kwargs)


def _metadata(image):
    """Copy scientific metadata without cloning legacy live plotting handles."""
    metadata = owner_of(image)._state.metadata
    return _copy_metadata(metadata, {id(metadata[key]): metadata[key] for key in ("ax", "fig") if key in metadata})


class ImageFitter:
    """Fit selected intensities with lmfit, following Data.fit's accessor conventions.

    Args:
        image (ImageFile):
            Live owner; lookup does not read pixels or create figures/models.

    Notes:
        Calling defaults to lmfit. Registered plane and quadratic_surface shortcuts
        use the same preparation/output path. x/y select spatial coordinates;
        intensity is dependent. Current ROI is the default estimation region.
        Fitting never corrects pixels; evaluate/flatten replay a captured report.
    """

    __slots__ = ("_image",)

    def __init__(self, image):
        self._image = image

    @property
    def models(self):
        """Return a live read-only catalogue of spatial or dimension-undeclared models."""
        return _ImageModels()

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.models))

    def __getattr__(self, name):
        if not name.startswith("_") and name in self.models:
            return _ImageModelFit(self, name)
        raise AttributeError(f"No image fitting method or compatible registered model named {name!r}.")

    def __call__(self, model, x=None, y=None, p0=None, sigma=None, **kwargs):
        """Fit with lmfit; accept the same arguments and outputs as :meth:`lmfit`."""
        return self.lmfit(model, x, y, p0, sigma, **kwargs)

    def lmfit(self, model, x=None, y=None, p0=None, sigma=None, *, where=None, box=None,
              coordinates=CoordinateMode.NATIVE, output=FitOutput.FIT, prefix=None, **fit_options):
        """Estimate a two-variable model on eligible reference pixels.

        Args:
            model (str, lmfit.Model, type or callable):
                Registered key, configured model, Model class or two-variable function.
            x, y (str or None):
                Existing separable spatial coordinates; None selects dimension axes.
            p0 (sequence, dict, lmfit.Parameters, callable or None):
                Physical initial coefficients/constraints. None guesses automatically.
            sigma (float, array, xarray.DataArray or None):
                Standard deviations; participating values must be finite and positive.

        Keyword Arguments:
            where (ROI, Boolean array, xarray.DataArray or None):
                None uses current ROI; explicit selection overrides it. ROI.all()
                bypasses current state while retaining exclusions/validity checks.
            box (crop specification or None):
                Optional legacy pixel box intersected with selection; never interactive.
            coordinates (CoordinateMode or str):
                Native by default, or explicit local pixel coordinates.
            output (FitOutput or str):
                FIT returns physical coefficients/covariance; REPORT is native
                ModelResult with immutable image_fit context; FULL is (report, row).
                ROW interleaves values/errors and ends in reduced chi-square;
                DICT names values/errors/statistics; DATA returns this image.
            prefix (str or None):
                Metadata namespace; default model key/class name, followed by a colon.
            **fit_options (dict):
                Native method, fit_kws, max_nfev, iter_cb, verbose, calc_covar and
                coerce_farray options. absolute_sigma=True disables residual covariance
                scaling; scale_covar is its inverse. Conflicting settings fail.

        Returns:
            object:
                Selected fitting output. Successful calls record prefix:image_fit;
                failures publish nothing. Pixels/exclusions/ROI are unchanged.

        Notes:
            Unconstrained built-in surfaces use conditioned free parameters. Native
            report.covar follows report.var_names; physical covariance is in FIT
            and report.image_fit.to_dict()['covariance']. Fixed/bounded/expression
            constraints use physical-basis fitting with identifiability checks.
        """
        output = FitOutput(_normalise_fit_output(output))
        report, namespace = self._fit(model, x, y, p0, sigma, where, box, coordinates, prefix, fit_options)
        record = report.image_fit.to_dict()
        covariance = np.asarray(record["covariance"])
        values = np.asarray(record["coefficients"])
        errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        row = np.r_[np.column_stack((values, errors)).ravel(), report.redchi]
        result = {}
        for name, value, error in zip(record["parameter_names"], values, errors):
            result[name], result[f"d_{name}"] = value, error
        result.update({"chi-square": report.chisqr, "red. chi-sqr": report.redchi, "nfev": report.nfev})
        current = owner_of(self._image)._state.dataset.intensity
        if (geometry_signature(current) != record["grid"]
                or current.attrs.get("units") != record["intensity_units"]):
            raise ValueError("Image geometry/units changed during fitting; metadata was not published.")
        metadata = _metadata(self._image)
        metadata[f"{namespace}image_fit"] = record
        owner_of(self._image)._replace_metadata(metadata)
        return {FitOutput.FIT: (values, covariance), FitOutput.REPORT: report, FitOutput.FULL: (report, row),
                FitOutput.ROW: row, FitOutput.DICT: result, FitOutput.DATA: self._image}[output]

    def _fit(self, model, x, y, p0, sigma, where, box, coordinates, prefix, options):
        options = dict(options)
        allowed = {"method", "iter_cb", "verbose", "fit_kws", "max_nfev", "calc_covar", "coerce_farray",
                   "absolute_sigma", "scale_covar", "nan_policy"}
        unknown = set(options)-allowed
        if unknown:
            raise TypeError(f"Unsupported image fitting options: {', '.join(sorted(unknown))}")
        if options.pop("nan_policy", "raise") != "raise":
            raise ValueError("Image fitting uses explicit eligibility; nan_policy must be 'raise'.")
        both_scaling = "absolute_sigma" in options and "scale_covar" in options
        absolute = options.pop("absolute_sigma", False)
        scale = options.pop("scale_covar", not absolute)
        if not isinstance(absolute, (bool, np.bool_)) or not isinstance(scale, (bool, np.bool_)):
            raise TypeError("absolute_sigma and scale_covar must be Boolean.")
        if both_scaling and scale == absolute:
            raise ValueError("absolute_sigma and scale_covar specify conflicting covariance policies.")
        from lmfit.minimizer import SCALAR_METHODS

        method = options.get("method", "leastsq")
        if not isinstance(method, str) or isinstance(method, Enum):
            raise TypeError("method must be a full lmfit optimiser name.")
        method = method.lower()
        methods = {"leastsq", "least_squares", "brute", "basinhopping", "ampgo", "emcee", "shgo", "dual_annealing"}
        methods.update(SCALAR_METHODS)
        methods.update(value.lower() for value in SCALAR_METHODS.values())
        if method not in methods:
            raise ValueError(f"Unknown lmfit optimiser {method!r}; use a full supported method name.")
        options["method"] = method
        model, key = construct_model(model)
        if prefix is not None and (not isinstance(prefix, str) or not prefix):
            raise ValueError("prefix must be a non-empty string or None.")
        namespace = (key if prefix is None else prefix).rstrip(":") + ":"
        dataset, grid, selected, eligible, xv, yv, data, weights = prepare(
            self._image, x, y, sigma, where, box, coordinates)
        independent = dict(zip(model.independent_vars, (xv, yv)))
        params, normal, transform, rank, condition = parameters(model, data, independent, p0, weights)
        if normal is not None and options.get("method", "leastsq") == "leastsq":
            kws = dict(options.get("fit_kws") or {})
            if not kws.get("col_deriv", False):
                kws.setdefault("Dfun", model._jacobian)
            options["fit_kws"] = kws
        report = model.fit(data, params=params, weights=weights, scale_covar=scale, nan_policy="raise",
                           **independent, **options)
        if not report.success or np.iscomplexobj(report.best_fit) or not np.isfinite(report.best_fit).all():
            raise RuntimeError(f"Image fitting failed: {report.message}")
        covariance = physical_covariance(report, transform)
        weighted = (weights/np.max(weights))**2
        from .. import __version__

        record = {
            "version": 1, "stoner_version": __version__, "model": key, "model_class": type(model).__qualname__,
            "parameter_names": list(model.param_names), "coefficients": [report.params[p].value for p in model.param_names],
            "covariance": covariance.tolist(), "coordinate_mode": str(CoordinateMode(coordinates)),
            "x": x, "y": y, "coordinate_labels": [grid.xlabel, grid.ylabel],
            "grid": geometry_signature(dataset.intensity), "normalization": normal,
            "selection": ROI.from_mask(selected, reference=dataset.intensity).to_dict(),
            "eligibility": ROI.from_mask(eligible, reference=dataset.intensity).to_dict(),
            "sigma": None if sigma is None else (1/weights).tolist(),
            "sample_count": int(eligible.sum()), "rank": rank, "condition": condition,
            "condition_basis": "normalised_surface" if normal is not None else "column_scaled_jacobian",
            "degrees_of_freedom": int(report.nfree), "absolute_sigma": not scale,
            "reference_mean": float(np.average(report.best_fit, weights=weighted)),
            "sample_span": [[float(xv.min()), float(xv.max())], [float(yv.min()), float(yv.max())]],
            "source": str(self._image.filename), "source_dtype": str(dataset.intensity.dtype),
            "intensity_units": dataset.intensity.attrs.get("units"),
            "coordinate_names": [x or "x", y or "y"],
            "coordinate_units": ["pixel", "pixel"] if CoordinateMode(coordinates) is CoordinateMode.PIXEL else [
                dataset.coords[name].attrs.get("units") for name in (x or "x", y or "y")],
            "success": bool(report.success), "method": str(report.method),
            "chi_square": float(report.chisqr), "reduced_chi_square": float(report.redchi),
            "nfev": int(report.nfev),
            "metadata_prefix": namespace,
        }
        if isinstance(model, _SurfaceModel):
            intensity = record["intensity_units"]
            ux, uy = record["coordinate_units"]
            units = [intensity, f"{intensity} / {ux}" if intensity and ux else None,
                     f"{intensity} / {uy}" if intensity and uy else None]
            if model._quadratic:
                units.extend((f"{intensity} / {ux}^2" if intensity and ux else None,
                              f"{intensity} / ({ux} * {uy})" if intensity and ux and uy else None,
                              f"{intensity} / {uy}^2" if intensity and uy else None))
            record["parameter_units"] = units
        report.image_fit = ImageFitContext(json.dumps(record, allow_nan=False), report.params.dumps())
        return report, namespace

    def _evaluate(self, report):
        if not isinstance(report, ModelResult) or not isinstance(getattr(report, "image_fit", None), ImageFitContext):
            raise TypeError("Use a REPORT from ImageFile.fit for evaluation/correction.")
        record = report.image_fit.to_dict()
        owner = owner_of(self._image)
        if owner is None or geometry_signature(owner._state.dataset.intensity) != record["grid"]:
            raise ValueError("Report coordinate grid does not match this image handle.")
        if owner._state.dataset.intensity.attrs.get("units") != record.get("intensity_units"):
            raise ValueError("Report intensity units do not match this image.")
        owner._check_writable()
        grid = resolve_grid(self._image, record["x"], record["y"], record["coordinate_mode"])
        xx, yy = np.meshgrid(grid.x, grid.y)
        params = Parameters().loads(report.image_fit.parameters)
        values = report.model.eval(params=params, **dict(zip(report.model.independent_vars, (xx, yy))))
        if np.iscomplexobj(values):
            raise ValueError("A background model must evaluate to real intensities.")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != owner.shape or not np.isfinite(values).all():
            raise ValueError("Fitted background is not finite across the whole image grid.")
        return values, record

    def _result(self, values, record, operation, reference=0.):
        from .storage import ImageStorage

        original = owner_of(self._image)._state
        metadata = _metadata(self._image)
        for name in ("ax", "fig"):
            metadata.pop(name, None)
        package = ImageStorage("image", original.dataset.copy(deep=True), metadata,
                               original.fill_value, version=original.version)
        package.dataset["intensity"].data = values.copy()
        package.fill_value = np.ma.default_fill_value(np.array(0., dtype=np.float64))
        package.metadata[f"{record['metadata_prefix']}{operation}"] = {
            "version": 1, "fit": deepcopy(record), "reference": reference, "operation": operation}
        result = type(self._image).from_storage(package)
        copy_roi(self._image, result)
        return result

    def evaluate(self, report):
        """Evaluate a captured report over the whole matching grid.

        Args:
            report (lmfit.model.ModelResult):
                Image fitting REPORT; current ROI and source value edits are ignored.

        Returns:
            ImageFile:
                Independent float64 background with source calibration/exclusions.
        """
        values, record = self._evaluate(report)
        return self._result(values, record, "background")

    def flatten(self, model="plane", x=None, y=None, p0=None, sigma=None, *, where=None, box=None,
                coordinates=CoordinateMode.NATIVE, reference=BackgroundReference.ZERO, inplace=False,
                prefix=None, **fit_options):
        """Subtract a fitted background over the whole image, retaining a chosen constant.

        Args:
            model (str, Model, type, callable or ModelResult):
                Model to fit, default plane, or captured REPORT to replay without refitting.
            x, y (str or None):
                Existing coordinate names, as for lmfit; omitted during report replay.
            p0 (sequence, dict, Parameters, callable or None):
                Initial physical parameters/constraints; omitted during report replay.
            sigma (float, array, DataArray or None):
                Standard deviations, as for lmfit; omitted during report replay.

        Keyword Arguments:
            where (ROI, Boolean array, DataArray or None):
                Estimation selection as for lmfit. Current ROI limits fitting, not correction.
            box (crop specification or None):
                Optional pixel box intersected with the fitting selection.
            coordinates (CoordinateMode or str):
                Native by default; pixel mode selects local indices for estimation.
            prefix (str or None):
                Metadata namespace for a new fit; omitted during report replay.
            reference (BackgroundReference, str or float):
                Zero removes the whole background; region_mean retains its captured
                inverse-variance weighted fitted mean; a finite number sets intensity.
            inplace (bool):
                False returns an independent image. True is deferred and rejected.
            **fit_options (dict):
                lmfit controls; output is not accepted because flatten returns ImageFile.

        Returns:
            ImageFile:
                Independent float64 measured-background+reference, without clipping,
                normalisation or source mutation. Excluded raw values are corrected
                while retaining exclusions; original non-finite values remain so.
        """
        if not isinstance(inplace, (bool, np.bool_)):
            raise TypeError("inplace must be Boolean.")
        if inplace:
            raise NotImplementedError("In-place correction is deferred; use the independent returned image.")
        if isinstance(coordinates, Enum) and not isinstance(coordinates, CoordinateMode):
            raise TypeError("coordinates must be CoordinateMode or its supported strings.")
        if isinstance(reference, Enum) and not isinstance(reference, BackgroundReference):
            raise TypeError("reference must be BackgroundReference, its string or a finite intensity.")
        if isinstance(reference, str):
            reference = BackgroundReference(reference)
        elif isinstance(reference, (bool, np.bool_)) or np.ndim(reference) != 0 or not np.isrealobj(reference):
            raise TypeError("reference must be a finite real intensity or BackgroundReference.")
        elif not np.isfinite(reference):
            raise ValueError("reference must be finite.")
        if isinstance(model, ModelResult):
            if (any(value is not None for value in (x, y, p0, sigma, where, box, prefix)) or fit_options
                    or coordinates != CoordinateMode.NATIVE):
                raise TypeError("A captured REPORT cannot be combined with fitting-only options.")
            report = model
        else:
            report, _ = self._fit(model, x, y, p0, sigma, where, box, coordinates, prefix, fit_options)
        background, record = self._evaluate(report)
        level = record["reference_mean"] if reference == BackgroundReference.REGION_MEAN else (
            0. if reference == BackgroundReference.ZERO else float(reference))
        measured = owner_of(self._image)._state.dataset.intensity.values.astype(np.float64)
        corrected = measured.copy()
        finite = np.isfinite(measured)
        corrected[finite] = measured[finite]-background[finite]+level
        return self._result(corrected, record, "correction", level)
