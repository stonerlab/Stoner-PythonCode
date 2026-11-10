"""Prepare image observations and preserve immutable scientific fitting context."""

import json
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from types import SimpleNamespace

import numpy as np
from lmfit import Model, Parameters

from ..analysis.fitting.models import model_registry
from ..analysis.fitting.models.surfaces import (
    _SurfaceModel,
    conditioning,
    design_matrix,
)
from ..core.selection import resolve_selection
from .coordinates import resolve_grid
from .numerical import box as legacy_box
from .roi import image_selection
from .storage_bridge import owner_of

__all__ = ["ImageFitContext"]

# NumPy exposes this dtype descriptor dynamically.
_FLOAT_EPS = np.finfo(float).eps  # pylint: disable=no-member


@dataclass(frozen=True, slots=True)
class ImageFitContext:
    """Immutable versioned provenance attached as native ModelResult.image_fit.

    Attributes:
        record (str):
            JSON snapshot of coordinates, selection, eligibility, coefficients,
            physical covariance, uncertainties and conditioning; contains no owner.
        parameters (str):
            Frozen lmfit Parameters serialisation for accessor evaluation/replay.

    Notes:
        Native report covariance follows report.var_names. Physical coefficient
        covariance is available through to_dict and the FIT output.
    """

    record: str
    parameters: str

    def to_dict(self):
        """Return a detached version-1 scientific record, independent of later report edits."""
        return json.loads(self.record)


def construct_model(model):
    """Construct/copy a model only on invocation, validating two independent variables."""
    key = model if isinstance(model, str) else getattr(model, "__name__", type(model).__name__)
    if isinstance(model, str):
        model = model_registry.create(model)
    elif isinstance(model, Model):
        model = deepcopy(model)
    elif isinstance(model, type) and issubclass(model, Model):
        model = model()
    elif callable(model):
        variables = list(signature(model).parameters)[:2]
        model = Model(model, independent_vars=variables)
    else:
        raise TypeError("model must be a registered key, lmfit Model/class or two-variable function.")
    if len(model.independent_vars) != 2:
        raise ValueError("Image fitting requires exactly two independent variables, in (x, y) order.")
    if not model.param_names:
        raise ValueError("Image fitting requires at least one model coefficient.")
    if isinstance(model, _SurfaceModel):
        model._conditioning = None
    return model, str(key)


def prepare(image, x, y, sigma, where, box, coordinates):
    """Snapshot observations with shared ROI/coordinate rules and exact uncertainty alignment."""
    owner = owner_of(image)
    if owner is None:
        raise ValueError("Image fitting requires a populated ImageFile.")
    owner._check_writable()
    grid = resolve_grid(image, x, y, coordinates)
    selected = _fitting_selection(image, owner, where, box)
    dataset = owner._state.dataset.copy(deep=True)
    target = dataset.intensity
    if target.dtype.kind not in "biuf":
        raise TypeError("Image fitting requires real intensities.")
    xx, yy = np.meshgrid(grid.x, grid.y)
    eligible = selected & ~dataset.excluded.values & np.isfinite(target.values)
    eligible &= np.isfinite(xx) & np.isfinite(yy)
    if not eligible.any():
        raise ValueError("No eligible reference pixels remain for image fitting.")
    weights = _fitting_weights(sigma, target, eligible)
    return dataset, grid, selected, eligible, xx[eligible], yy[eligible], target.values[eligible].astype(float), weights


def rank_check(matrix):
    """Require positive residual degrees of freedom and identifiable, conditioned free coefficients."""
    if matrix.ndim != 2 or matrix.shape[1] == 0 or matrix.shape[0] <= matrix.shape[1]:
        raise ValueError("The reference region needs more eligible pixels than free coefficients.")
    scales = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError("Reference geometry does not identify every free coefficient.")
    singular = np.linalg.svd(matrix/scales, compute_uv=False)
    condition = singular[0]/singular[-1] if singular[-1] else np.inf
    rank = int(np.sum(singular > singular[0]*max(matrix.shape)*_FLOAT_EPS))
    if rank != matrix.shape[1] or condition > 1e10:
        raise ValueError("Reference geometry is rank deficient or ill-conditioned for the free coefficients.")
    return rank, float(condition)


def parameters(model, data, independent, p0, weights):
    """Prepare native constraints or a stable normal-coordinate surface parameterisation."""
    names = model.param_names
    params = _initial_parameters(model, data, independent, p0)
    unconstrained = (set(params) == set(names) and all(
        param.vary and param.expr is None and np.isneginf(param.min) and np.isposinf(param.max)
        for param in params.values()))
    if isinstance(model, _SurfaceModel) and unconstrained:
        return _condition_parameters(model, data, independent, params, p0, weights)
    free = [name for name, param in params.items() if param.vary and not param.expr]
    jacobian = numerical_jacobian(model, params, independent, free)
    rank, condition = rank_check(jacobian*weights[:, None])
    return params, None, None, rank, condition


def numerical_jacobian(model, params, independent, free):
    """Differentiate native model values while respecting parameter expressions."""
    columns = []
    for name in free:
        step = np.cbrt(_FLOAT_EPS)*max(1., abs(params[name].value))
        plus, minus = deepcopy(params), deepcopy(params)
        plus[name].value += step
        minus[name].value -= step
        plus.update_constraints()
        minus.update_constraints()
        width = plus[name].value - minus[name].value
        if width == 0:
            raise ValueError("A free parameter has no resolvable range.")
        columns.append((model.eval(params=plus, **independent)-model.eval(params=minus, **independent))/width)
    return np.column_stack(columns) if columns else np.empty((len(next(iter(independent.values()))), 0))


def physical_covariance(report, transform):
    """Transform complete covariance, retaining correlations and fixed coefficients."""
    if report.covar is None or not np.isfinite(report.covar).all():
        raise RuntimeError("lmfit did not estimate a finite covariance for this fit.")
    if transform is not None:
        columns = [report.model._normal_names.index(name) for name in report.var_names]
        jacobian = transform[:, columns]
    else:
        names = report.model.param_names
        columns = []
        for name in report.var_names:
            step = np.cbrt(_FLOAT_EPS)*max(1., abs(report.params[name].value))
            plus, minus = deepcopy(report.params), deepcopy(report.params)
            plus[name].value += step
            minus[name].value -= step
            plus.update_constraints()
            minus.update_constraints()
            width = plus[name].value-minus[name].value
            columns.append([(plus[p].value-minus[p].value)/width for p in names])
        jacobian = np.column_stack(columns)
    covariance = jacobian @ report.covar @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("Physical coefficient covariance exceeds floating-point range.")
    return covariance


def _fitting_selection(image, owner, where, box):
    """Intersect explicit pixel bounds with the shared current or supplied ROI."""
    selected = image_selection(image, where)
    if box is not None:
        # The parser needs only geometry and metadata. Do not construct/crop a
        # numerical image, copy intensities or invoke an interactive selector.
        reference = SimpleNamespace(shape=owner.shape, max_box=(0, owner.shape[1], 0, owner.shape[0]),
                                    metadata=image.metadata)
        if isinstance(box, str):
            box = image.metadata[box]
            if isinstance(box, str) or not hasattr(box, "__len__") or len(box) != 4:
                raise ValueError("A metadata fitting box must contain four explicit pixel bounds.")
        bounds = legacy_box(reference, box)
        inside = np.zeros(owner.shape, dtype=bool)
        inside[bounds] = True
        selected &= inside
    return selected


def _fitting_weights(sigma, target, eligible):
    """Validate aligned participating standard deviations and invert them safely."""
    if sigma is None:
        uncertainty = np.ones(target.shape)
    elif np.ndim(sigma) == 0 and not hasattr(sigma, "dims"):
        if isinstance(sigma, (bool, np.bool_)) or not np.isrealobj(sigma):
            raise TypeError("sigma must contain positive real standard deviations.")
        uncertainty = np.full(target.shape, float(sigma))
    else:
        if hasattr(sigma, "dims"):
            resolve_selection(sigma.notnull(), target)
            sigma = sigma.values
        uncertainty = np.ma.asarray(sigma)
        if uncertainty.shape != target.shape or uncertainty.dtype.kind not in "iuf":
            raise ValueError("sigma must be a real array matching the exact image geometry.")
        if np.ma.getmaskarray(uncertainty)[eligible].any():
            raise ValueError("Participating sigma values must not be masked.")
        uncertainty = uncertainty.data
    errors = uncertainty[eligible]
    if not np.isfinite(errors).all() or np.any(errors <= 0):
        raise ValueError("Participating sigma values must be finite and strictly positive.")
    weights = 1/errors
    if not np.isfinite(weights).all() or np.any(weights == 0):
        raise ValueError("sigma produces non-finite or zero fitting weights.")
    return weights


def _initial_parameters(model, data, independent, p0):
    """Construct independent native parameters and validate initial coefficient values."""
    names = model.param_names
    if isinstance(p0, Parameters):
        params = deepcopy(p0)
    else:
        if callable(p0):
            p0 = p0(data, **independent)
        if p0 is None and not isinstance(model, _SurfaceModel):
            try:
                p0 = model.guess(data, **independent)
            except NotImplementedError:
                p0 = np.ones(len(names))
        if isinstance(p0, Parameters):
            params = deepcopy(p0)
        elif p0 is None:
            params = model.make_params()
        elif isinstance(p0, dict):
            permitted = set(names) | {name[len(model.prefix):] for name in names}
            if set(p0)-permitted:
                raise ValueError("p0 contains an unrecognised model coefficient.")
            params = model.make_params(**p0)
        else:
            values = np.asarray(p0, dtype=float)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("p0 must supply one finite initial value per model coefficient.")
            params = model.make_params(**dict(zip(names, values)))
    if not set(names).issubset(params):
        raise ValueError("p0 Parameters must include every model coefficient.")
    params.update_constraints()
    if not all(np.isfinite(param.value) for param in params.values()):
        raise ValueError("Initial model coefficients must be finite.")
    return params


def _condition_parameters(model, data, independent, params, p0, weights):
    """Solve and initialise an identifiable surface in stable normal coordinates."""
    x, y = independent.values()
    normal, transform = conditioning(x, y, model._quadratic)
    cx, cy, sx, sy = normal
    matrix = design_matrix((x-cx)/sx, (y-cy)/sy, model._quadratic)
    rank, condition = rank_check(matrix*weights[:, None])
    initial = np.linalg.lstsq(matrix*weights[:, None], data*weights, rcond=None)[0]
    if p0 is not None:
        physical = np.array([params[name].value for name in model.param_names], dtype=np.longdouble)
        initial = _normal_initial(physical, normal, model._quadratic)
    params = model._condition(params, normal, transform, initial)
    return params, normal, transform, rank, condition


def _normal_initial(physical, normal, quadratic):
    """Transform supplied physical coefficients into the stable fitting basis."""
    cx, cy, sx, sy = normal
    a, b, c = physical[:3]
    initial = np.array([a+b*cx+c*cy, b*sx, c*sy], dtype=np.longdouble)
    if quadratic:
        d, e, f = physical[3:]
        initial = np.array([a+b*cx+c*cy+d*cx*cx+e*cx*cy+f*cy*cy,
                            sx*(b+2*d*cx+e*cy), sy*(c+e*cx+2*f*cy), d*sx*sx, e*sx*sy, f*sy*sy])
    return initial
