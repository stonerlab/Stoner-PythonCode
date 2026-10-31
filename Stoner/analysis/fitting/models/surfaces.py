"""Fit joint plane and quadratic backgrounds in physical or conditioned coordinates."""

import numpy as np
from lmfit import Model

__all__ = ["Plane", "QuadraticSurface", "plane_surface", "quadratic_surface"]


def plane_surface(x, y, offset, slope_x, slope_y):
    """Evaluate a plane in raw intensity units.

    Args:
        x, y (array-like):
            Broadcast-compatible physical coordinate arrays.
        offset, slope_x, slope_y (float):
            Intercept and gradients in intensity and intensity/coordinate units.

    Returns:
        numpy.ndarray:
            offset + slope_x*x + slope_y*y at each coordinate pair.
    """
    x, y = np.asarray(x), np.asarray(y)
    return offset + slope_x*x + slope_y*y


def quadratic_surface(x, y, offset, slope_x, slope_y, quadratic_x, cross_xy, quadratic_y):
    """Evaluate a full quadratic surface including its x*y cross-term.

    Args:
        x, y (array-like):
            Broadcast-compatible physical coordinate arrays.
        offset, slope_x, slope_y (float):
            Constant and linear coefficients, as for plane_surface.
        quadratic_x, cross_xy, quadratic_y (float):
            Coefficients of x*x, x*y and y*y respectively.

    Returns:
        numpy.ndarray:
            Surface intensities at each coordinate pair, without normalisation.
    """
    x, y = np.asarray(x), np.asarray(y)
    return plane_surface(x, y, offset, slope_x, slope_y) + quadratic_x*x*x + cross_xy*x*y + quadratic_y*y*y


def design_matrix(x, y, quadratic=False):
    """Return polynomial columns in the documented physical coefficient order."""
    x, y = np.broadcast_arrays(x, y)
    columns = [np.ones_like(x), x, y]
    if quadratic:
        columns.extend((x*x, x*y, y*y))
    return np.stack(columns, axis=-1)


def conditioning(x, y, quadratic):
    """Return centred/scaled coordinates and the full coefficient transformation."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    sx, sy = float(np.max(np.abs(x-cx))), float(np.max(np.abs(y-cy)))
    sx, sy = sx or 1., sy or 1.
    transform = np.array([[1, -cx/sx, -cy/sy], [0, 1/sx, 0], [0, 0, 1/sy]], dtype=float)
    if quadratic:
        transform = np.array([
            [1, -cx/sx, -cy/sy, (cx/sx)**2, cx*cy/(sx*sy), (cy/sy)**2],
            [0, 1/sx, 0, -2*cx/sx**2, -cy/(sx*sy), 0],
            [0, 0, 1/sy, 0, -cx/(sx*sy), -2*cy/sy**2],
            [0, 0, 0, 1/sx**2, 0, 0], [0, 0, 0, 0, 1/(sx*sy), 0],
            [0, 0, 0, 0, 0, 1/sy**2],
        ])
    if not np.isfinite(transform).all():
        raise ValueError("Coordinate transformation exceeds floating-point range.")
    return (cx, cy, sx, sy), transform


class _SurfaceModel(Model):
    """Retain stable evaluation while exposing native physical parameter expressions."""

    _stoner_independent_vars = ("x", "y")
    _quadratic = False

    def __init__(self, **kwargs):
        super().__init__(quadratic_surface if self._quadratic else plane_surface, independent_vars=["x", "y"], **kwargs)
        self._conditioning = None
        for name in self.param_names:
            self.set_param_hint(name[len(self.prefix):], value=0.)

    def guess(self, data, x=None, y=None, **kwargs):
        """Guess physical coefficients using a centred full-rank linear solution."""
        x, y = np.asarray(x), np.asarray(y)
        normal, transform = conditioning(x, y, self._quadratic)
        cx, cy, sx, sy = normal
        matrix = design_matrix((x-cx)/sx, (y-cy)/sy, self._quadratic)
        solution, _, rank, _ = np.linalg.lstsq(matrix, data, rcond=None)
        if rank != matrix.shape[1]:
            raise ValueError("Reference region does not identify the surface coefficients.")
        return self.make_params(**dict(zip(self.param_names, transform @ solution)))

    def _condition(self, params, normal, transform, initial):
        """Fit normal coefficients, exposing physical values through lmfit expressions."""
        self._conditioning = tuple(normal)
        self._normal_names = tuple(f"{self.prefix}image_normal_{i}" for i in range(len(initial)))
        for name, value in zip(self._normal_names, initial):
            params.add(name, value=float(value))
        for row, name in zip(transform, self.param_names):
            expression = "+".join(f"({value!r})*{var}" for value, var in zip(row.tolist(), self._normal_names))
            params[name].set(expr=expression)
        params.update_constraints()
        return params

    def eval(self, params=None, **kwargs):
        physical_override = any(name in kwargs or name[len(self.prefix):] in kwargs for name in self.param_names)
        if (self._conditioning is not None and params is not None and not physical_override
                and all(name in params for name in self._normal_names)):
            cx, cy, sx, sy = self._conditioning
            matrix = design_matrix((np.asarray(kwargs["x"])-cx)/sx, (np.asarray(kwargs["y"])-cy)/sy, self._quadratic)
            return matrix @ np.array([params[name].value for name in self._normal_names])
        return super().eval(params=params, **kwargs)

    def _jacobian(self, params, data, weights, x=None, y=None):
        """Return the exact weighted residual Jacobian for conditioned free coefficients."""
        cx, cy, sx, sy = self._conditioning
        matrix = design_matrix((np.asarray(x)-cx)/sx, (np.asarray(y)-cy)/sy, self._quadratic)
        return -matrix if weights is None else -matrix*np.asarray(weights)[:, None]


class Plane(_SurfaceModel):
    """Joint plane model with offset, slope_x and slope_y in physical coordinates."""


class QuadraticSurface(_SurfaceModel):
    """Joint quadratic model including quadratic_x, cross_xy and quadratic_y."""

    _quadratic = True
