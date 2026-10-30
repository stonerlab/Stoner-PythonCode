"""Provide owner-bound fitting algorithms and registered-model shortcuts."""

from difflib import get_close_matches
from types import MappingProxyType

from . import functions
from .registry import _create_model, model_registry

__all__ = ["DataFitter"]


def _reject_legacy_coordinates(kwargs):
    """Reject old selector keywords before reading or fitting measurements."""
    legacy = [name for name in ("xcol", "ycol") if name in kwargs]
    if legacy:
        replacements = ", ".join(f"{name}= -> {name[0]}=" for name in legacy)
        raise TypeError(f"Data.fit uses x and y column selectors; replace {replacements}")


class _ModelFit:
    """Retain an accessor and a registry key, without retaining a model instance."""

    __slots__ = ("_fitter", "_name", "_registration")

    def __init__(self, fitter, name, registration=None):
        self._fitter = fitter
        self._name = name
        self._registration = registration

    @property
    def __name__(self):
        return self._name

    @property
    def __doc__(self):
        info = self._registration or model_registry.registration(self._name)
        return (
            f"Fit the registered {self._name} model using lmfit.\n\n"
            f"Registered model: {info.module}.{info.qualname}.\n"
            "x and y select owner columns; p0, sigma and keyword options follow DataFitter.lmfit.\n"
            "The current registration is resolved on each invocation.\n\n"
            + (getattr(info.model, "__doc__", None) or "")
        )

    def __call__(self, x=None, y=None, p0=None, sigma=None, **kwargs):
        _reject_legacy_coordinates(kwargs)
        model = self._registration.model if self._registration is not None else model_registry[self._name]
        if not isinstance(model, type):
            model = _create_model(model, self._name)
        return self._fitter.lmfit(model, x, y, p0, sigma, **kwargs)


class DataFitter:
    """Fit the current columns of a measurement through existing algorithms.

    Args:
        data (Data):
            Owner of the live columns, exclusions, roles and fit metadata.

    Notes:
        The models property is a live read-only mapping from registered snake_case
        names to model classes/factories.

        Calling the accessor defaults to lmfit. Registered names such as
        ``data.fit.linear`` provide lmfit shortcuts with the model already selected.
        Lookup and completion do not construct models, read data or create figures.
        Each call uses current owner state and resolves the current registration.

        x and y are column selectors, not raw coordinate arrays. Their defaults,
        and all other fitting options and return values, follow the selected
        algorithm. Legacy xcol/ycol keywords are rejected on this accessor.
        Existing direct Data fitting methods retain their original interface.

        Explicit methods delegate to the numerical functions, bypassing legacy
        Data method overrides. To customise accessor behaviour, subclass DataFitter
        and override the Data.fit property to return that subclass. The callable
        accessor and model shortcuts honour its lmfit override.

    See Also:
        :py:attr:`Stoner.core.data.Data.fit`:
            Obtain an accessor bound to a measurement.
        :doc:`/fit-shortcuts`:
            Built-in registered-model shortcuts and their model classes.
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        """Bind to an owner without reading columns or storing an accessor on it."""
        self._data = data

    @property
    def models(self):
        """Return a live read-only view of registered model classes and factories."""
        return MappingProxyType(model_registry)

    def __getattr__(self, name):
        """Resolve a registered model name without constructing or fitting it."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in model_registry:
            return _ModelFit(self, name)
        matches = get_close_matches(name, model_registry, n=3)
        suggestion = f" Did you mean {', '.join(matches)}?" if matches else ""
        raise AttributeError(f"No fitting method or registered model named {name!r}.{suggestion}")

    def __dir__(self):
        """Include registered-model shortcuts in interactive completion."""
        return sorted(set(super().__dir__()) | set(model_registry))

    def __call__(self, model, x=None, y=None, p0=None, sigma=None, **kwargs):
        """Fit a model with lmfit; accept the same options as the lmfit method.

        Args:
            model (lmfit.Model, type or callable):
                Configured model, model class or numerical model function.
            x, y (column selector or None):
                Owner columns; None uses the existing role/default resolution.
            p0 (iterable, lmfit.Parameters, callable or None):
                Initial parameters or existing repeated-initial-guess specification.
            sigma (column selector or array or None):
                Y uncertainties accepted by the existing lmfit implementation.

        Keyword Arguments:
            **kwargs (dict):
                Existing lmfit options, including output, prefix and result storage.

        Returns:
            object:
                The existing lmfit output selected by output; defaults to fit.
        """
        return self.lmfit(model, x, y, p0, sigma, **kwargs)

    def lmfit(self, model, x=None, y=None, p0=None, sigma=None, **kwargs):
        """Fit through lmfit using x/y column selectors.

        Args:
            model (lmfit.Model, type or callable):
                Model instance, model class or numerical model function.
            x, y (column selector or None):
                Coordinate/response columns, defaulting to assigned roles.
            p0 (iterable, lmfit.Parameters, callable or None):
                Existing initial-parameter or repeated-guess specification.
            sigma (column selector or array or None):
                Y uncertainties accepted by the numerical fitting implementation.

        Keyword Arguments:
            **kwargs (dict):
                Options documented by Stoner.analysis.fitting.functions.lmfit, including
                FitOutput/string output, parameter hints, prefix and fitted columns.

        Returns:
            object:
                The selected existing lmfit output, without accessor conversion.

        Raises:
            TypeError:
                If xcol or ycol is supplied instead of x or y.

        See Also:
            :py:func:`Stoner.analysis.fitting.functions.lmfit`:
                Complete algorithm-specific options, selection and return rules.
        """
        _reject_legacy_coordinates(kwargs)
        return functions.lmfit(self._data, model, xcol=x, ycol=y, p0=p0, sigma=sigma, **kwargs)

    def curve_fit(self, func, x=None, y=None, sigma=None, **kwargs):
        """Fit with scipy curve_fit using x/y column selectors.

        Args:
            func (callable, lmfit.Model or type):
                Numerical function or existing supported model specification.
            x, y (column selector or None):
                Coordinate/response columns; retain existing defaults and multi-channel rules.
            sigma (column selector or array or None):
                Y uncertainties passed through to the existing implementation.

        Keyword Arguments:
            **kwargs (dict):
                Options documented by Stoner.analysis.fitting.functions.curve_fit, including p0, where,
                parameter_bounds, FitOutput/string output and result storage.

        Returns:
            object:
                The existing selected output or collection of channel outputs.

        Raises:
            TypeError:
                If xcol or ycol is supplied instead of x or y.

        See Also:
            :py:func:`Stoner.analysis.fitting.functions.curve_fit`.
        """
        _reject_legacy_coordinates(kwargs)
        return functions.curve_fit(self._data, func, xcol=x, ycol=y, sigma=sigma, **kwargs)

    def odr(self, model, x=None, y=None, **kwargs):
        """Fit with orthogonal distance regression using x/y column selectors.

        Args:
            model (callable, lmfit.Model, scipy.odr.Model or type):
                Model accepted by the existing ODR implementation.
            x, y (column selector or None):
                Coordinate/response columns, defaulting to assigned roles.

        Keyword Arguments:
            **kwargs (dict):
                Stoner.analysis.fitting.functions.odr options, including p0, sigma, sigma_x, scaling and output.

        Returns:
            object:
                The selected existing ODR output.

        Raises:
            TypeError:
                If xcol or ycol is supplied instead of x or y.

        See Also:
            :py:func:`Stoner.analysis.fitting.functions.odr`.
        """
        _reject_legacy_coordinates(kwargs)
        return functions.odr(self._data, model, xcol=x, ycol=y, **kwargs)

    def differential_evolution(self, model, x=None, y=None, p0=None, sigma=None, **kwargs):
        """Fit with differential evolution using x/y column selectors.

        Args:
            model (callable, lmfit.Model or type):
                Model accepted by the existing differential-evolution implementation.
            x, y (column selector or None):
                Coordinate/response columns, defaulting to assigned roles.
            p0 (iterable, lmfit.Parameters, callable or None):
                Existing algorithm-specific initial-parameter specification.
            sigma (column selector or array or None):
                Y uncertainties accepted by the implementation.

        Keyword Arguments:
            **kwargs (dict):
                Stoner.analysis.fitting.functions.differential_evolution options, including output and result storage.

        Returns:
            object:
                The selected existing differential-evolution output.

        Raises:
            TypeError:
                If xcol or ycol is supplied instead of x or y.

        See Also:
            :py:func:`Stoner.analysis.fitting.functions.differential_evolution`.
        """
        _reject_legacy_coordinates(kwargs)
        return functions.differential_evolution(self._data, model, xcol=x, ycol=y, p0=p0, sigma=sigma, **kwargs)

    def polyfit(self, x=None, y=None, polynomial_order=2, bounds=lambda x, y: True,
                result=None, replace=False, header=None):
        """Fit polynomial coefficients using the existing legacy polynomial method.

        Args:
            x, y (column selector or None):
                Coordinate/response columns, defaulting to assigned roles.
            polynomial_order (int):
                Polynomial order, defaulting to two.
            bounds (callable):
                Existing row-selection predicate; the default includes all rows.
            result (column selector, bool or None):
                Existing fitted-column storage option.
            replace (bool):
                Replace rather than insert a stored fitted column.
            header (str or None):
                Fitted-column label; None retains the existing generated label.

        Returns:
            numpy.ndarray:
                Existing coefficient vector or array of coefficient vectors.

        Notes:
            This retains the legacy method, including its existing deprecation
            note. It has no output selector and is not the accessor's default.

        See Also:
            :py:func:`Stoner.analysis.fitting.functions.polyfit`.
        """
        return functions.polyfit(self._data, x, y, polynomial_order, bounds, result, replace, header)
