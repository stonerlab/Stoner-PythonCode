"""Register named model classes and factories without constructing them."""

from collections.abc import Mapping
from dataclasses import dataclass
import keyword
import re

from lmfit import Model

__all__ = ["ModelRegistration", "ModelRegistry", "model_registry", "register_model"]

_RESERVED_NAMES = frozenset({"curve_fit", "lmfit", "odr", "differential_evolution", "polyfit", "models"})


def _model_name(name):
    """Convert a public class/function name to ASCII snake_case."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _validate_name(name):
    """Reject invalid, private, reserved or non-canonical registry keys."""
    if not isinstance(name, str):
        raise TypeError("A model name must be a string")
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) or keyword.iskeyword(name):
        raise ValueError(f"Model name {name!r} must be a public ASCII snake_case identifier")
    if name in _RESERVED_NAMES:
        raise ValueError(f"Model name {name!r} is reserved for the fitting accessor")
    return name


def _create_model(model, name, **kwargs):
    """Invoke a registered factory and validate its result without registry mutation."""
    result = model(**kwargs)
    if not isinstance(result, Model):
        raise TypeError(f"Factory for model {name!r} returned {type(result).__name__}, expected lmfit.Model")
    return result


@dataclass(frozen=True)
class ModelRegistration:
    """Describe a registered class or factory without holding a model instance.

    Attributes:
        name (str):
            Public snake_case registry key.
        model (callable):
            Model subclass or factory, called only when a model is requested.
        module (str):
            Source module of the registered callable.
        qualname (str):
            Qualified source name used in collision diagnostics.
        display_name (str):
            Original public class or decorated-function name.
    """

    name: str
    model: object
    module: str
    qualname: str
    display_name: str


class ModelRegistry(Mapping):
    """Map public names to model classes/factories through controlled registration.

    Notes:
        Mapping access is read-only and live. Registration does not construct a
        model. Each create call invokes the registered class/factory afresh.
        Registries are process-local; custom registrations are not automatically
        installed in newly spawned worker processes.
    """

    def __init__(self):
        """Initialise an empty registry."""
        self._entries = {}

    def __getitem__(self, name):
        """Return a registered callable, raising KeyError for an unknown name."""
        return self._entries[name].model

    def __iter__(self):
        """Iterate registered keys in registration order."""
        return iter(self._entries)

    def __len__(self):
        """Return the number of registered names."""
        return len(self._entries)

    def registration(self, name):
        """Return immutable provenance for a registered name."""
        return self._entries[name]

    def register(self, model, *, name=None, replace=False):
        """Register a Model subclass or explicitly named factory.

        Args:
            model (type or callable):
                Model subclass or factory returning a new lmfit Model.

        Keyword Arguments:
            name (str or None):
                Explicit snake_case key. None derives it from a Model subclass.
                Factories require an explicit name.
            replace (bool):
                Permit replacing a different registration. Defaults to False.

        Returns:
            callable:
                The original class/factory, unchanged for decorator use.

        Raises:
            TypeError:
                If model is an instance, non-callable or unrelated class.
            ValueError:
                If a name is invalid/reserved, a factory has no name, or a
                different object is already registered without replace=True.
        """
        is_model = isinstance(model, type) and issubclass(model, Model)
        if isinstance(model, Model) or not callable(model) or (isinstance(model, type) and not is_model):
            raise TypeError("Register an lmfit.Model subclass or a factory, not an instance or unrelated class")
        source = getattr(model, "_model_function", model)
        if name is None:
            if not is_model:
                raise ValueError("A model factory requires an explicit registry name")
            name = _model_name(source.__name__)
        name = _validate_name(name)
        info = ModelRegistration(
            name, model, getattr(source, "__module__", type(source).__module__),
            getattr(source, "__qualname__", type(source).__qualname__),
            getattr(source, "__name__", type(source).__name__),
        )
        previous = self._entries.get(name)
        if previous is not None:
            if previous.model is model:
                return model
            if not replace:
                raise ValueError(
                    f"Model {name!r} is already registered as {previous.module}.{previous.qualname}; "
                    f"cannot register {info.module}.{info.qualname} without replace=True"
                )
        self._entries[name] = info
        return model

    def create(self, name, **kwargs):
        """Construct a model through a registered class or factory.

        Args:
            name (str):
                Registered snake_case key.

        Keyword Arguments:
            **kwargs (dict):
                Constructor/factory options, such as prefix or nan_policy.

        Returns:
            lmfit.Model:
                The result of invoking the registered callable.

        Raises:
            KeyError:
                If the name is unregistered.
            TypeError:
                If a factory returns something other than an lmfit Model.

        Notes:
            A factory must return fresh state on each call. It is invoked once;
            the registry does not cache, copy or otherwise alter its result.
        """
        return _create_model(self[name], name, **kwargs)


#: Process-local catalogue shared by all measurement objects.
model_registry = ModelRegistry()


def register_model(model=None, *, name=None, replace=False):
    """Register a model in the shared catalogue, directly or as a decorator.

    Args:
        model (type, callable or None):
            Model subclass or explicitly named factory. None returns a decorator.

    Keyword Arguments:
        name (str or None):
            Registry key; None derives snake_case from a Model subclass name.
        replace (bool):
            Explicitly replace a different object under the same name.

    Returns:
        callable:
            The original model/factory, or a decorator when model is None.

    Raises:
        TypeError:
            If the supplied object cannot be registered.
        ValueError:
            If the name is invalid, reserved, missing for a factory or duplicated.
    """
    if model is None:
        return lambda candidate: register_model(candidate, name=name, replace=replace)
    return model_registry.register(model, name=name, replace=replace)
