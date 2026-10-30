"""Classes and support functions for the :py:attr:`~Stoner.folders.core.BaseFolder.each`.magic attribute."""

__all__ = ["FitProxy", "Item"]
from collections.abc import MutableSequence
from functools import partial, wraps
from inspect import signature
from types import MappingProxyType
from traceback import format_exc

from multiprocess.pool import Pool as ProcessPool
from multiprocess.reduction import ForkingPickler
import numpy as np

from ..analysis.fitting.accessor import DataFitter, _ModelFit
from ..analysis.fitting.registry import model_registry
from ..compat import string_types
from ..plot.accessor import _FIELD_KINDS, _STAT_KINDS, _XY_KINDS
from ..tools import isiterable
from .utils import get_pool

_FIT_ALGORITHMS = ("lmfit", "curve_fit", "odr", "differential_evolution", "polyfit")


def _is_fit_call(func):
    """Validate a fitting path and distinguish it from other bulk operations."""
    if not isinstance(func, str) or (func != "fit" and not func.startswith("fit.")):
        return False
    if func != "fit" and func[4:] not in (*_FIT_ALGORITHMS, *model_registry):
        raise AttributeError(f"Unsupported bulk fitting method {func!r}.")
    return True


def _worker(d, **kwargs):
    """Support function to run an arbitrary function over a :py:class:`~Stoner.core.data.Data` object."""
    byname = kwargs.get("byname", False)
    func = kwargs.get("func", lambda x: x)
    source_name = func
    registration = kwargs.get("fit_registration")
    plotting = byname and isinstance(func, str) and (func == "plot" or func.startswith("plot."))
    args = kwargs.get("args", ())
    kwargs = kwargs.get("kwargs", {})
    try:
        if byname and registration is not None:
            # Resolve the owner accessor in the worker, without installing custom
            # registrations in a process/thread's global catalogue.
            func = _ModelFit(d.fit, registration.name, registration=registration)
        elif byname:
            resolved = d
            for part in func.split("."):
                resolved = getattr(resolved, part)
            func = resolved
        if not plotting:
            if hasattr(d, "setas"):
                d["setas"] = list(d.setas)
            d["args"] = args
            d["kwargs"] = kwargs
            d["func"] = (
                source_name.rsplit(".", 1)[-1]
                if byname and registration is None and _is_fit_call(source_name)
                else getattr(func, "__name__", type(func).__name__)
            )
        if byname:  # Ut's an instance bound moethod
            ret = func(*args, **kwargs)
        else:  # It's an arbitrary function
            ret = func(d, *args, **kwargs)
    # User callbacks can fail arbitrarily; the bulk API returns their traceback.
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        ret = e, format_exc()
    return (d, ret)


class _FitCall:
    """Retain a bulk fitting path, resolving the registration at invocation."""

    __slots__ = ("_proxy", "_name")

    def __init__(self, proxy, name):
        self._proxy = proxy
        self._name = name

    @property
    def __name__(self):
        return self._name

    @property
    def __signature__(self):
        method = getattr(DataFitter, self._name) if self._name in _FIT_ALGORITHMS else _ModelFit.__call__
        sig = signature(method)
        return sig.replace(parameters=list(sig.parameters.values())[1:])

    @property
    def __doc__(self):
        detail = (
            getattr(DataFitter, self._name).__doc__
            if self._name in _FIT_ALGORITHMS else _ModelFit(None, self._name).__doc__
        )
        return f"Fit folder members in order; _serial and _return follow each.iter.\n\n{detail}"

    def __call__(self, *args, **kwargs):
        return self._proxy._folder.each(f"fit.{self._name}", *args, **kwargs)


class FitProxy:
    """Fit folder members using explicit algorithms or registered-model shortcuts.

    Args:
        folder (DataFolder):
            Folder owning the member measurements.

    Notes:
        Calling the proxy defaults to lmfit. The models property is a live,
        read-only catalogue. Lookup and completion do not load folder members.
        Coordinate selectors use x/y, as on Data.fit. Calls return results in
        folder order and honour existing _serial and _return controls.

        Each shortcut invocation snapshots the selected registration for that
        bulk operation. Workers receive its definition and create member-bound
        accessors locally; they do not change their global model registry.
    """

    __slots__ = ("_folder",)

    def __init__(self, folder):
        """Retain the folder without loading measurements."""
        self._folder = folder

    @property
    def models(self):
        """Return a live read-only view of registered model definitions."""
        return MappingProxyType(model_registry)

    def __call__(self, model, x=None, y=None, p0=None, sigma=None, **kwargs):
        """Fit every member with lmfit, using the Data.fit calling convention.

        Args:
            model (lmfit.Model, type or callable):
                Model accepted by Data.fit.lmfit.
            x, y (column selector or None):
                Member coordinate/response columns, defaulting to each member's roles.
            p0 (iterable, lmfit.Parameters, callable or None):
                Initial parameter specification.
            sigma (column selector, array or None):
                Y uncertainty specification passed to each member.

        Keyword Arguments:
            **kwargs (dict):
                Fitting options and existing _serial/_return bulk controls.

        Returns:
            list:
                Member results, or existing (exception, traceback) failure tuples,
                in folder order. Updated measurements are retained in the folder.
        """
        return self._folder.each("fit", model, x, y, p0, sigma, **kwargs)

    def __getattr__(self, name):
        """Return an explicit algorithm call or a registered-model bulk shortcut."""
        if name.startswith("_"):
            raise AttributeError(name)
        _is_fit_call(f"fit.{name}")
        return _FitCall(self, name)

    def __dir__(self):
        """Include fitting algorithms and registered models without loading members."""
        return sorted(set(super().__dir__()) | set(_FIT_ALGORITHMS) | set(model_registry))


class PlotProxy:
    """Call member plotting accessors serially without collecting accessor objects.

    Only drawing methods are exposed. Configure styles/layouts outside bulk calls
    and pass explicit ax handles when overlaying members on a shared plot.
    """

    __slots__ = ("_folder",)

    def __init__(self, folder):
        """Retain the folder without loading members or creating figures."""
        self._folder = folder

    def __call__(self, *args, **kwargs):
        """Return primary axes from automatic member plots in folder order."""
        return self._folder.each("plot", *args, **kwargs)

    def __getattr__(self, name):
        """Return a serial bulk call for a declared DataPlotter drawing method."""
        if name not in (*_XY_KINDS, *_FIELD_KINDS, *_STAT_KINDS):
            raise AttributeError(f"each.plot exposes drawing methods, not {name!r}; configure individual accessors.")

        def draw(*args, **kwargs):
            return self._folder.each(f"plot.{name}", *args, **kwargs)

        draw.__name__ = name
        return draw


class SetasWrapper(MutableSequence):
    """Manages wrapping each member of the folder's setas attribute."""

    def __init__(self, parent):
        """Note a reference to the parent item class instance and folder."""
        self._each = parent
        self._folder = parent._folder

    def __call__(self, *args, **kwargs):
        """Pass through the calls the setas method of each item in our folder."""
        _ = self._folder._object_attrs.pop("setas", None)
        for obj in self._folder:
            obj.setas(*args, **kwargs)
        self._folder._object_attrs["setas"] = self.collapse()

        return self._folder

    def __len__(self):
        """Return the shortest length of all the setas elements in the folder."""
        lengths = np.array([len(data.setas) for data in self._folder])
        return abs(lengths.min())

    def __getitem__(self, index):
        """Get the corresponding item from all the setas items in the folder."""
        return [data.setas[index] for data in self._folder]

    def __setitem__(self, index, value):
        """Set the value of the specified item on the setas elements in the folder.

        Args:
            index (int):
                The column index of the individual setas attributes to change
            value (iterable):
                The set of values to apply to the setas attributes.

        Notes:
            If value has a length less than the data folder, then the final value is repeated to encompass the
            remaining elements.
        """
        if len(value) < len(self._folder):
            value = value + value[-1] * (len(self._folder) - len(value))
        v = "."
        for v, data in zip(value, self._folder):
            data.setas[index] = v
        setas = self._folder._object_attrs.get("setas", self.collapse())
        setas[index] = v
        self._folder._object_attrs["setas"] = setas

    def __delitem__(self, index):
        """Cannot delete items from the proxy setas object - so simply clear it instead."""
        for data in self._folder:
            data.setas[index] = "."
        setas = self._folder._object_attrs.get("setas", self.collapse())
        setas[index] = "."
        self._folder._object_attrs["setas"] = setas

    def insert(self, index, value):
        """Cannot insert items into the proxy setas object."""
        raise IndexError("Cannot insert into the objectFolder's setas - insdert into the objectFolder instead!")

    def collapse(self):
        """Collapse the setas into a single list if possible."""
        setas = []
        for v in self:
            if len(v):
                if np.unique(v).size == 1:
                    setas.append(v[0])
                else:
                    setas.append("-")
        return setas


class Item:
    """Provides a proxy object for accessing methods on the inividual members of a Folder.

    Notes:
        The purpose of this class is to allow it to be explicit that we're calling methods
        on the members of the folder rather than a collective method. This allows us to work
        around nameclashes.
    """

    _folder = None

    def __init__(self, folder):
        """Create the each proxy object.

        Notes the partent folder that created us."""
        self._folder = folder

    @property
    def fit(self):
        """Return member fitting calls with algorithm and registered-model shortcuts."""
        return FitProxy(self._folder)

    @property
    def plot(self):
        """Return serial automatic/named member plotting calls, without reading data."""
        return PlotProxy(self._folder)

    @property
    def setas(self):
        """Return a proxy object for manipulating all the setas objects in a folder."""
        return SetasWrapper(self)

    @setas.setter
    def setas(self, value):
        """Manipualte the setas property of all the objects in the folder."""
        setas = self.setas
        setas(value)
        self._folder._object_attrs["setas"] = setas.collapse()
        return setas

    def __call__(self, func, *args, **kwargs):
        """Iterate over the BaseFolder, calling func on each item.

        Args:
            func (callable, str):
                Either a callable object, or the name of a callable object (either method or global) that must take
                a metadataObject type instance as it's first argument.

        Keyword Arguments:
            _return (None, bool or str): Controls how the return value from *func* is added to the DataFolder

        Returns:
            A list of the results of evaluating *func* for each item in the folder.

        Notes:
            If *_return* is None and the return type of *func* is the same type as the :py:class:`BaseFolder` is
            storing, then the return value replaces the original :py:class:`Stoner.core.base.metadataObject` in the
            :py:class:`BaseFolder`. If *_result* is True the return value is added to the
            :py:class:`Stoner.core.base.metadataObject`'s metadata under the name of the function. If *_result* is a
            string. then return result is stored in the corresponding name.
        """
        # Just call the iter generator but assemble into a list.
        if _is_fit_call(func):
            kwargs["_byname"] = True
        if isinstance(func, str) and (func == "plot" or func.startswith("plot.")):
            if func != "plot" and func[5:] not in (*_XY_KINDS, *_FIELD_KINDS, *_STAT_KINDS):
                raise AttributeError(f"Unsupported bulk plotting method {func!r}.")
            kwargs["_byname"] = True
        if isinstance(func, string_types) and "_byname" not in kwargs:
            if func in globals() and callable(globals()[func]):
                func = globals()[func]
            else:
                func = getattr(self, func)
        return list(self.iter(func, *args, **kwargs))

    def __dir__(self):
        """Return a list of the common set of attributes of the instances in the folder."""
        if self._folder and len(self._folder) != 0:
            res = set(dir(self._folder[0]))
        else:
            res = set()
        if self._folder and len(self._folder) > 0:
            for d in self._folder[1:]:
                res &= set(dir(d))
        return list(res)

    def __delattr__(self, name):
        """Handle removing an attribute from the folder, including proxied attributes."""
        if name in dir(self._folder.instance) or (
            len(self._folder) and hasattr(self._folder[0], name)
        ):  # This is an instance attribute
            # If we're tracking the object attributes and have a type set, then we can store this for adding to all
            # loaded objects on read.
            del self._folder._object_attrs[name]
            for _, this in self._folder.loaded:
                try:
                    delattr(this, name)
                except AttributeError:
                    pass
        elif name in self._folder._instance_attrs:
            delattr(self._folder, name)
        else:
            raise AttributeError(f"Unrecognised attribute {name}")

    def __getattr__(self, name):
        """Handle some special case attributes that provide alternative views of the objectFolder.

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        Notes:
            If *name* is not present on the empty member instance, then the first member of the folder is checked as
            well. This allows the attributes of a :py:class:`~Stoner.core.data.Data` object that derive from the
            :py:attr:`Stoner.core.data.Data.setas` attribute (such as *.x*, *.y* or *.e* etc.) can be accessed.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            instance = self._folder.instance
            if callable(getattr(instance, name, None)):  # It's a method
                ret = self.__getattr_proxy(name)
            else:  # It's a static attribute
                if name in self._folder._object_attrs and len(self._folder) == 0:
                    ret = self._folder._object_attrs[name]
                elif len(self._folder):
                    ret = [(not hasattr(x, name), getattr(x, name, None)) for x in self._folder]
                    mask, values = zip(*ret)
                    lens = np.unique([len(v) if isiterable(v) else 0 for v in values])
                    if lens.size == 1:
                        ret = np.ma.MaskedArray(values)
                    else:
                        ret = np.ma.MaskedArray(values, dtype=object)
                    ret.mask = mask
                else:
                    ret = getattr(instance, name, None)
                if ret is None or (hasattr(ret, "mask") and np.all(ret.mask)):
                    raise AttributeError
        except AttributeError as err:  # Ok, pass back
            raise AttributeError(f"{name} is not an Attribute of {type(self)} or {type(instance)}") from err
        # except TypeError as err:  # Can be triggered if self.instance lacks the attribute
        #     if len(self._folder) and hasattr(self._folder[0], name):
        #         ret = [(not hasattr(x, name), getattr(x, name, None)) for x in self._folder]
        #         mask, values = zip(*ret)
        #         ret = np.ma.MaskedArray(values)
        #         ret.mask = mask
        #     else:
        #         raise err

        return ret

    def __setattr__(self, name, value):
        """Proxy call to set an attribute.

        Setting the attribute on .each sets it on all instantiated objects and in _object_attrs.

        Args:
            name(str): Attribute to set
            value (any): Value to set

        Notes:
            If *name* is not present on the empty member instance, then the first member of the folder is checked as
            well. This allows the attributes of a :py:class:`~Stoner.core.data.Data` object that derive from the
            :py:attr:`Stoner.core.data.Data.setas` attribute (such as *.x*, *.y* or *.e* etc.) can be accessed.

            If *value* is iterable and the same length as the folder, then each element in the folder is loaded and
            the corresponding element of *value* is assigned to the attribute of the member.
        """
        if hasattr(type(self), name) or name.startswith("_"):  # Handle setting our own attributes
            super().__setattr__(name, value)
        elif name in dir(self._folder.instance) or (
            len(self._folder) and hasattr(self._folder[0], name)
        ):  # This is an instance attribute
            if isiterable(value) and len(value) == len(self._folder):
                force_load = True
            else:
                force_load = False
                self._folder._object_attrs[name] = value  # Add to attributes to be set on load
                value = [value] * len(self._folder)
            for d, v in zip(self._folder.__names__(), value):  # And set on all instantiated objects
                if force_load or isinstance(self._folder.__getter__(d, instantiate=False), self._folder.type):
                    d = self._folder.__getter__(d)
                    setattr(d, name, v)
        else:
            raise AttributeError(f"Unknown attribute {name}")

    def __getattr_proxy(self, item):
        """Make a prpoxy call to access a method of the metadataObject like types.

        Args:
            item (string): Name of method of metadataObject class to be called

        Returns:
            Either a modified copy of this objectFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        meth = getattr(self._folder.instance, item, None)

        @wraps(meth)
        def _wrapper_(*args, **kwargs):
            """Wrap a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None): specify to store the return value in the individual object's metadata

            Notes:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item
            """
            kwargs["_byname"] = True
            return self(item, *args, **kwargs)  # Develove to self.__call__ where we have multiprocess magic

        # Ok that's the wrapper function, now return  it for the user to mess around with.
        return _wrapper_

    def __rmatmul__(self, other):
        """Implement callable@DataFolder as a generic iterate a function over DataFolder members.

        Returns:
            An object that supports __call__ and knows about this DataFolder.
        """
        if not callable(other):
            return NotImplemented

        @wraps(other)
        def _wrapper_(*args, **kwargs):
            """Wrap a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None): specify to store the return value in the individual object's metadata

            Notes:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item
            """
            kwargs["_byname"] = False  # Force the __call__ to use the callable function
            return self(other, *args, **kwargs)  # Delegate to self.__call__ which has multiprocess magic.

        # Ok that's the wrapper function, now return  it for the user to mess around with.
        return _wrapper_

    def iter(self, func, *args, **kwargs):
        """Iterate over the BaseFolder, calling func on each item.

        Args:
            func (callable): A Callable object that must take a metadataObject type instance as it's first argument.

        Keyword Arguments:
            _return (None, bool or str): Controls how the return value from *func* is added to the DataFolder

        Returns:
            A list of the results of evaluating *func* for each item in the folder.

        Notes:
            If *_return* is None and the return type of *func* is the same type as the :py:class:`BaseFolder` is
            storing, then the return value replaces the original :py:class:`Stoner.core.base.metadataObject` in the
            :py:class:`BaseFolder`. If *_result* is True the return value is added to the
            :py:class:`Stoner.core.base.metadataObject`'s metadata under the name of the function. If *_result* is a
            string. then return result is stored in the corresponding name.
        """
        fitting = _is_fit_call(func)
        if fitting:
            kwargs["_byname"] = True
        if isinstance(func, str) and (func == "plot" or func.startswith("plot.")):
            if func != "plot" and func[5:] not in (*_XY_KINDS, *_FIELD_KINDS, *_STAT_KINDS):
                raise AttributeError(f"Unsupported bulk plotting method {func!r}.")
            kwargs["_byname"] = True
        _return = kwargs.pop("_return", None)
        _byname = kwargs.pop("_byname", False)
        _serial = kwargs.pop("_serial", False)
        plotting = _byname and isinstance(func, str) and (func == "plot" or func.startswith("plot."))
        if plotting:
            _serial = True
        else:
            self._folder.fetch()  # Preserve generic bulk prefetching; plots load on the rendering thread.
        registration = (
            model_registry.registration(func[4:])
            if fitting and func.startswith("fit.") and func[4:] not in _FIT_ALGORITHMS else None
        )
        p, imap = get_pool(_serial)
        worker = partial(_worker, func=func, args=args, kwargs=kwargs, byname=_byname,
                         fit_registration=registration)
        complete = False
        try:
            if fitting and isinstance(p, ProcessPool):
                try:
                    ForkingPickler.dumps(worker)
                # User model definitions can fail arbitrary serializer hooks.
                except Exception as err:  # noqa: BLE001
                    raise TypeError(
                        f"Cannot transfer bulk fitting call {func!r} to workers; "
                        "use _serial=True or provide a serialisable model/factory definition"
                    ) from err
            items = (item.clone for item in self._folder) if "_stack_owner" in self._folder.__dict__ else self._folder
            yield from self._collect(imap(worker, items), func, _return)
            complete = True
        finally:
            if p is not None:
                if complete:
                    p.close()
                else:
                    p.terminate()
                p.join()

    def _collect(self, results, func, _return):
        """Publish member updates and retain the existing bulk-return conventions."""
        for ix, (new_d, ret) in enumerate(results):
            if self._folder.debug:
                print(ix, type(ret))
            if isinstance(ret, self._folder._type) and _return is None:
                try:  # Check if ret has same data type, otherwise will not overwrite well
                    if ret.dtype != new_d.dtype:
                        continue
                    new_d = ret
                except AttributeError:
                    pass
            elif _return is not None:
                if isinstance(_return, bool) and _return:
                    _return = (
                        func.rsplit(".", 1)[-1]
                        if isinstance(func, str)
                        else getattr(func, "__name__", type(func).__name__)
                    )
                new_d[_return] = ret
            name = self._folder.__names__()[ix]
            self._folder.__setter__(name, new_d)
            yield ret
