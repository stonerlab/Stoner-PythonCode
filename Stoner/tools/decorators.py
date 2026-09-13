# -*- coding: utf-8 -*-
"""Provide some decorators and associated functions for modifying package classes and functions."""

import inspect
import re
from collections.abc import Iterable
from copy import copy
from functools import wraps
from importlib import import_module
from os import environ

import numpy as np

from .tests import isproperty

_RTD = "READTHEDOCS" in environ


def image_file_adaptor(workingfunc):
    """Make wrappers for ImageFile functions.

    Notes:
        The wrapped functions take additional keyword arguments that are stripped off from the call.

    Keyword Arguments:
        _box(:py:meth:`Stoner.Image.core.ImageFile.crop` arguments):
            Crops the image first before calling the parent method.
        _(bool, None):
            Controls whether a :py:class:`ImageFile` return will be substituted for the current
            :py:class:`ImageFile`.

            * True: - all ImageFile return types are substituted.
            * False (default) - Imagearray return types are substituted if they are the same size as the original
            * None - A copy of the current object is taken and the returned ImageFile provides the data.
    """
    # Avoid PEP257/black issue

    @wraps(workingfunc)
    def gen_func(self, *args, **kwargs):
        """Wrap a called method to capture the result back into the calling object."""
        from ..Image.storage_bridge import call_method
        if "_image_owner" not in self.__dict__:
            raise ValueError("Assign image pixels before numerical operations")
        return call_method(self, workingfunc, args, kwargs)

    return fix_signature(gen_func, workingfunc)


def image_file_raw_adaptor(workingfunc):
    """Make wrappers for ImageFile functions.

    Notes:
        The wrapped functions take additional keyword arguments that are stripped off from the call.

    Keyword Arguments:
        _box(:py:meth:`Stoner.Image.core.ImageFile.crop` arguments):
            Crops the image first before calling the parent method.
        _(bool, None):
            Controls whether a :py:class:`ImageFile` return will be substituted for the current
            :py:class:`ImageFile`.

            * True: - all ImageFile return types are substituted.
            * False (default) - Imagearray return types are substituted if they are the same size as the original
            * None - A copy of the current object is taken and the returned ImageFile provides the data.
    """
    # Avoid PEP257/black issue

    @wraps(workingfunc)
    def gen_func(self, *args, **kwargs):
        """Wrap a called method to capture the result back into the calling object."""
        from ..Image.storage_bridge import call_method
        if "_image_owner" not in self.__dict__:
            raise ValueError("Assign image pixels before numerical operations")
        return call_method(self, workingfunc, args, kwargs)

    return fix_signature(gen_func, workingfunc)


def array_file_property(workingfunc):
    """Wrap an arbitrary callbable to make it a bound method of this class.

    Args:
        workingfunc (callable):
            The callable object to be wrapped.

    Returns:
        (function):
            A function with enclosure that holds additional information about this object.

    The function object returned simply calls the working function having got the _image property.
    """
    if workingfunc is None:  # We may not in fact be wrapping anything here!
        return None

    @wraps(workingfunc)
    def gen_func(self, *args, **kwargs):
        """Wrap magic proxy function call."""
        from ..Image.storage_bridge import call_method
        if "_image_owner" not in self.__dict__:
            raise ValueError("Assign image pixels before numerical operations")
        return call_method(self, workingfunc, args, kwargs, setter=True)

    return fix_signature(gen_func, workingfunc)


def array_file_attr(name):
    """Construct a property that will handle getting setting ande deleting the name attribute."""

    def getter(self):
        owner = self.__dict__.get("_image_owner")
        if owner is not None:
            if name in {"shape", "dtype"}:
                return getattr(owner, name)
            if name == "size":
                return int(np.prod(owner.shape))
            if name == "ndim":
                return 2
        return getattr(self._image, name)

    def setter(self, value):
        if "_image_owner" in self.__dict__:
            from ..Image.storage_bridge import snapshot, replace_image
            if name in {"filename", "debug", "_title", "_mask_color", "_mask_alpha"}:
                self.__dict__["_image_attrs"][name] = value
                return
            draft = snapshot(self, writable=True)
            setattr(draft, name, value)
            replace_image(self, draft)
            return
        return setattr(self._image, name, value)

    def deleter(self):
        return delattr(self._image, name)

    return property(getter, setter, deleter, f"Pass thrpough for {name}")




def label(**kwargs):
    """A decoratory that adds attributes to a callable.

    Keyword Arguments:
        **kwargs:
            All keyword arguments are added to the functions __dict__.
    """

    def _decorator(func):
        """Actual decorator."""
        func.__dict__.update(kwargs)
        return func

    return _decorator


def class_modifier(
    module,
    adaptor=None,
    transpose=False,
    overload=False,
    proxy_cls=None,
    RTD_restrictions=True,  # pylint: disable=invalid-name
    no_long_names=False,
    alias=None,
):
    """Create a decorator that attaches functions from modules to a class.

    The purpose of this is to incorporate the functions within a module into being methods of the class being
    defined here.

    Args:
        module (module or iterable of modules):
            Source modules whose public functions should be attached.

    Keyword Arguments:
        adaptor (callable or None):
            Factory that wraps each source function, or None to attach it unchanged.
            None attaches the original function without adapting inputs or results.
        transpose (bool):
            Whether there functions in the module need to have their data transposed to work.
        overload (bool):
            If False, don't overwrite the existing method.'
        proxy_cls (class or None):
            If not None, the class whose attributes we are being augmented with these functions - need to check for
            clashing names.
        RTD_restrictions (bool):
            If True (default), do not add members from outside our own package when on ReadTheDocs.
        no_long_names (bool):
            To avoid name collision the default is to create two entries in the class __dict__ - one for the
            standard name and one to include the full module path. This disables the latter.
        alias (str or None):
            Regular expression used to select defining module names. None uses
            the source module's name as the prefix pattern.

    Returns:
        callable:
            A decorator that attaches the selected methods and returns the class.
    """

    def actual_decorator(cls):
        proxy_class = cls if proxy_cls is None else proxy_cls
        mods = module if isinstance(module, Iterable) else [module]
        for mod in mods:
            if alias is None:
                mod_name = mod.__name__.replace(".", r"\.")
                mod_name = "^" + mod_name
            else:
                mod_name = alias
            if (RTD_restrictions and _RTD) and not getattr(mod, "__package__", "Stoner").startswith("Stoner"):
                continue  # Do not bind all the external functions if we're in ReadTheDocs
            for fname in dir(mod):
                if not fname.startswith("_"):
                    try:
                        func = getattr(mod, fname)
                    except AttributeError:  # This shouldn't happen, but it did for scipy.ndimage!
                        continue
                    fmod = getattr(func, "__module__", getattr(getattr(func, "__class__", None), "__module__", ""))
                    if callable(func) and isinstance(fmod, str) and re.search(mod_name, fmod):
                        if transpose:
                            func.transpose = transpose
                        name = f"{fmod}__{fname}".replace(".", "__")
                        if adaptor is not None:
                            proxy = adaptor(func)
                        else:
                            proxy = func
                        setattr(proxy, "_src_mod", fmod)
                        if not no_long_names:
                            setattr(cls, name, proxy)
                        if overload or fname not in dir(proxy_class):
                            setattr(cls, fname, proxy)
        return cls

    return actual_decorator


def class_wrapper(
    target=None,
    adaptor=image_file_raw_adaptor,
    getter_adaptor=image_file_raw_adaptor,
    setter_adaptor=array_file_property,
    deleter_adaptor=array_file_property,
    attr_pass=array_file_attr,
    exclude_below=None,
):
    """Create entries in the current class for all attributes of klass that are not already defined.

    Keyword Arguments:
        target (type):
            The target class whose attributes we're going to link through to.
        adaptor (callable):
            A factory function to make methods to the the connection to the underlying attributes.

    Returns:
        class:
            Modified class definition.

    Notes:
        We exclude attributes with which have the attribute _src_mod as these are being patched already.
    """

    def actual_decorator(cls):
        for name in dir(target):
            if name.startswith("_"):
                continue
            attr = getattr(target, name)
            if callable(attr) and not isproperty(target, name) and name not in dir(cls):
                proxy = adaptor(attr)
                setattr(cls, name, proxy)
            elif (
                isproperty(target, name)
                and hasattr(attr, "fget")
                and (name not in dir(cls) or name in dir(exclude_below))
            ):
                fget = getter_adaptor(getattr(attr, "fget"))
                fset = setter_adaptor(getattr(attr, "fset", None))
                fdel = deleter_adaptor(getattr(attr, "fdel", None))
                doc = getattr(attr, "__doc__", "")
                setattr(cls, name, property(fget, fset, fdel, doc))
            elif name not in cls.__dict__ and not callable(attr) and not isproperty(target, name):
                setattr(cls, name, attr_pass(name))
        return cls

    return actual_decorator


def changes_size(func):
    """Mark a function as one that changes the size of the ImageFile."""
    func.changes_size = True
    return func


def keep_return_type(func):
    """Mark a function as one that Should not be converted from an array to an ImageFile."""
    func.keep_class = True
    return func


def clones(func):
    """Mark the method as one that expects it's input to be cloned."""
    func.clones = True
    return func


def fix_signature(proxy_func, wrapped_func):
    """Update proxy_func to have a signature that matches the wrapped func."""
    try:
        proxy_func.__wrapped__.__signature__ = inspect.signature(wrapped_func)
    except (AttributeError, ValueError):  # Non-critical error
        try:
            proxy_func.__signature__ = inspect.signature(wrapped_func)
        except (AttributeError, ValueError):
            pass
    if hasattr(wrapped_func, "changes_size"):
        proxy_func.changes_size = wrapped_func.changes_size
    return proxy_func


def make_Data(*args, **kwargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a special case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.core.data").Data
    return import_module("Stoner.core.data").Data(*args, **kwargs)


def make_Image(*args, **kwargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a special case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.Image.core").ImageFile
    return import_module("Stoner.Image.core").ImageFile(*args, **kwargs)


def make_Class(cls, *args, **kwargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a special case to return the Data class ratther than an instance
    """
    parts = cls.split(".")
    cls = parts.pop()
    mod = ".".join(["Stoner"] + parts)

    if len(args) == 1 and args[0] is None:
        return getattr(import_module(mod), cls)
    return getattr(import_module(mod), cls)(*args, **kwargs)
