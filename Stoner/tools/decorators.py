# -*- coding: utf-8 -*-
"""Provide some decorators and associated functions for modifying package classes and functions."""

from functools import wraps
import inspect
from importlib import import_module
from collections.abc import Iterable

import numpy as np

try:
    from memoization import cached
except ImportError:

    def cached(func, *_):
        """Null dectorator."""
        return func


def image_array_adaptor(workingfunc):
    """Wrap an arbitary callbable to make it a bound method of this class.

        Args:
            workingfunc (callable):
                The callable object to be wrapped.

        Returns:
            (function):
                A function with enclosure that holds additional information about this object.

        The function returned from here will call workingfunc with the first argument being a clone of this
        ImageArray. If the meothd returns an ndarray, it is wrapped back to our own class and the metadata dictionary
        is updated. If the function returns a :py:class:`Stoner.Data` object then this is also updated with our
        metadata.

        This method also updates the name and documentation strings for the wrapper to match the wrapped function -
        thus ensuring that Spyder's help window can generate useful information.

        """
    # Avoid PEP257/black issue

    @wraps(workingfunc)
    def gen_func(self, *args, **kwargs):
        """Wrap magic proxy function call."""
        transpose = getattr(workingfunc, "transpose", False)
        if transpose:
            change = self.T
        else:
            change = self
        r = workingfunc(change, *args, **kwargs)  # send copy of self as the first arg
        if isinstance(r, make_Data(None)):
            pass  # Data return is ok
        elif isinstance(r, np.ndarray) and np.prod(r.shape) == np.max(r.shape):  # 1D Array
            r = make_Data(r)
            r.metadata = self.metadata.copy()
            r.column_headers[0] = workingfunc.__name__
        elif isinstance(r, np.ndarray):  # make sure we return a ImageArray
            if transpose:
                r = r.T
            if isinstance(r, self.__class__) and np.shares_memory(r, self):  # Assume everything was inplace
                return r
            r = r.view(self.__class__)
            sm = self.metadata.copy()  # Copy the currenty metadata
            sm.update(r.metadata)  # merge in any new metadata from the call
            r.metadata = sm  # and put the returned metadata as the merged data
        # NB we might not be returning an ndarray at all here !
        return r

    return fix_signature(gen_func, workingfunc)


def class_modifier(module, adaptor=image_array_adaptor, transpose=False, overload=False):
    """Decorate  a class by addiding member functions from module.

    The purpose of this is to incorporate the functions within a module into being methods of the class being
    defined here.

    Args:
        cls (class):
            The class being defined
        module (imported module):
            The module whose functions members should be added to the class.

    Keyword Arguments:
        adaptor (callable):
            The factor function that takes the module function and produces a method that will call the function and
            take care of adapting the result.
        transpose (bool):
            Whether ther functions in the module need to have their data transposed to work.
        overload (bool):
            If False, don't overwrite the existing method.'

    Returns:
        (class):
            The class with the additional methods added to it.
    """

    def actual_decorator(cls):
        mods = module if isinstance(module, Iterable) else [module]
        for mod in mods:
            for fname in dir(mod):
                if not fname.startswith("_"):
                    func = getattr(mod, fname)
                    fmod = getattr(func, "__module__", getattr(getattr(func, "__class__", None), "__module__", ""))
                    if callable(func) and fmod[:5] in ["Stone", "scipy", "skima"]:
                        if transpose:
                            func.transpose = transpose
                        name = f"{fmod}__{fname}".replace(".", "__")
                        setattr(cls, name, adaptor(func))
                        if overload or fname not in dir(cls):
                            setattr(cls, fname, adaptor(func))
        return cls

    return actual_decorator


def changes_size(func):
    """Mark a function as one that changes the size of the ImageArray."""
    func.changes_size = True
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


@cached
def make_Data(*args, **kargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a speical case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.core.data").Data
    return import_module("Stoner.core.data").Data(*args, **kargs)
