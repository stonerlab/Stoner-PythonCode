# -*- coding: utf-8 -*-
"""Provide some decorators and associated functions for modifying package classes and functions."""

from functools import wraps
import inspect
from importlib import import_module
from collections.abc import Iterable
from copy import copy
from os import environ

import numpy as np

from .tests import isproperty

_RTD = "READTHEDOCS" in environ


def image_file_adaptor(workingfunc):
    """Make wrappers for ImageFile functions.

    Notes:
        The wrapped functions take additional keyword arguments that are stripped off from the call.

    Keyword Arguments:
        _box(:py:meth:`Stoner.ImageArray.crop` arguments):
            Crops the image first before calling the parent method.
        _(bool, None):
            Controls whether a :py:class:`ImageArray` return will be substituted for the current
            :py:class:`ImageArray`.

            * True: - all ImageArray return types are substituted.
            * False (default) - Imagearray return types are substituted if they are the same size as the original
            * None - A copy of the current object is taken and the returned ImageArray provides the data.
    """
    # Avoid PEP257/black issue

    @wraps(workingfunc)
    def gen_func(self, *args, **kargs):
        """Wrap a called method to capture the result back into the calling object."""
        box = kargs.pop("_box", False)
        transpose = getattr(workingfunc, "transpose", False)
        if isinstance(box, bool) and not box:
            im = self.image
        else:
            im = self.image[im._box(box)]
        if transpose:
            im = im.T
        args = list(args)
        for ix, a in enumerate(args):
            if isinstance(a, type(self)):
                args[ix] = a.image

        if getattr(workingfunc, "changes_size", False) and "_" not in kargs:
            # special case for common function crop which will change the array shape
            force = True
        else:
            force = kargs.pop("_", False)
        r = workingfunc(im, *args, **kargs)
        if getattr(workingfunc, "keep_class", False):
            return r
        if isinstance(r, type(self.image)):
            self.image = r
            return self
        if isinstance(r, np.ndarray) and np.prod(r.shape) == np.max(r.shape):  # 1D Array
            ret = make_Data(r)
            ret.metadata = self.metadata.copy()
            ret.column_headers[0] = workingfunc.__name__
        elif isinstance(r, np.ndarray):  # make sure we return a ImageArray
            if transpose:
                r = r.T
            if isinstance(r, type(im)) and np.shares_memory(r, im):  # Assume everything was inplace
                self.image = r

                return self
            r = r.view(type(im))
            if r.shape == self.shape:
                if im.metadata is not r.metadata:
                    self.image = r.clone  # We're going to need to clone the return data because we can do fast copies.
                self.image[...] = r[...]
                self.metadata.update(r.metadata)
                return self
            ret = self.clone if not force else self
            ret.image = r.view(type(im))
            metadata = copy(self.metadata)
            metadata.update(r.metadata)
            ret.metadata = metadata
            return ret
        else:
            return r

    return fix_signature(gen_func, workingfunc)


def image_file_raw_adaptor(workingfunc):
    """Make wrappers for ImageFile functions.

    Notes:
        The wrapped functions take additional keyword arguments that are stripped off from the call.

    Keyword Arguments:
        _box(:py:meth:`Stoner.ImageArray.crop` arguments):
            Crops the image first before calling the parent method.
        _(bool, None):
            Controls whether a :py:class:`ImageArray` return will be substituted for the current
            :py:class:`ImageArray`.

            * True: - all ImageArray return types are substituted.
            * False (default) - Imagearray return types are substituted if they are the same size as the original
            * None - A copy of the current object is taken and the returned ImageArray provides the data.
    """
    # Avoid PEP257/black issue

    @wraps(workingfunc)
    def gen_func(self, *args, **kargs):
        """Wrap a called method to capture the result back into the calling object."""
        box = kargs.pop("_box", False)
        transpose = getattr(workingfunc, "transpose", False)
        clones = getattr(workingfunc, "clones", False)
        if isinstance(box, bool) and not box:
            im = self.image
        else:
            im = self.image[im._box(box)]
        if transpose:
            im = im.T
        args = list(args)
        for ix, a in enumerate(args):
            if isinstance(a, type(self)):
                args[ix] = a.image

        if getattr(workingfunc, "changes_size", False) and "_" not in kargs:
            # special case for common function crop which will change the array shape
            force = True
        else:
            force = kargs.pop("_", False)
        r = workingfunc(im, *args, **kargs)
        if getattr(workingfunc, "keep_class", False):
            return r
        if isinstance(r, np.ndarray) and r.ndim != 2:  # 1D Array goes back straight
            return r
        if isinstance(r, np.ndarray):  # make sure we return a ImageArray
            if transpose:
                r = r.T
            ret = self if not clones else self.clone
            if isinstance(r, type(im)) and np.shares_memory(r, im):  # Assume everything was inplace
                ret.image = r
                return ret
            r = r.view(type(im))
            if r.shape == ret.shape:
                ret.image = ret.image.astype(r.dtype)
                ret.image[...] = r[...]
                ret.metadata.update(r.metadata)
                return ret
            ret = self.clone if not force else self
            ret.image = r.view(type(im))
            metadata = copy(ret.metadata)
            metadata.update(r.metadata)
            ret.metadata = metadata
            return ret
        return r

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
    def gen_func(self, *args, **kargs):
        """Wrap magic proxy function call."""
        transpose = getattr(workingfunc, "transpose", False)
        clones = getattr(workingfunc, "clones", False)
        im = self.image if not clones else self.clone.image
        if transpose:
            im = im.T
        args = list(args)
        for ix, a in enumerate(args):
            if isinstance(a, type(self)):
                args[ix] = a.image

        ret = workingfunc(im, *args, **kargs)
        # This shouldn't in fact be returning anything
        return ret

    return fix_signature(gen_func, workingfunc)


def array_file_attr(name):
    """Construct a property that will handle getting setting ande deleting the name attribute."""

    def getter(self):
        return getattr(self._image, name)

    def setter(self, value):
        return setattr(self._image, name, value)

    def deleter(self):
        return delattr(self._image, name)

    return property(getter, setter, deleter, f"Pass thrpough for {name}")


def image_array_adaptor(workingfunc):
    """Wrap an arbitrary callbable to make it a bound method of this class.

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
            if isinstance(r, type(self)) and np.shares_memory(r, self):  # Assume everything was inplace
                return r
            r = r.view(type(self))
            sm = self.metadata.copy()  # Copy the currently metadata
            sm.update(r.metadata)  # merge in any new metadata from the call
            r.metadata = sm  # and put the returned metadata as the merged data
        # NB we might not be returning an ndarray at all here !
        return r

    gen_func.keep_class = getattr(workingfunc, "keep_class", False)
    return fix_signature(gen_func, workingfunc)


def class_modifier(
    module,
    adaptor=image_array_adaptor,
    transpose=False,
    overload=False,
    proxy_cls=None,
    RTD_restrictions=True,
    no_long_names=False,
):
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
        proxy (class,None):
            If not None, the class whose attributes we are being augmented with these functions - need to check for
            clashing names.
        RTD_Restrictions (bool):
            If True (default), do not add members from outside our own package when on ReadTheDocs.
        no_long_names (bool):
            To avoid name collision the default is to create two entries in the class __dict__ - one for the standard name
            and one to include the full module path. This disables the latter.


    Returns:
        (class):
            The class with the additional methods added to it.
    """

    def actual_decorator(cls):
        proxy_class = cls if proxy_cls is None else proxy_cls
        mods = module if isinstance(module, Iterable) else [module]
        for mod in mods:
            if (RTD_restrictions and _RTD) and not getattr(mod, "__package__", "Stoner").startswith("Stoner"):
                continue  # Do not bind all the external functions if we're in ReadTheDocs
            for fname in dir(mod):
                if not fname.startswith("_"):
                    try:
                        func = getattr(mod, fname)
                    except AttributeError:  # This shouldn't happen, but it did for scipy.ndimage!
                        continue
                    fmod = getattr(func, "__module__", getattr(getattr(func, "__class__", None), "__module__", ""))
                    if callable(func) and isinstance(fmod, str) and fmod[:5] in ["Stone", "scipy", "skima"]:
                        if transpose:
                            func.transpose = transpose
                        name = f"{fmod}__{fname}".replace(".", "__")
                        proxy = adaptor(func)
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

    Note:
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
    """Mark a function as one that changes the size of the ImageArray."""
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


def make_Data(*args, **kargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a special case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.core.data").Data
    return import_module("Stoner.core.data").Data(*args, **kargs)


def make_Image(*args, **kargs):
    """Return an instance of Stoner.Data passig through constructor arguments.

    Calling make_Data(None) is a special case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.Image.core").ImageFile
    return import_module("Stoner.Image.core").ImageFile(*args, **kargs)
