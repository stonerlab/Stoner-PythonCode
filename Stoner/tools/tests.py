# -*- coding: utf-8 -*-
"""Utility functions that test whether things are true or not."""

__all__ = [
    "all_size",
    "all_type",
    "isanynone",
    "isComparable",
    "isiterable",
    "isLikeList",
    "isnone",
    "isproperty",
    "isTuple",
]
from typing import Optional, Iterable as IterableType, Tuple, Union, Any

from collections.abc import Iterable
from importlib import import_module

from numpy import ndarray, dtype, isnan, logical_and  # pylint: disable=redefined-builtin

from ..compat import string_types
from ..core.Typing import NumericArray


def all_size(iterator: IterableType, size: Optional[Union[int, Tuple]] = None) -> bool:
    """Check whether each element of *iterator* is the same length/shape.

    Arguments:
        iterator (Iterable): list or other iterable of things with a length or shape

    Keyword Arguments:
        size(int, tuple or None): Required size of each item in iterator.

    Returns:
        True if all objects are the size specified (or the same size if size is None).
    """
    if hasattr(iterator[0], "shape"):
        sizer = lambda x: x.shape
    else:
        sizer = len

    if size is None:
        size = sizer(iterator[0])
    ret = False
    for i in iterator:
        if sizer(i) != size:
            break
    else:
        ret = True
    return ret


def all_type(iterator: IterableType, typ: type) -> bool:
    """Determine if an interable omnly contains a common type.

    Arguments:
        iterator (Iterable):
            The object to check if it is all iterable
        typ (class):
            The type to check for.

    Returns:
        True if all elements are of the type typ, or False if not.

    Notes:
        Routine will iterate the *iterator* and break when an element is not of
        the search type *typ*.
    """
    ret = False
    if isinstance(iterator, ndarray):  # Try to short circuit for arrays
        try:
            return iterator.dtype == dtype(typ)
        except TypeError:
            pass
    if isiterable(iterator):
        for i in iterator:
            if not isinstance(i, typ):
                break
        else:
            ret = True
    return ret


def isanynone(*args: Any) -> bool:
    """Intelligently check whether any of the inputs are None."""
    for arg in args:
        if arg is None:
            return True
    return False


def isComparable(v1: NumericArray, v2: NumericArray) -> bool:
    """Return true if v1 and v2 can be compared sensibly.

    Args:
        v1,v2 (any):
            Two values to compare

    Returns:
        False if both v1 and v2 are numerical and both nan, otherwise True.
    """
    try:
        return not (isnan(v1) and isnan(v2))
    except TypeError:
        return True
    except ValueError:
        try:
            return not logical_and(isnan(v1), isnan(v2)).any()
        except TypeError:
            return False


def isiterable(value: Any) -> bool:
    """Chack to see if a value is iterable.

    Args:
        value :
            Entity to check if it is iterable

    Returns:
        (bool):
            True if value is an instance of collections.Iterable.
    """
    return isinstance(value, Iterable)


def isLikeList(value: Any) -> bool:
    """Return True if value is an iterable but not a string."""
    return isiterable(value) and not isinstance(value, string_types)


def isnone(iterator: Optional[IterableType]) -> bool:
    """Return True if input is None or an empty iterator, or an iterator of None.

    Args:
        iterator (None or Iterable):

    Returns:
        True if iterator is None, empty or full of None.
    """
    if iterator is None:
        ret = True
    elif isiterable(iterator) and not isinstance(iterator, string_types):
        try:
            l = len(iterator)
        except TypeError:
            l = 0
        if l == 0:  # pylint: disable=len-as-condition
            ret = True
        else:
            for i in iterator:
                if i is not None:
                    ret = False
                    break
            else:
                ret = True
    else:
        ret = False
    return ret


def isproperty(obj: Any, name: str) -> bool:
    """Check whether an attribute of an object or class is a property.

    Args:
        obj (instance or class):
            Thing that has the attribute to check
        name (str):
            Name of the attribute that might be a property

    Returns:
        (bool):
            Whether the name is a property or not.
    """
    if not isinstance(obj, type):
        obj = type(obj)
    elif not issubclass(obj, object):
        raise TypeError(f"Can only check for property status on attributes of an object or a class not a {type(obj)}")
    return hasattr(obj, name) and isinstance(getattr(obj, name), property)


def isTuple(obj: Any, *args: type, strict: bool = True) -> bool:
    """Determine if obj is a tuple of a certain signature.

    Args:
        obj(object):
            The object to check
        *args(type):
            Each of the succeeding arguments are used to determine the expected type of each element.

    Keywoprd Arguments:
        strict(bool):
            Whether the elements of the tuple have to be exactly the type specified or just castable as the type

    Returns:
        (bool):
            True if obj is a matching tuple.
    """
    if not isinstance(obj, tuple):
        return False
    if args and len(obj) != len(args):
        return False
    for t, e in zip(args, obj):
        if strict:
            if not isinstance(e, t):
                bad = True
                break
        else:
            try:
                _ = t(e)
            except ValueError:
                bad = True
                break
    else:
        bad = False
    return not bad


class ClassTester:
    """Dynamically load classes on attribute access for structural pattern matching."""

    def __init__(self, **kargs):
        """Store a mapping of attribute name to a string of dot notation classes."""
        self._kargs = kargs

    def __call__(self, **kargs):
        """Update the mapping of attribute names and class mappings."""
        self._kargs |= kargs

    def __getattr__(self, name):
        """Lookup an attribute name in the stored name-class name mapping and return it.

        Args:
            name (str):
                Attribute name to lookup.

        Returns:
            (type):
                Class type matchignt he attribute.

        If the mapping contains a string then it is split into class and module. If the module exists in
        sys.modules then just get the module from there, otherwise, load it with importlib machinery. Finally
        get the class as an attribute in the module and return it. Also, set the type into the stored mapping.
        """
        if name not in self._kargs:
            return AttributeError(f"{name} is not an attrbute or mapped class alias.")
        mod = self._kargs[name]
        if isinstance(mod, str):
            parts = mod.split(".")
            cls = parts.pop()
            mod = ".".join(parts)
            mod = import_module(mod)
            self._kargs[name] = getattr(mod, cls)
        return self._kargs[name]
