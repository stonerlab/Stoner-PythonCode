# -*- coding: utf-8 -*-
"""Useful Utility classes."""

__all__ = [
    "attributeStore",
    "typedList",
    "Options",
    "get_option",
    "set_option",
    "itersubclasses",
    "subclasses",
    "copy_into",
]

import copy
from typing import Optional, Dict, Any, List, Iterable as IterableType, Union
from collections.abc import MutableSequence

from .tests import all_type, isiterable

_options = {
    "short_repr": False,
    "short_data_repr": False,
    "short_folder_rrepr": True,
    "short_img_repr": True,
    "no_figs": True,
    "multiprocessing": False,  # Change default to not use multiprocessing for now
    "threading": False,
}


class attributeStore(dict):

    """A dictionary=like class that provides attributes that work like indices.

    Used to implement the mapping of column types to indices in the setas attriobutes.
    """

    def __init__(self, *args: Any, **kargs: Any) -> None:
        """Initialise from a dictionary."""
        if len(args) == 1 and isinstance(args[0], dict):
            self.update(args[0])
        else:
            super().__init__(*args, **kargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute (equivalent to setting an item)."""
        self[name] = value

    def __getattr__(self, name: str) -> Any:
        """Get an attrbute (equivalent to getting an item)."""
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError from err


def itersubclasses(cls: type, _seen: Optional[set] = None) -> List[type]:
    """Iterate over subclasses of a given class.

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    if not isinstance(cls, type):
        raise TypeError(f"itersubclasses must be called with new-style classes, not {cls}")
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            itersubclasses(sub, _seen)
    return list(_seen)


def subclasses(cls: Optional[type] = None) -> Dict:  # pylint: disable=no-self-argument
    """Return a list of all in memory subclasses of this DataFile."""
    # pylint: disable=E1136, E1135
    if cls is None:
        from ..Core import DataFile  # pylint: disable=import-outside-toplevel

        cls = DataFile

    _subclasses = getattr(cls, "_subclasses", None)
    if _subclasses is None:
        _subclasses = dict()
    tmp = itersubclasses(cls)
    if len(_subclasses) < 1 or _subclasses[0] != len(tmp):  # Rebuild index
        tmp = {
            x: (getattr(x, "priority", 256), x.__name__)
            for x in sorted(tmp, key=lambda c: (getattr(c, "priority", 256), getattr(c, "__name__", "None")))
        }
        tmp = {v[1]: k for k, v in tmp.items()}
        ret = dict()
        ret[cls.__name__] = cls
        ret.update(tmp)
        _subclasses = dict()
        _subclasses = (len(tmp), ret)
    else:
        ret = dict(_subclasses[1])
    setattr(cls, "_subclasses", _subclasses)
    return ret


class typedList(MutableSequence):

    """Subclass list to make setitem enforce  strict typing of members of the list."""

    def __init__(self, *args: Any, **kargs: Any) -> None:
        """Construct the typedList."""
        self._store = []
        if (not args) or not (isinstance(args[0], type) or (isinstance(args[0], tuple) and all_type(args[0], type))):
            self._type = str  # Default list type is a string
        else:
            args = list(args)
            self._type = args.pop(0)
        if not args:
            self._store = list(*args, **kargs)
        elif len(args) == 1 and all_type(args[0], self._type):
            self._store = list(*args, **kargs)
        else:
            if len(args) > 1:
                raise SyntaxError("List should be constructed with at most two arguments, a type and an iterable")
            raise TypeError(f"List should be initialised with elements that are all of type {self._type}")

    def __add__(self, other: IterableType) -> "typedList":
        """Add operator works like ordinary lists."""
        if isiterable(other):
            new = copy.deepcopy(self)
            new.extend(other)
            return new
        return NotImplemented

    def __iadd__(self, other: IterableType) -> "typedList":
        """Inplace-add works like a list."""
        if isiterable(other):
            self.extend(other)
            return self
        return NotImplemented

    def __radd__(self, other: IterableType) -> "typedList":
        """Support add on the right like a list."""
        if isinstance(other, list):
            return other + self._store
        return NotImplemented

    def __eq__(self, other: List) -> bool:
        """Equality test."""
        return self._store == other

    def __delitem__(self, index: int) -> None:
        """Remove an item like in a list."""
        del self._store[index]

    def __getitem__(self, index: int) -> Any:
        """Get an item like in a list."""
        return self._store[index]

    def __len__(self) -> int:
        """Implement the len function like a list."""
        return len(self._store)

    def __repr__(self) -> str:
        """Textual representation like a list."""
        return repr(self._store)

    def __setitem__(self, name: Union[int, IterableType, slice], value: Any) -> None:
        """Sett an item and do some type checks."""
        if isiterable(name) or isinstance(name, slice):
            if not isiterable(value) or not all_type(value, self._type):
                raise TypeError(
                    f"Elelements of this list should be of type {self._type} and must set "
                    + "the correct number of elements"
                )
        elif not isinstance(value, self._type):
            raise TypeError(f"Elelements of this list should be of type {self._type}")
        self._store[name] = value

    def extend(self, other: IterableType) -> None:  # pylint:  disable=arguments-differ
        """Extend the list and do some type checking."""
        if not isiterable(other) or not all_type(other, self._type):
            raise TypeError(f"Elelements of this list should be of type {self._type}")
        self._store.extend(other)

    def index(  # pylint:  disable=arguments-differ
        self, search: Any, start: int = 0, end: Optional[int] = None
    ) -> int:
        """Index works like a list except we support Python 3 optional parameters everywhere."""
        if end is None:
            end = len(self._store)
        return self._store[start:end].index(search) + start

    def insert(self, index: int, obj: Any) -> None:  # pylint:  disable=arguments-differ
        """Insert an element and do some type checking."""
        if not isinstance(obj, self._type):
            raise TypeError(f"Elelements of this list should be of type {self._type}")
        self._store.insert(index, obj)


def get_option(name: str) -> bool:
    """Return the option value."""
    if name not in _options.keys():
        raise IndexError(f"{name} is not a valid package option")
    return _options[name]


def set_option(name: str, value: bool) -> None:
    """Set a global package option.

    Args:
        name (str):
            Option Name, one of:
                - short_repr (bool):
                    Instead of using a rich representation, use a short description for DataFile and Imagefile.
                - short_data_repr (bool):
                    Just use short representation for DataFiles
                - short_img_repr (bool):
                    Just use a short representation for image file
                - no_figs (bool):
                    Do not return figures from plotting functions, just plot them.
        value (depends on name):
            The value to set (see *name*)
    """
    if name not in _options.keys():
        raise IndexError(f"{name} is not a valid package option")
    if not isinstance(value, bool):
        raise ValueError(f"{name} takes a boolean value not a {type(value)}")
    _options[name] = value


class Options:

    """Dead simple class to allow access to package options."""

    def __init__(self):
        """Options class wraps the get/set_option calls into an object orientated interface."""
        self._defaults = copy.copy(_options)

    def __setattr__(self, name: str, value: bool) -> None:
        """Set an option value."""
        if name.startswith("_"):
            return super().__setattr__(name, value)
        if name not in _options:
            raise AttributeError(f"{name} is not a recognised option.")
        if not isinstance(value, type(_options[name])):
            raise ValueError(f"{name} takes a {type(_options[name])} not a {type(value)}")
        set_option(name, value)

    def __getattr__(self, name: str) -> bool:
        """Lookup an option value."""
        if name not in _options:
            raise AttributeError(f"{name} is not a recognised option.")
        return get_option(name)

    def __delattr__(self, name: str) -> None:
        """Clear and Option value back to defaults."""
        if name not in _options:
            raise AttributeError(f"{name} is not a recognised option.")
        set_option(name, self._defaults[name])

    def __dir__(self) -> List[str]:
        """Return a list of the aviailable options."""
        return list(_options.keys())

    def __repr__(self) -> str:
        """Make a standard text representation of Options class."""
        s = "Stoner Package Options\n"
        s += "~~~~~~~~~~~~~~~~~~~~~~\n"
        for k in dir(self):
            s += f"{k} : {get_option(k)}\n"
        return s


def copy_into(source: "DataFile", dest: "DataFile") -> "DataFile":
    """Copy the data associated with source to dest.

    Args:
        source(DataFile): The DataFile object to be copied from
        dest (DataFile): The DataFile objrct to be changed by recieving the copiued data.

    Returns:
        The modified *dest* DataFile.

    Unlike copying or deepcopying a DataFile, this function preserves the class of the destination and just
    overwrites the attributes that represent the data in the DataFile.
    """
    dest.data = source.data.copy()
    dest.setas = source.setas
    for attr in source._public_attrs:
        if not hasattr(source, attr) or callable(getattr(source, attr)) or attr in ["data"]:
            continue
        try:
            setattr(dest, attr, copy.deepcopy(getattr(source, attr)))
        except (NotImplementedError, TypeError, ValueError):  # Deepcopying failed, so just copy a reference instead
            try:
                setattr(dest, attr, getattr(source, attr))
            except ValueError:
                pass
    dest._punlic_attrs = source._public_attrs
    return dest
