# -*- coding: utf-8 -*-
"""Useful Utility classes"""

__all__ = ["attributeStore", "typedList", "Options", "get_option", "set_option"]

import copy
from collections.abc import MutableSequence

from .tests import all_type, isIterable

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

    def __init__(self, *args, **kargs):
        """Initialise from a dictionary."""
        if len(args) == 1 and isinstance(args[0], dict):
            self.update(args[0])
        else:
            super(attributeStore, self).__init__(*args, **kargs)

    def __setattr__(self, name, value):
        """Setting an attribute is equivalent to setting an item."""
        self[name] = value

    def __getattr__(self, name):
        """Getting an attrbute is equivalent to getting an item."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError


class typedList(MutableSequence):

    """Subclass list to make setitem enforce  strict typing of members of the list."""

    def __init__(self, *args, **kargs):
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
            raise TypeError(
                "List should be initialised with elements that are all of type {} not {}".format(
                    self._type, args[0].dtype
                )
            )

    def __add__(self, other):
        """Add operator works like ordinary lists."""
        if isIterable(other):
            new = copy.deepcopy(self)
            new.extend(other)
            return new
        return NotImplemented

    def __iadd__(self, other):
        """Inplace-add works like a list."""
        if isIterable(other):
            self.extend(other)
            return self
        return NotImplemented

    def __radd__(self, other):
        """Support add on the right like a list."""
        if isinstance(other, list):
            return other + self._store
        return NotImplemented

    def __eq__(self, other):
        """Equality test."""
        return self._store == other

    def __delitem__(self, index):
        """Remove an item like in a list."""
        del self._store[index]

    def __getitem__(self, index):
        """Get an item like in a list."""
        return self._store[index]

    def __len__(self):
        """Implement the len function like a list."""
        return len(self._store)

    def __repr__(self):
        """Textual representation like a list."""
        return repr(self._store)

    def __setitem__(self, name, value):
        """Setting an item requires some type checks."""
        if isIterable(name) or isinstance(name, slice):
            if not isIterable(value) or not all_type(value, self._type):
                raise TypeError(
                    "Elelements of this list should be of type {} and must set the correct number of elements".format(
                        self._type
                    )
                )
        elif not isinstance(value, self._type):
            raise TypeError("Elelements of this list should be of type {}".format(self._type))
        self._store[name] = value

    def extend(self, other):  # pylint:  disable=arguments-differ
        """Extending a list also requires some type checking."""
        if not isIterable(other) or not all_type(other, self._type):
            raise TypeError("Elelements of this list should be of type {}".format(self._type))
        self._store.extend(other)

    def index(self, search, start=0, end=None):  # pylint:  disable=arguments-differ
        """Index works like a list except we support Python 3 optional parameters everywhere."""
        if end is None:
            end = len(self._store)
        return self._store[start:end].index(search) + start

    def insert(self, index, obj):  # pylint:  disable=arguments-differ
        """Inserting an element also requires some type checking."""
        if not isinstance(obj, self._type):
            raise TypeError("Elelements of this list should be of type {}".format(self._type))
        self._store.insert(index, obj)


def get_option(name):
    """Return the option value"""
    if name not in _options.keys():
        raise IndexError("{} is not a valid package option".format(name))
    return _options[name]


def set_option(name, value):
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
        raise IndexError("{} is not a valid package option".format(name))
    if not isinstance(value, bool):
        raise ValueError("{} takes a boolean value not a {}".format(name, type(value)))
    _options[name] = value


class Options:

    """Dead simple class to allow access to package options."""

    def __init__(self):
        self._defaults = copy.copy(_options)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return super(Options, self).__setattr__(name, value)
        if name not in _options:
            raise AttributeError("{} is not a recognised option.".format(name))
        if not isinstance(value, type(_options[name])):
            raise ValueError("{} takes a {} not a {}".format(name, type(_options[name]), type(value)))
        set_option(name, value)

    def __getattr__(self, name):
        if name not in _options:
            raise AttributeError("{} is not a recognised option.".format(name))
        return get_option(name)

    def __delattr__(self, name):
        if name not in _options:
            raise AttributeError("{} is not a recognised option.".format(name))
        set_option(name, self.defaults[name])

    def __dir__(self):
        return list(_options.keys())

    def __repr__(self):
        s = "Stoner Package Options\n"
        s += "~~~~~~~~~~~~~~~~~~~~~~\n"
        for k in dir(self):
            s += "{} : {}\n".format(k, get_option(k))
        return s
