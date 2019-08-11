#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setas module provides the setas class for DataFile and friends
"""
__all__ = ["setas"]
import re
import copy
import numpy as _np_

from ..compat import string_types, int_types, index_types, _pattern_type
from ..tools import _attribute_store, isiterable, typedList, islike_list
from .utils import decode_string

from collections import MutableMapping, Mapping


class setas(MutableMapping):

    """A Class that provides a mechanism for managing the column assignments in a DataFile like object.

    Implements a MutableMapping bsed on the column_headers as the keys (with a few tweaks!).

    Note:
        Iterating over setas will return the column assignments rather than the standard mapping behaviour of iterating over the keys. Otherwise
        the interface is essentially as a Mapping class.

    Calling an existing setas instance and the constructor share the same signatgure:

    setas("xyzuvw")
    setas(["x"],["y"],["z"],["u"],["v"],["w"])
    setas(x="column_1",y=3,column4="z")

    Keyword Arguments:
        _self (bool):
            If True, make the call return a copy of the setas object, if False, return _object attribute, if None,
            return None
        reset (bool):
            If False then preserve the existing set columns and simply add the new ones. Otherwise, clear all column
            assignments before setting new ones (default).
    """

    def __init__(self, row=False, bless=None):
        """Constructs the setas instance and sets an initial value.

        Args:
            ref (DataFile): Contains a reference to the owning DataFile instance

        Keyword Arguments:
            initial_val (string or list or dict): Initial values to set
        """
        self._row = row
        self._cols = _attribute_store()
        self._shape = tuple()
        self._setas = list()
        self._column_headers = typedList(string_types)
        self._object = bless
        self._col_defaults = {
            2: {
                "axes": 2,
                "xcol": 0,
                "ycol": [1],
                "zcol": [],
                "ucol": [],
                "vcol": [],
                "wcol": [],
                "xerr": None,
                "yerr": [],
                "zerr": [],
            },  # xy
            3: {
                "axes": 2,
                "xcol": 0,
                "ycol": [1],
                "zcol": [],
                "ucol": [],
                "vcol": [],
                "wcol": [],
                "xerr": None,
                "yerr": [2],
                "zerr": [],
            },  # xye
            4: {
                "axes": 2,
                "xcol": 0,
                "ycol": [2],
                "zcol": [],
                "ucol": [],
                "vcol": [],
                "wcol": [],
                "xerr": 1,
                "yerr": [3],
                "zerr": [],
            },  # xdye
            5: {
                "axes": 5,
                "xcol": 0,
                "ycol": [1],
                "zcol": None,
                "ucol": [2],
                "vcol": [3],
                "wcol": [4],
                "xerr": None,
                "yerr": [],
                "zerr": [],
            },  # xyuvw
            6: {
                "axes": 6,
                "xcol": 0,
                "ycol": [1],
                "zcol": [2],
                "ucol": [3],
                "vcol": [4],
                "wcol": [5],
                "xerr": None,
                "yerr": [],
                "zerr": [],
            },
        }  # xyzuvw

    def _prepare_call(self, args, kargs):
        """Extract a value to be used to evaluate the setas attribute during a call."""
        reset = kargs.pop("reset", True)
        if not isinstance(reset, bool):
            reset = True

        if args:
            value = args[0]
            if isinstance(value, string_types):  # expand the number-code combos in value
                if reset:
                    self.setas = []
                value = decode_string(value)
            elif isinstance(value, setas):
                if value is not self:
                    value = value.setas
                else:
                    value = self._setas
        else:
            value = kargs
            if reset:
                self.setas = []
        return value

    @property
    def _size(self):
        """Calculate a size of the setas attribute."""
        if len(self._shape) == 1 and self._row:
            c = self._shape[0]
        elif len(self._shape) == 1:
            c = 1
        elif len(self._shape) > 1:
            c = self.shape[1]
        else:
            c = len(self._column_headers)
        return c

    @property
    def _unique_headers(self):
        """Return either a column header or an index if the column_header is duplicated."""
        ret = []
        for i, ch in enumerate(self.column_headers):
            if ch not in ret:
                ret.append(ch)
            else:
                ret.append(i)
        return ret

    @property
    def clone(self):
        """Create an exact copy of the current object."""
        cls = self.__class__
        new = cls()
        for attr in self.__dict__:
            if not callable(self.__dict__[attr]):
                new.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return new

    @property
    def cols(self):
        """Get the current column assignments."""
        self._cols.update(self._get_cols())
        return self._cols

    @property
    def x(self):
        """Quick access to the x column number
        Just a convenience read only property. If we want to change the setas.x
        value we should use the setas(x=1,y=2) style call (so that reset can
        be handled properly)
        """
        return self.cols["xcol"]

    @property
    def y(self):
        """Quick access to the y column numbers list"""
        return self.cols["ycol"]

    @property
    def z(self):
        """Quick access to the z column numbers list"""
        return self.cols["zcol"]

    @property
    def column_headers(self):
        """Get the current column headers."""
        c = self._size
        l = len(self._column_headers)
        if l < c:  # Extend the column headers if necessary
            self._column_headers.extend(["Column {}".format(i + l) for i in range(c - l)])
        return self._column_headers

    @column_headers.setter
    def column_headers(self, value):
        """Set the colum headers."""
        if isinstance(value, _np_.ndarray):  # Convert ndarray to list of strings
            value = value.astype(str).tolist()
        elif isinstance(value, string_types):  # Bare strings get turned into lists
            value = [value]
        self._column_headers = typedList(string_types, value)

    @property
    def not_set(self):
        """Return a boolean array if not set."""
        return _np_.array([x == "." for x in self._setas])

    @property
    def set(self):
        """Return a boolean array if column is set."""
        return ~self.not_set

    @property
    def setas(self):
        """Guard the setas attribute."""
        c = self._size
        l = len(self._setas)
        if c > l:
            self._setas.extend(["."] * (c - l))
        self._setas = self._setas[:c]
        return self._setas

    @setas.setter
    def setas(self, value):
        """Minimal attribute setter."""
        self._setas = value

    @property
    def shape(self):
        """Return the shape of the array that we think we are."""
        return self._shape

    @shape.setter
    def shape(self, value):
        """Update the note of our shape."""
        value = tuple(value)
        if 0 <= len(value) <= 2:
            self._shape = tuple(value)
        else:
            raise AttributeError("shape attribute should be a 2-tuple not a {}-tuple".format(len(value)))

    def __call__(self, *args, **kargs):
        """Treat the current instance as a callable object and assign columns accordingly.

        Variois forms of this method are accepted::

        setas("xyzuvw")
        setas(["x"],["y"],["z"],["u"],["v"],["w"])
        setas(x="column_1",y=3,column4="z")

        Keyword Arguments:
            _self (bool):
                If True, make the call return a copy of the setas object, if False, return _object attribute, if None,
                return None
            reset (bool):
                If False then preserve the existing set columns and simply add the new ones. Otherwise, clear all column
                assignments before setting new ones (default).
        """
        return_self = kargs.pop("_self", None)
        if not (args or kargs):  # New - bare call to setas will return the current value.
            return self.setas
        if len(args) == 1 and isinstance(args[0], setas):
            args = list(args)
            args[0] = args[0].to_list()
        if len(args) == 1 and not (isinstance(args[0], string_types + (setas,)) or isiterable(args[0])):
            raise SyntaxError(
                "setas should be called with eother a string, iterable object or setas object, not a {}".format(
                    type(args[0])
                )
            )

        # If reset is neither in kargs nor a False boolean, then clear the existing setas assignments
        value = self._prepare_call(args, kargs)
        _ = self.setas  # Forxce setas to be the right length
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, string_types) and len(k) == 1 and k in "xyzuvwdef":  # of the form x:column_name
                    for v_item in self.find_col(v, force_list=True):
                        try:
                            self._setas[v_item] = k
                        except (IndexError, KeyError):
                            pass
                elif (
                    isinstance(k, index_types) and isinstance(v, string_types) and len(v) == 1 and v in "xyzuvwdef"
                ):  # of the form column_name:x
                    k = self.find_col(k)
                    self._setas[k] = v
                else:
                    raise IndexError(
                        "Unable to workout what do with {}:{} when setting the setas attribute.".format(k, v)
                    )
        elif isiterable(value):
            if len(value) > self._size:
                value = value[: self._size]
            elif len(value) < self._size:
                value = [v for v in value]  # Ensure value is now a list
                value.extend(list("." * (self._size - len(value))))
            value = value[: self._size]
            for i, v in enumerate(value):
                if v.lower() not in "xyzedfuvw.-":
                    raise ValueError("Set as column element is invalid: {}".format(v))
                if v != "-":
                    self.setas[i] = v.lower()
        else:
            raise ValueError("Set as column string ended with a number")
        self.cols.update(self._get_cols())
        if return_self is None:
            return None
        elif return_self:
            return self
        else:
            return self._object

    def __contains__(self, item):
        """Use getitem to test for membership. Either column assignments or column index types are tested."""
        try:
            _ = self[item]
        except (IndexError, KeyError, ValueError):
            return False
        return True

    def __delitem__(self, name):
        """Unset either by column index or column assignment.

        Equivalent to unsetting the same object."""
        self.unset(name)

    def __eq__(self, other):
        """Checks to see if this is the same object, or has the same headers and the same setas values."""
        ret = False
        if isinstance(other, string_types):  # Expand strings and convert to list
            other = [c for c in decode_string(other)]
        if not isinstance(other, setas):  # Ok, need to check whether items match
            if isiterable(other) and len(other) <= self._size:
                for m in self.setas[len(other) :]:  # Check that if other is short we don't have assignments there
                    if m != ".":
                        return False
                for o, m in zip(other, self.setas):
                    if o != m:  # Look for mis-matched assignments
                        return False
                return True
            else:  # If other is longer then we can't matchj
                return False
        elif id(self) == id(other):
            ret = True
        else:
            ret = self.column_headers == other.column_headers and self.setas == other.setas
        return ret

    def __getattr__(self, name):
        """Try to see if attribute name is a key in self.cols and return that instead."""
        if name != "_cols" and name in self._cols:
            return self._cols[name]
        return getattr(super(setas, self), name)

    def __getitem__(self, name):
        """Permit the setas attribute to be treated like either a list or a dictionary.

        Args:
            name (int, slice or string): if *name* is an integer or a slice, return the column type
                of the corresponding column(s). If a string, should be a single letter
                from the set x,y,z,u,v,w,d,e,f - if so returns the corresponding
                column(s)

        Returns:
            Either a single letter x,y,z,u,v,w,d,e or f, or a list of letters if used in
            list mode, or a single coliumn name or list of names if used in dictionary mode.
        """
        if isinstance(name, string_types) and len(name) == 1 and name in "xyzuvwdef.-":
            ret = self.to_dict()[name]
            if len(ret) == 1:
                ret = ret[0]
        elif isinstance(name, string_types) and len(name) == 2 and name[0] == "#" and name[1] in "xyzuvwdef.-":
            ret = list()
            name = name[1]
            s = 0
            while name in self._setas[s:]:
                s = self._setas.index(name) + 1
                ret.append(s - 1)
            if len(ret) == 1:
                ret = ret[0]
        elif isinstance(name, index_types):
            ret = self.setas[self.find_col(name)]
        elif isinstance(name, slice):
            indices = name.indices(len(self.setas))
            name = range(*indices)
            ret = [self[x] for x in name]
        elif isiterable(name):
            ret = [self[x] for x in name]
        else:
            raise IndexError("{} was not found in the setas attribute.".format(name))
        return ret

    def __iter__(self):
        """Iterate over thew column assignments.

        .. warn::

            This class does not follow standard Mapping semantics - iterating iterates over the values and not the items.
        """
        _ = self.setas  # Force setas to fix size
        for c in self._setas:
            yield c

    def __ne__(self, other):
        """!= is the same as no ==."""
        return not self.__eq__(other)

    def __setitem__(self, name, value):
        """Allow setting of the setas variable like a dictionary or a list.

        Args:
            name (string or int): If name is a string, it should be in the set x,y,z,u,v,w,d,e or f
                and value should be a column index type. If name is an integer, then value should be
                a single letter string in the set above.
            value (integer or column index): See above.
        """
        if islike_list(name):  # Sipport indexing with a list like object
            if islike_list(value) and len(value) == len(name):
                for n, v in zip(name, value):
                    self._setas[n] = v
            else:
                for n in name:
                    self[n] = value
        elif isinstance(name, string_types) and len(name) == 1 and name in "xyzuvwdef.-":  # indexing by single letter
            for c in self.find_col(value, force_list=True):
                self._setas[c] = name
        elif (
            isinstance(name, index_types)
            and isinstance(value, string_types)
            and len(value) == 1
            and value in "xyzuvwdef.-"
        ):
            for c in self.find_col(name, force_list=True):
                self.setas[c] = value

    def __len__(self):
        """Return our own length."""
        return self._size

    def __repr__(self):
        """Our representation is as a list of the values."""
        return self.setas.__repr__()

    def __str__(self):
        """Our string representation is just fromed by joing the assingments together."""
        # Quick string conversion routine
        return "".join(self.setas)

    #################################################################################################################
    #############################   Operator Methods ################################################################

    def __add_core__(self, new, other):
        """Allow the user to add a dictionary to setas to add extra columns."""
        if not isinstance(other, dict):
            try:
                tmp = self.clone
                tmp(other)
                other = tmp.to_dict()
            except Exception:
                return NotImplemented
        for k, v in other.items():
            if isinstance(k, string_types) and len(k) == 1 and k in "xyzuvwdef":  # of the form x:column_name
                for v in new.find_col(v, force_list=True):
                    new._setas[v] = k
            elif (
                isinstance(k, index_types) and isinstance(v, string_types) and len(v) == 1 and v in "xyzuvwdef"
            ):  # of the form column_name:x
                k = new.find_col(k)
                new._setas[k] = v
            else:
                raise IndexError("Unable to workout what do with {}:{} when setting the setas attribute.".format(k, v))
        return new

    def __add__(self, other):
        """Jump to the core."""
        new = self.clone
        return self.__add_core__(new, other)

    def __iadd__(self, other):
        """Jump to the core."""
        new = self
        return self.__add_core__(new, other)

    def __sub_core__(self, new, other):
        """Implement subtracting either column indices or x,y,z,d,e,f,u,v,w for the current setas."""
        if isinstance(other, string_types) and len(other) == 1 and other in "xyzuvwdef":
            while True:
                try:
                    new._setas[new._setas.index(other)] = "."
                except ValueError:
                    break
            return new
        elif isinstance(other, index_types):
            try:
                new._setas[new.find_col(other)] = "."
                return new
            except KeyError:
                other = new.clone(other, _self=True)

        if isinstance(other, Mapping):
            me = new.to_dict()
            other = new.clone(other, _self=True).to_dict()
            for k, v in other.items():
                v = [v] if not isinstance(v, list) else v
                if k in me:
                    for header in v:
                        if header in me[k]:
                            if isinstance(me[k], list):
                                me[k].remove(header)
                            else:
                                me[k] = ""
                        else:
                            raise ValueError("{} is not set as {}".format(header, k))
                        if len(me[k]) == 0:
                            del me[k]
                else:
                    raise ValueError("No column is set as {}".format(k))
            new.clear()
            new(me)
            return new
        elif isiterable(other):
            for o in other:
                new = self.__sub_core__(new, o)
                if new is NotImplemented:
                    return NotImplemented
            return new
        return NotImplemented

    def __sub__(self, other):
        """Jump to the core."""
        new = self.clone
        return self.__sub_core__(new, other)

    def __isub__(self, other):
        """Jump to the core."""
        new = self
        return self.__sub_core__(new, other)

    def find_col(self, col, force_list=False):
        """Indexes the column headers in order to locate a column of data.shape.

        Indexing can be by supplying an integer, a string, a regular experssion, a slice or a list of any of the above.

        -   Integer indices are simply checked to ensure that they are in range
        -   String indices are first checked for an exact match against a column header
            if that fails they are then compiled to a regular expression and the first
            match to a column header is taken.
        -   A regular expression index is simply matched against the column headers and the
            first match found is taken. This allows additional regular expression options
            such as case insensitivity.
        -   A slice index is converted to a list of integers and processed as below
        -   A list index returns the results of feading each item in the list at :py:meth:`find_col`
            in turn.

        Args:
            col (int, a string, a re, a slice or a list):  Which column(s) to retuirn indices for.

        Keyword Arguments:
            force_list (bool): Force the output always to be a list. Mainly for internal use only

        Returns:
            The matching column index as an integer or a KeyError
        """
        if isinstance(col, int_types):  # col is an int so pass on
            if col >= len(self.column_headers):
                raise IndexError("Attempting to index a non - existant column {}".format(col))
            if col < 0:
                col = col % len(self.column_headers)
        elif isinstance(col, string_types):  # Ok we have a string
            col = str(col)
            if col in self.column_headers:  # and it is an exact string match
                col = self.column_headers.index(col)
            else:  # ok we'll try for a regular expression
                test = re.compile(col)
                possible = [x for x in self.column_headers if test.search(x)]
                if not possible:
                    try:
                        col = int(col)
                    except ValueError:
                        raise KeyError(
                            'Unable to find any possible column matches for "{} in {}"'.format(
                                col, self.column_headers
                            )
                        )
                    if col < 0 or col >= self.data.shape[1]:
                        raise KeyError("Column index out of range")
                else:
                    col = self.column_headers.index(possible[0])
        elif isinstance(col, _pattern_type):
            test = col
            possible = [x for x in self.column_headers if test.search(x)]
            if not possible:
                raise KeyError("Unable to find any possible column matches for {}".format(col.pattern))
            else:
                col = self.find_col(possible)
        elif isinstance(col, slice):
            indices = col.indices(self.shape[1])
            col = range(*indices)
            col = self.find_col(col)
        elif isiterable(col):
            col = [self.find_col(x) for x in col]
        else:
            raise TypeError("Column index must be an integer, string, list or slice, not a {}".format(type(col)))
        if force_list and not isinstance(col, list):
            col = [col]
        return col

    def clear(self):
        """"Clear the current setas attrbute.

        Notes:
            Equivalent to doing :py:meth:`setas.unset` with no argument.
        """
        self.unset()

    def get(self, name, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            return self[name]
        except (IndexError, KeyError):
            if default is not None:
                return default
            else:
                raise KeyError("{} is not in setas and no default was given.".format(name))

    def keys(self):
        """Mapping keys are the same as iterating over the unique headers"""
        for c in self._unique_headers:
            yield c

    def values(self):
        """Mapping values are the same as iterating over setas."""
        for v in self.setas:
            yield v

    def items(self):
        """Mapping items iterates over keys and values."""
        for k, v in zip(self._unique_headers, self.setas):
            yield k, v

    def pop(self, name, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            ret = self[name]
            self.unset(name)
            return ret
        except (IndexError, KeyError):
            if default is not None:
                return default
            else:
                raise KeyError("{} is not in setas and no default was given.".format(name))

    def popitem(self):
        for c in "xdyezfuvw":
            if c in self:
                v = self[c]
                self.unset(c)
                return (c, v)
        raise KeyError("No columns set in setas!")

    def setdefault(self, name, default=None):  # pylint:  disable=arguments-differ
        """Implement a setdefault method."""
        try:
            return self[name]
        except (IndexError, KeyError):
            self[name] = default
            return default

    def unset(self, what=None):
        """Remove column settings from the setas attribute in  method call.

        Parameters:
            what (str,iterable,dict or None): What to unset.

        Notes:
            The *what* parameter determines what to unset, possible values are:

            -   A single lets from *xyzuvwdef* - all column assignments of the corresponding type are unset
            -   A column index type - all matching columns are unset
            -   A list or other iterable of the above - all matching entries are unset
            -   None - all setas assignments are cleared.
        """
        if what is None:
            self.setas = []
            _ = self.setas
        else:
            self -= what

    def update(self, other=(), **kwds):  # pylint:  disable=arguments-differ
        """Replace any assignments in self with assignments from other."""
        if isinstance(other, setas):
            other = other.to_dict()
        elif isinstance(other, tuple) and len(other) == 0:
            other = kwds
        else:
            try:
                other = dict(other)
            except (ValueError, TypeError):
                raise TypeError("setas.update requires a dictionary not a {}".format(type(other)))
        vals = list(other.values())
        keys = list(other.keys())
        for k in "xyzuvwdef":
            if k in other:
                try:
                    c = self[k]
                    self[c] = "."
                except (KeyError, IndexError):
                    pass
                self[k] = other[k]
            elif k in vals:
                try:
                    c = self[k]
                    self[c] = "."
                except IndexError:
                    pass
                self[k] = keys[vals.index(k)]
        return self

    def to_dict(self):
        """Return the setas attribute as a dictionary.

        If multiple columns are assigned to the same type, then the column names are
        returned as a list. If column headers are duplicated"""
        ret = dict()
        for (k, ch) in zip(self._setas, self._unique_headers):
            if k != ".":
                if k in ret:
                    ret[k].append(ch)
                else:
                    ret[k] = [ch]
        for k in ret:
            if len(ret[k]) == 1:
                ret[k] = ret[k][0]
        return ret

    def to_list(self):
        """Returns the setas attribute as a list of letter types."""
        return list(self)

    def to_string(self, encode=False):
        """"Return the setas attribute encoded as a string, optionally replacing runs of 3 or more identical characters with a precediung digit."""
        expanded = "".join(self)
        if encode:
            pat = re.compile(r"((.)\2{2,9})")
            while True:
                res = pat.search(expanded)
                if not res:
                    break
                start, stop = res.span()
                let = str(stop - start) + res.group(2)
                expanded = expanded[:start] + let + expanded[stop:]
        return expanded

    def _get_cols(self, what=None, startx=0, no_guess=False):
        """Uses the setas attribute to work out which columns to use for x,y,z etc.

        Keyword Arguments:
            what (string): Returns either xcol, ycol, zcol, ycols,xcols rather than the full dictionary
            startx (int): Start looking for x columns at this column.
        Returns:
            A single integer, a list of integers or a dictionary of all columns.
        """
        # Do the xcolumn and xerror first. If only one x column then special case to reset startx to get any
        # y columns
        if self.setas.count("x") == 1:
            xcol = self.setas.index("x")
            maxcol = len(self.setas) + 1
            startx = 0
            xerr = self.setas.index("d") if "d" in self.setas else None
        elif self.setas.count("x") > 1:
            xcol = self.setas[startx:].index("x") + startx
            startx = xcol
            try:
                maxcol = self.setas[xcol + 1 :].index("x") + xcol + 1
            except ValueError:
                maxcol = len(self.setas)
            xerr = self.setas[startx:maxcol].index("d") if "d" in self.setas[startx:maxcol] else None
        else:
            xcol = None
            maxcol = len(self.setas) + 1
            startx = 0
            xerr = None

        # No longer enforce ordering of yezf - allow them to appear in any order.
        columns = {"y": [], "e": [], "z": [], "f": [], "u": [], "v": [], "w": []}
        for ix, lett in enumerate(self.setas[startx:maxcol]):
            if lett in columns:
                columns[lett].append(ix + startx)

        if xcol is None:
            axes = 0
        elif not columns["y"]:
            axes = 1
        elif not columns["z"]:
            axes = 2
        else:
            axes = 3
        if axes == 2 and len(columns["u"]) * len(columns["v"]) > 0:
            axes = 4
        elif axes == 3:
            if len(columns["u"]) * len(columns["v"]) * len(columns["w"]) > 0:
                axes = 6
            elif len(columns["u"]) * len(columns["v"]) > 0:
                axes = 5

        ret = _attribute_store()
        ret.update({"axes": axes, "xcol": xcol, "xerr": xerr})

        for ck, rk in {
            "y": "ycol",
            "z": "zcol",
            "e": "yerr",
            "f": "zerr",
            "u": "ucol",
            "v": "vcol",
            "w": "wcol",
        }.items():
            ret[rk] = columns[ck]

        if axes == 0 and len(self.shape) >= 2 and self.shape[1] in self._col_defaults and not no_guess:
            ret = self._col_defaults[self.shape[1]]
        for n in ["xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol", "axes"]:
            ret["has_{}".format(n)] = not (ret[n] is None or (isinstance(ret[n], list) and not ret[n]))

        ret["has_uvw"] = ret["has_ucol"] & ret["has_vcol"] & ret["has_wcol"]

        if what in ["xcol", "xerr"]:
            ret = ret[what]
        elif what in ("ycol", "zcol", "ucol", "vcol", "wcol", "yerr", "zerr"):
            ret = ret[what][0]
        elif what in ("ycols", "zcols", "ucols", "vcols", "wcols", "yerrs", "zerrs"):
            ret = ret[what[0:-1]]
        return ret
