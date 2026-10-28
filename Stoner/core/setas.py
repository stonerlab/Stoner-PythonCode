#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setas module provides the setas class for DataFile and friends."""

__all__ = ["Setas"]
import copy
import re
from collections.abc import MutableMapping

import numpy as np

from ..compat import _pattern_type, string_types
from ..tools import TypedList, isiterable
from .utils import decode_string
from .column_schema import (
    _add_roles, _assign_roles, _parse_roles, _resolve_roles,
    _role_groups, _subtract_roles, _update_roles,
)


class Setas(MutableMapping):
    """Expose scientific column roles through the shared schema/role operations.

    Notes:
        Iteration yields role characters rather than mapping keys. Callable,
        mapping, sequence and arithmetic forms retain the Setas compatibility
        API. Detached instances own their role/header lists; Data supplies a thin
        facade over its current immutable ColumnSchema. Derived groups are fresh
        results, never another authoritative role store.
    """

    _owner_bound = False

    def __init__(self, row=False, bless=None):
        """Initialise detached role descriptions.

        Args:
            row (bool):
                Interpret a one-dimensional shape as a row of columns when True.
            bless (object or None):
                Return this object after callable assignments with _self=False.
        """
        self._row = row
        self._shape = tuple()
        self._setas = []
        self._column_headers = TypedList(string_types)
        self._object = bless

    def __setstate__(self, state):
        """Read earlier role snapshots without reviving redundant schema/group caches."""
        self.__dict__.update({key: value for key, value in state.items()
                              if key not in {"_schema", "_cols", "_col_defaults"}})

    # ==========================================================================
    # Role views and compatibility descriptors
    # ==========================================================================


    @property
    def _size(self):
        """Calculate a size of the setas attribute."""
        match len(self.shape):
            case 1 if self._row:
                return self._shape[0]
            case 1:
                return 1
            case 0:
                return len(self._column_headers)
            case _:
                return self.shape[1]

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
        """Detach role/header state without cloning the blessed owning object.

        The owner reference is retained only for callable return conventions;
        edits to this clone never change the owner's column descriptions.
        """
        cls = Setas if self._owner_bound else type(self)
        new = cls()
        new._row, new._object = self._row, self._object
        new.shape = self.shape
        new.column_headers = list(self.column_headers)
        new._setas = self.to_list()
        for name, value in self.__dict__.items():
            if name not in {"_row", "_shape", "_setas", "_column_headers", "_object", "_data"} and not callable(value):
                new.__dict__[name] = copy.deepcopy(value)
        return new

    @property
    def cols(self):
        """Return newly derived groups without caching mutable defaults."""
        return self._get_cols()

    @property
    def _cols(self):
        """Return fresh groups for legacy internal consumers without a mutable cache."""
        return self.cols

    def _commit_roles(self, roles):
        """Replace detached roles after a complete assignment has validated."""
        self._setas = list(roles)

    def _check_writable(self):
        """Provide the mutation guard used by the owner-bound facade."""

    @property
    def x(self):
        """Quick access to the x column number.

        Just a convenience read only property. If we want to change the setas.x
        value we should use the setas(x=1,y=2) style call (so that reset can
        be handled properly)
        """
        return self.cols["xcol"]

    @property
    def y(self):
        """Quick access to the y column numbers list."""
        return self.cols["ycol"]

    @property
    def z(self):
        """Quick access to the z column numbers list."""
        return self.cols["zcol"]

    @property
    def column_headers(self):
        """Get the current column headers."""
        cols = self._size
        length = len(self._column_headers)
        if length < cols:  # Extend the column headers if necessary
            self._column_headers.extend([f"Column {i + length}" for i in range(cols - length)])
        return self._column_headers

    @column_headers.setter
    def column_headers(self, value):
        """Set the column headers."""
        if isinstance(value, np.ndarray):  # Convert ndarray to list of strings
            value = value.astype(str).tolist()
        elif isinstance(value, string_types):  # Bare strings get turned into lists
            value = [value]
        self._column_headers = TypedList(string_types, value)

    @property
    def empty(self):
        """Determine if any columns are set."""
        return len(self._setas) == 0 or np.all(np.array(self._setas) == ".")

    @property
    def ndim(self):
        """Return the number of dimensions of the array."""
        return len(self.shape)

    @property
    def not_set(self):
        """Return a boolean array if not set."""
        return np.array([x == "." for x in self._setas])

    @property
    def set(self):
        """Return a boolean array if column is set."""
        return np.array([x != "." for x in self._setas])

    @property
    def setas(self):
        """Guard the setas attribute."""
        cols = self._size
        length = len(self._setas)
        if cols > length:
            self._setas.extend(["."] * (cols - length))
        self._setas = self._setas[:cols]
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
            raise AttributeError(f"shape attribute should be a 2-tuple not a {value}-tuple")

    def __call__(self, *args, **kwargs):
        """Read or update the column-role assignments.

        Args:
            *args (str, iterable, dict or Setas):
                Supply one positional assignment: a role string, a sequence of role
                characters, a mapping of roles to columns or columns to roles, or another
                Setas object. Without a positional assignment, use keyword assignments.

        Keyword Arguments:
            _self (bool or None):
                After an assignment, True returns this Setas object, False returns its
                owning object and None returns None. Defaults to False. No copy is made.
            reset (bool):
                Clear existing roles before applying a string or keyword assignment when
                True (the default). False preserves roles not explicitly changed.
                Positional sequences and mappings are handled without this preliminary reset.
            **kwargs (column selectors or str):
                Assign roles using keywords such as ``x="column_1"`` or ``column_4="z"``.
                Column selectors are resolved by find_col.

        Returns:
            list, Setas, object or None:
                With no assignments or reset keyword, return the current role list, even
                if _self was supplied. After an assignment, return the object selected by _self.

        Raises:
            SyntaxError:
                If a single positional argument has an unsupported type.
            IndexError:
                If a mapping entry cannot be interpreted or a column cannot be resolved.
            ValueError:
                If a role character or encoded role string is invalid.

        Notes:
            Role sequences are truncated or padded to the number of columns. A period
            clears a role; a hyphen leaves that position unchanged after any preliminary
            reset. Assignments validate completely before updating this object.
            Errors leave both bound and detached assignments unchanged.

        Example:
            Assign a role string, or use named columns and numerical indices::

                data.setas("xyzuvw")
                data.setas(["x", "y", "z", "u", "v", "w"])
                data.setas(x="column_1", y=3, column_4="z")
                data.setas(y=2, reset=False)
        """
        return_self = kwargs.pop("_self", False)
        if not (args or kwargs):
            return self.setas
        self._check_writable()
        if args and isinstance(args[0], Setas):
            args = (args[0].to_list(),) + args[1:]
        roles = _parse_roles(self.column_headers, self.to_list(), args, kwargs)
        self._commit_roles(roles)
        return None if return_self is None else self if return_self else self._object

    def __contains__(self, item):
        """Use getitem to test for membership. Either column assignments or column index types are tested."""
        try:
            _ = self[item]
        except (IndexError, KeyError, ValueError):
            return False
        return True

    def __delitem__(self, name):
        """Unset either by column index or column assignment.

        Equivalent to unsetting the same object.
        """
        self.unset(name)

    def __eq__(self, other):
        """Check to see if this is the same object, or has the same headers and the same setas values."""
        ret = False
        if isinstance(other, string_types):  # Expand strings and convert to list
            other = list(decode_string(other))
        if not isinstance(other, Setas):  # Ok, need to check whether items match
            if isiterable(other) and len(other) <= self._size:
                for m in self.setas[len(other) :]:  # Check that if other is short we don't have assignments there
                    if m != ".":
                        return False
                for o, m in zip(other, self.setas):
                    if o != m:  # Look for mis-matched assignments
                        return False
                return True
            return False
        if id(self) == id(other):
            ret = True
        else:
            ret = self.column_headers == other.column_headers and self.setas == other.setas
        return ret

    def __getattr__(self, name):
        """Try to see if attribute name is a key in self.cols and return that instead."""
        if (name in {"axes", "xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol"}
                or name.startswith("has_")):
            groups = self.cols
            if name in groups:
                return groups[name]
        raise AttributeError(name)

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
        match name:
            case "x" | "y" | "z" | "u" | "v" | "w" | "d" | "e" | "f" | "." | "-":
                ret = self.to_dict()[name]
                if len(ret) == 1:
                    ret = ret[0]
                return ret
            case str() if len(name) == 2 and ((name[0] == "#" and name[1] in "xyzdefuvw") or name == "y#"):
                role = name[1] if name[0] == "#" else name[0]
                ret = [i for i, current in enumerate(self.setas) if current == role]
                if len(ret) == 1:
                    ret = ret[0]
                return ret
            case int() | str() | _pattern_type():
                positions = self.find_col(name)
                return ([self.setas[i] for i in positions]
                        if isinstance(positions, list) else self.setas[positions])
            case slice():
                indices = name.indices(len(self.setas))
                name = range(*indices)
                return [self[x] for x in name]
            case _ if isiterable(name):
                result = []
                for item in name:
                    selected = self[item]
                    result.extend(selected if isinstance(selected, list) else [selected])
                return result
            case _:
                raise IndexError(f"{name} was not found in the setas attribute.")

    def __iter__(self):
        """Iterate over thew column assignments.

        .. warn::

            This class does not follow standard Mapping semantics - iterating iterates over the values and not
            the items.
        """
        _ = self.setas  # Force setas to fix size
        yield from self._setas

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
        self._check_writable()
        roles = _assign_roles(self.column_headers, self.to_list(), name, value)
        self._commit_roles(roles)

    def __len__(self):
        """Return our own length."""
        return self._size

    def __repr__(self):
        """Our representation is as a list of the values."""
        return self.setas.__repr__()

    def __str__(self):
        """Our string representation is just formed by joing the assignments together."""
        # Quick string conversion routine
        return "".join(self.setas)

    #################################################################################################################
    #############################   Operator Methods ################################################################

    def _add_core_(self, new, other):
        """Allow the user to add a dictionary to setas to add extra columns."""
        new._check_writable()
        if isinstance(other, Setas):
            other = other.to_list()
        try:
            roles = _add_roles(new.column_headers, new.to_list(), other)
        except SyntaxError:
            return NotImplemented
        new._commit_roles(roles)
        return new

    def __add__(self, other):
        """Jump to the core."""
        new = self.clone
        return self._add_core_(new, other)

    def __iadd__(self, other):
        """Jump to the core."""
        new = self
        return self._add_core_(new, other)

    def _sub_core_(self, new, other):
        """Implement subtracting either column indices or x,y,z,d,e,f,u,v,w for the current setas."""
        new._check_writable()
        if isinstance(other, Setas):
            other = other.to_dict()
        try:
            roles = _subtract_roles(new.column_headers, new.to_list(), other)
        except TypeError:
            return NotImplemented
        new._commit_roles(roles)
        return new

    def __sub__(self, other):
        """Jump to the core."""
        new = self.clone
        return self._sub_core_(new, other)

    def __isub__(self, other):
        """Jump to the core."""
        return self._sub_core_(self, other)

    def find_col(self, col, force_list=False, *, first=False):
        """Resolve column selectors with the shared Stoner matching rules.

        Args:
            col (int, str, re.Pattern, slice or sequence):
                Exact names, then glob or substring, then regex fallback. All
                matches are retained; compiled patterns explicitly request regex.

        Keyword Arguments:
            force_list (bool):
                Always return a list, including a single match.
            first (bool):
                Return only the first position after validating the complete selector.

        Returns:
            int or list of int:
                Positions, preserving ordered repeated selections. Strings with
                one match return an integer; pattern/collection results remain lists.

        Raises:
            KeyError:
                If no name/pattern matches or first is requested on an empty selection.
            IndexError:
                If a position is out of bounds.
            TypeError:
                If a selector has an unsupported type, including Boolean values.
            re.error:
                If regex fallback is required and the pattern is invalid.
        """
        return _resolve_roles(self.column_headers, col, force_list=force_list, first=first)

    def clear(self):
        """Clear the current setas attribute.

        Notes:
            Equivalent to doing :py:meth:`setas.unset` with no argument.
        """
        self.unset()

    def get(self, key, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            return self[key]
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"{key} is not in setas and no default was given.") from err

    def keys(self):
        """Access mapping keys.

        Mapping keys are the same as iterating over the unique headers
        """
        yield from self._unique_headers

    def values(self):
        """Access mapping values.

        Mapping values are the same as iterating over setas.
        """
        yield from self.setas

    def items(self):
        """Access mapping items.

        Mapping items iterates over keys and values.
        """
        yield from zip(self._unique_headers, self.setas)

    def pop(self, key, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            ret = self[key]
            self.unset(key)
            return ret
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"{key} is not in setas and no default was given.") from err

    def popitem(self):
        """Return and clear a column assignment."""
        for c in "xdyezfuvw":
            if c in self:
                v = self[c]
                self.unset(c)
                return (c, v)
        raise KeyError("No columns set in setas!")

    def setdefault(self, key, default=None):  # pylint:  disable=arguments-differ
        """Implement a setdefault method."""
        try:
            return self[key]
        except (IndexError, KeyError):
            self[key] = default
            return default

    def unset(self, what=None):
        """Remove column settings from the setas attribute in  method call.

        Args:
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
        """Replace declared role groups atomically, retaining unrelated assignments.

        Multiple mapping entries targeting one role replace that group together.
        Entries are applied in mapping order, so later writes win on overlaps.
        """
        self._check_writable()
        if isinstance(other, Setas):
            other = other.to_dict()
        try:
            assignments = dict(other, **kwds)
        except (TypeError, ValueError) as error:
            raise TypeError(f"setas.update requires a mapping, not {type(other)}") from error
        self._commit_roles(_update_roles(self.column_headers, self.to_list(), assignments))
        return self

    def to_dict(self):
        """Return the setas attribute as a dictionary.

        If multiple columns are assigned to the same type, then the column names are
        returned as a list. If column headers are duplicated.
        """
        ret = {}
        for k, ch in zip(self._setas, self._unique_headers):
            if k != ".":
                if k in ret:
                    ret[k].append(ch)
                else:
                    ret[k] = [ch]
        for k, v in ret.items():
            if len(v) == 1:
                ret[k] = v[0]
        return ret

    def to_list(self):
        """Return the setas attribute as a list of letter types."""
        return list(self)

    def to_string(self, encode=False):
        """Return the setas attribute encoded as a string.

        Optionally replaces runs of 3 or more identical characters with a precediung digit.
        """
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
        """Use the setas attribute to work out which columns to use for x,y,z etc.

        Keyword Arguments:
            what (string): Returns either xcol, ycol, zcol, ycols,xcols rather than the full dictionary
            startx (int): Start looking for x columns at this column.

        Returns:
            A single integer, a list of integers or a dictionary of all columns.
        """
        return _role_groups(self.setas, self.shape, row=self._row, startx=startx, no_guess=no_guess, what=what)
