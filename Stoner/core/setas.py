#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setas module provides the setas class for DataFile and friends."""
__all__ = ["Setas"]
import re
import copy
from collections.abc import MutableMapping, Mapping, Iterable
from warnings import warn
import fnmatch
from typing import List, Union, Any, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from ..compat import string_types, int_types, index_types, _pattern_type
from ..tools import AttributeStore, isiterable, typedList, isLikeList, make_Data, isnone
from .utils import decode_string

_col_defaults: dict = {
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


class Setas(MutableMapping):

    """A Class that provides a mechanism for managing the column assignments in a DataFile like object.

    Implements a MutableMapping bsed on the column_headers as the keys (with a few tweaks!).

    Note:
        Iterating over setas will return the column assignments rather than the standard mapping behaviour of
        iterating over the keys. Otherwise
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

    codes: List[str] = [l for l in "xyzdefuvw"]
    setable_codes: List[str] = codes + ["."]
    full_codes: List[str] = codes + [".", "-"]

    def __init__(self, datafile: "Stoner.Core.DataFile"):
        """Construct the setas object - linked to the specified datafile object."""
        self._datafile = datafile
        self._index = pd.Series({ch: "." for ch in self._datafile._data.columns})
        self._cols = {}

    def __call__(self, *args, **kargs) -> Optional[Union["Setas", "Stoner.Core.DataFile"]]:
        """Treat the current instance as a callable object and assign columns accordingly.

        Variois forms of this method are accepted::

        setas("xyzuvw")
        setas(["x"],["y"],["z"],["u"],["v"],["w"])
        setas(x="column_1",y=3,column4="z")

        Keyword Arguments:
            _self (bool):
                If True, make the call return a copy of the setas object, if False, return _object attribute, if None,
                return None. Default - **False**
            reset (bool):
                If False then preserve the existing set columns and simply add the new ones. Otherwise, clear
                all column assignments before setting new ones (default).
        """
        return_self = kargs.pop("_self", False)
        if not (args or kargs):
            return self._index
        if len(args) == 1 and not isiterable(args[0]):
            raise SyntaxError(
                f"setas should be called with eother a string, iterable object or setas object, not a {type(args[0])}"
            )
        elif len(args) == 0 and len(kargs) > 0:
            args = [kargs]
        elif len(args) >= 2:
            args = [args]
        # At this point all the arguments are in args[0]

        if isinstance(args[0], string_types):
            self.from_string(args[0])
        elif isinstance(args[0], Mapping):
            self.from_dict(args[0])
        elif isLikeList(args[0]):
            self.from_list(args[0])
        else:
            raise ValueError(f"Unable to set from {args[0]}")
        self._get_cols()
        if return_self is None:
            return None
        if return_self:
            return self
        return self._datafile

    def __contains__(self, item: str) -> bool:
        """Use getitem to test for membership. Either column assignments or column index types are tested."""
        if item in self.codes:
            return item in self._index.values
        else:
            return item in self._index.index

    def __delitem__(self, name: str) -> None:
        """Unset either by column index or column assignment.

        Equivalent to unsetting the same object."""
        self.unset(name)

    def __eq__(self, other: Union[str, List[str], Mapping]) -> bool:
        """Check to see if this is the same object, or has the same headers and the same setas values."""
        # Try to convert other into a pd.Series to compare with self._index
        if isinstance(other, string_types):  # Expand strings and convert to list
            other = pd.Series({ch: c for ch, c in zip(self._index.columns, decode_string(other))})
        if isinstance(other, Mapping):  # This will include self setas to
            other = pd.Series(other)
        if isiterable(other) and not isinstance(other, pd.Series):
            other = pd.Series({ch: c for ch, c in zip(self._index.columns, other)})
        return other == self._index

    def __getitem__(self, name: Union[str, int, List, _pattern_type]) -> pd.Series:
        """Permit the setas attribute to be treated like either a list or a dictionary.

        Args:
            name (int, slice or string): if *name* is an integer or a slice, return the column type
                of the corresponding column(s). If a string, should be a single letter
                from the set x,y,z,u,v,w,d,e,f - if so returns the corresponding
                column(s)

        Returns:
            pandas.Series object with an index of columm names and the corresponding codes.
        """
        if name in self.codes:
            ret = self.to_dict()[name]
            if len(ret) == 1:
                ret = ret[0]
        elif isinstance(name, index_types + (slice, Iterable)):
            ret = self._index[self.find_col(name)]
        else:
            raise IndexError(f"{name} was not found in the setas attribute.")
        return ret

    def __iter__(self):
        """Iterate over thew column assignments.

        .. warn::

            This class does not follow standard Mapping semantics - iterating iterates over the values and not
            the items.
        """
        for c in self._index:
            yield c

    def __ne__(self, other: Union[str, List[str], Mapping]) -> bool:
        """!= is the same as no ==."""
        return not self.__eq__(other)

    def __setitem__(self, name: Union[str, List[str], Mapping], value: str):
        """Allow setting of the setas variable like a dictionary or a list.

        Args:
            name (string or int): If name is a string, it should be in the set x,y,z,u,v,w,d,e or f
                and value should be a column index type. If name is an integer, then value should be
                a single letter string in the set above.
            value (integer or column index): See above.
        """
        if isLikeList(name):  # Support indexing with a list like object
            if isLikeList(value) and len(value) == len(name):
                for n, v in zip(name, value):
                    self._setas[n] = v
            else:
                for n in name:
                    self[n] = value
        elif name in self.setable_codes:  # indexing by single letter
            self._index[self.find_col(value)] = name
        elif isinstance(name, index_types) and value in self.setable_codes:
            for c in self.find_col(name):
                self._index[c] = value
        else:
            raise IndexError(f"Failed to set setas as couldn't workout what todo with setas[{name}] = {value}")
        self._get_cols()

    def __len__(self) -> int:
        """Return our own length."""
        return self._index.size

    def __repr__(self) -> str:
        """Our representation is as a list of the values."""
        return repr(list(self._index))

    def __str__(self) -> str:
        """Our string representation is just fromed by joing the assingments together."""
        # Quick string conversion routine
        return self.to_string()

    def _match(self, string: str) -> pd.Series:
        """Try matching the string against all the column names in our _index."""
        result = []
        try:
            pat = re.compile(string)
        except re.error:
            pat = None
        for ch in self._index.index:
            if not isinstance(ch, str):
                ch = str(ch)
            if string in ch:
                result.append(True)
                continue
            if fnmatch.fnmatch(ch, string):
                result.append(True)
                continue
            if pat is not None and pat.search(ch):
                result.append(True)
                continue
            result.append(False)
        return pd.Series(result, index=self._index.index)

    def _col_args(
        self,
        scalar=True,
        xcol=None,
        ycol=None,
        zcol=None,
        ucol=None,
        vcol=None,
        wcol=None,
        xerr=None,
        yerr=None,
        zerr=None,
        **kargs,
    ):  # pylint: disable=unused-argument
        """Create an object which has keys  based either on arguments or setas attribute."""
        cols = {
            "xcol": xcol,
            "ycol": ycol,
            "zcol": zcol,
            "ucol": ucol,
            "vcol": vcol,
            "wcol": wcol,
            "xerr": xerr,
            "yerr": yerr,
            "zerr": zerr,
        }
        no_guess = kargs.get("no_guess", True)
        for i in cols.values():
            if i is not None:  # User specification wins out
                break
        else:  # User didn't set any values, setas will win
            no_guess = kargs.get("no_guess", False)
        ret = AttributeStore(self._get_cols(no_guess=no_guess))
        force_list = kargs.get("force_list", not scalar)
        for c in list(cols.keys()):
            if isnone(cols[c]):  # Not defined, fallback on setas
                del cols[c]
                continue
            if isinstance(cols[c], bool) and not cols[c]:  # False, delete column altogether
                del cols[c]
                if c in ret:
                    del ret[c]
                continue
            if c in ret and isinstance(ret[c], list):
                if isinstance(cols[c], float) or (
                    isinstance(cols[c], (np.ndarray, pd.Series)) and cols[c].size == len(self)
                ):
                    continue
            if isinstance(cols[c], float):
                continue
            cols[c] = self.setas.find_col(cols[c])
            if not force_list and isLikeList(cols[c]) and len(cols[c] == 1):
                cols[c] = cols[c][0]
        ret.update(cols)
        if scalar:
            for c in ret:
                if isLikeList(ret[c]):
                    if len(ret[c]) > 0:
                        ret[c] = ret[c][0]
                    else:
                        ret[c] = None
        elif force_list or (isinstance(scalar, bool) and not scalar):
            for c in ret:
                if c.startswith("x") or c.startswith("has_"):
                    continue
                if not isLikeList(ret[c]) and ret[c] is not None:
                    ret[c] = pd.Index([ret[c]])
                elif ret[c] is None:
                    ret[c] = pd.Index([])
        for n in ["xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol", "axes"]:
            ret[f"has_{n}"] = n in ret and not (ret[n] is None or (isinstance(ret[n], list) and not ret[n]))

        return ret

    def _get_cols(
        self, what: Optional[str] = None, startx: Union[str, int, _pattern_type] = 0, no_guess: bool = False,
    ) -> Dict:
        """Use the setas attribute to work out which columns to use for x,y,z etc.

        Keyword Arguments:
            what (string): Returns either xcol, ycol, zcol, ycols,xcols rather than the full dictionary
            startx (int): Start looking for x columns at this column.

        Returns:
            A single integer, a list of integers or a dictionary of all columns.
        """
        # Do the xcolumn and xerror first. If only one x column then special case to reset startx to get any
        # y columns
        maxcol = self._index.tail(1)

        if self.count("x") == 1:
            xcol = self.index("x")
            startx = self._index.head(1)
            xerr = self.index("d") if "d" in self else None
        elif self.count("x") > 1:
            startx = self.find_col(startx)
            xcol = self.index("x", start=startx)
            startx = xcol
            xerr = self.index("d", start=startx) if "d" in self else None
        else:
            xcol = None
            startx = self._index.head(1)
            xerr = None
        # No longer enforce ordering of yezf - allow them to appear in any order.
        columns = {"y": [], "e": [], "z": [], "f": [], "u": [], "v": [], "w": []}
        if not maxcol.empty:
            for col, lett in self._index[startx.index[0] : maxcol.index[0]].iteritems():
                if lett in columns:
                    columns[lett].append(col)
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
        ret = AttributeStore()
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
        if (
            axes == 0
            and self._datafile._data.ndim >= 2
            and self._datafile._index.size in _col_defaults
            and not no_guess
        ):
            ret = _col_defaults[self._index.size]
        for n in [
            "xcol",
            "xerr",
            "ycol",
            "yerr",
            "zcol",
            "zerr",
            "ucol",
            "vcol",
            "wcol",
            "axes",
        ]:
            ret[f"has_{n}"] = not (ret[n] is None or (isinstance(ret[n], list) and not ret[n]))
        ret["has_uvw"] = ret["has_ucol"] & ret["has_vcol"] & ret["has_wcol"]

        if what in ["xcol", "xerr"]:
            ret = ret[what]
        elif what in ("ycol", "zcol", "ucol", "vcol", "wcol", "yerr", "zerr"):
            ret = ret[what][0]
        elif what in ("ycols", "zcols", "ucols", "vcols", "wcols", "yerrs", "zerrs"):
            ret = ret[what[0:-1]]
        self._cols = ret
        self.__dict__.update(ret)
        return ret

    @property
    def clone(self) -> "Setas":
        """Minimal clone method."""
        breakpoint()
        return self.copy(self._datafile)

    @property
    def cols(self) -> Dict:
        """Get the current column assignments."""
        return self._cols

    @property
    def empty(self) -> bool:
        """Determine if any columns are set."""
        return self._index.size == 0 or np.all(self._index == ".")

    @property
    def not_set(self) -> pd.Series:
        """Return a boolean array if not set."""
        return self._index.index[self._index == "."]

    @property
    def set(self) -> pd.Series:
        """Return a boolean array if column is set."""
        return self._index.index[self._index != "."]

    def find_col(
        self, col: Union[str, int, _pattern_type, List[Union[str, int, _pattern_type]]], as_int: bool = False
    ) -> pd.Index:
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

        Returns:
            The mpandas.Index a KeyError
        """
        if as_int:  # Recursive call for returning integers, but running through index()
            return [self.index(x) for x in self.find_col(col)]
        if isinstance(col, slice):  # Deal with slices
            if isinstance(col.start, string_types) and isinstance(col.stop, string_types):
                start = self.find_col(col.sart)
                stop = self.find_col(col.stop)
                return self._index[start : stop : col.step].index
            indices = col.indices(len(self._index))
            col = range(*indices)
            return self.find_col(col)
        if isLikeList(col):  # Deal with a list like iterable
            ret = pd.Index([])
            for ix, c in enumerate(col):
                ret = ret.union(self.find_col(c))
            return ret.to_list()
        if isinstance(col, int):
            return pd.Index([self._index.index[col]]).to_list()
        if isinstance(col, string_types):
            if col in self.codes:
                return self._index.index[self._index == col].to_list()
            if col in self._index:
                return self._index.index[self._index.index == col].to_list()
            return self._index.index[self._match(col) == True].to_list()
        if isinstance(col, _pattern_type):
            return self._index.index[[col.search(ch) is not None for ch in self._index.index].to_list()]
        raise KeyError(f"{type(col)} could not be interpreted as a coluimn index")

    def clear(self):
        """Clear the current setas attrbute.

        Notes:
            Equivalent to doing :py:meth:`setas.unset` with no argument.
        """

    def count(self, item):
        """Cound the number of columns set as item."""
        return self._index[self._index == item].count()

    def get(self, name, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            return self[name]
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"{name} is not in setas and no default was given.") from err

    def keys(self):
        """Acess mapping keys.

        Mapping keys are the same as iterating over the headers"""
        for c in self._index.index:
            yield c

    def values(self):
        """Access mapping values.

        Mapping values are the same as iterating over setas."""
        for _, v in self.items():
            yield v

    def index(self, item, start=None, stop=None):
        """Return the first column with the speficied item."""

        if item in self.codes:
            working = self._index.to_list()
        else:
            working = self._index.index.to_list()

        for check in (start, stop):
            if check is not None and check not in self:
                raise ValueError(f"{check} not in setas list")

        if start is not None:
            start = self._index.index.to_list().index(start)
        else:
            start = 0
        if stop is not None:
            stop = self._index.index.to_list().index(stop)
        else:
            stop = len(self)

        return working.index(item, start, stop)

    def items(self) -> Tuple[str, str]:
        """Access mapping items.

        Mapping items iterates over keys and values."""
        for k in self.keys():
            yield k, self._index[k]

    def pop(self, name, default=None):  # pylint:  disable=arguments-differ
        """Implement a get method."""
        try:
            ret = self[name]
            self.unset(name)
            return ret
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"{name} is not in setas and no default was given.") from err

    def popitem(self):
        """Return and clear a column assignment."""
        for c in "xdyezfuvw":
            if c in self._index:
                v = self._index[c]
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

        Returns:
            (Setas):
                The modified Setas object

        Notes:
            The *what* parameter determines what to unset, possible values are:

            -   A single lets from *xyzuvwdef* - all column assignments of the corresponding type are unset
            -   A column index type - all matching columns are unset
            -   A list or other iterable of the above - all matching entries are unset
            -   None - all setas assignments are cleared.
        """
        if what is None:
            self._index = pd.Series({c: "." for c in self._datefile._data.columns})
        else:
            self -= what
        return self

    def update(self, other=(), **kwds):  # pylint:  disable=arguments-differ
        """Replace any assignments in self with assignments from other."""
        if isinstance(other, Setas):
            other = other.to_dict()
        elif isinstance(other, tuple) and len(other) == 0:
            other = kwds
        else:
            try:
                other = dict(other)
            except (ValueError, TypeError) as err:
                raise TypeError(f"setas.update requires a dictionary not a {type(other)}") from err
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

    def copy(self, ref, adapt=True):
        """Create a copy of the atribute and reassign to the new reference.

        Args:
            ref (DataFile):
                The new DataFile instance that this setas is attached to.

        Keyword Args:
            adapt (bool):
                If True, rename the columns in this object to adapt to, if
                False, rename the columns in the newly refernce object.

        Returns:
            (Setas):
                A new setas object.
        """
        new = Setas(ref)
        if not adapt:
            mapping = {c: v for c, v in zip(ref._data.columns, self._index.index) if c != v}
            new._data.rename(columns=mapping, inplace=True)
            new._maske.rename(columns=mapping, inplace=True)
            new._index.rename(index=mapping, inplace=True)
        new._datafile.setas = self.to_list()
        return new

    def from_dict(self, dictionary):
        """Set from a dictionary."""
        self.clear()
        for k, v in dictionary.items():
            if v in self.full_codes:
                v, k = k, v  # Swap the key and value around so we can do code=value
            if k in self.setable_codes:  # set with code = column
                column = self.find_col(v)
                self._index[column] = k
                continue
            elif k in self.full_codes:
                continue
            else:
                raise KeyError(f"Cannot set column {v} to code {k}.")
        return self

    def from_list(self, listlike):
        """Sety the column assignments from an iterable list."""
        for c, v in zip(self._index.index, listlike):
            if v not in self.full_codes:
                raise ValueError(f"Unable to set a column as {v}.")
            if v == "-":
                continue
            self._index[c] = v
        return self

    def from_string(self, string):
        """Set the columns using a string - allowing for expansion of codes."""
        string = decode_string(string)
        return self.from_list([l for l in string])

    def to_dict(self):
        """Return the setas attribute as a dictionary.

        If multiple columns are assigned to the same type, then the column names are
        returned as a list. If column headers are duplicated"""
        ret = {}
        for k in self.codes:
            ret[k] = self._index.index[self._index == k]
        return ret

    def to_list(self):
        """Return the setas attribute as a list of letter types."""
        return list(self)

    def to_string(self, encode=False):
        """Return the setas attribute encoded as a string.

        Optionally replaces runs of 3 or more identical characters with a precediung digit."""
        expanded = "".join(self)
        if encode:
            pat = re.compile(r"((.)\2{2,})")
            while True:
                res = pat.search(expanded)
                if not res:
                    break
                start, stop = res.span()
                let = str(stop - start) + res.group(2)
                expanded = expanded[:start] + let + expanded[stop:]
        return expanded
