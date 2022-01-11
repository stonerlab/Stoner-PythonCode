# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:51:50 2022

@author: phygbu
"""
__all__ = ["Column_Headers"]
from collections.abc import MutableSequence
from pprint import pformat

from ..tools import isLikeList


class Column_Headers(MutableSequence):

    """Provide an interface to a DataFrame's columns that is mutable."""

    def __init__(self, obj):
        """Construct the sequence.

        Args:
            obj (DataFile):
                The DataFile which we're mapping columns for.
        """
        self._obj = obj

    def __contains__(self, x):
        """Contains checks the column names int he data."""
        return x in self._obj._data.columns

    def __getitem__(self, i):
        """Minimally return the corresponding columns item."""
        return self._obj._data.columns[i]

    def __setitem__(self, name, value):
        """Carry out a rename operation on the _data, _mask and setas._index."""
        oldname = self._obj._data.columns[name]
        self._obj._data.rename(columns={oldname: value}, inplace=True)
        self._obj._mask.rename(columns={oldname: value}, inplace=True)
        self._obj._setas._index.rename(index={oldname: value}, inplace=True)

    def __delitem__(self, name):
        """Deletions are not supported!"""
        return NotImplemented

    def __len__(self):
        """Length is always the number of columns."""
        return self._obj.shape[1]

    def __repr__(self):
        return pformat([x for x in self])

    def insert(self, index, object):
        """Insertions are also not supported."""
        return NotImplemented

    def set_all(self, seq):
        """Set all the headers in one go from a sequence."""
        mapping = {c: s for c, s in zip(self._obj._data.columns, seq)}
        self._obj._data.rename(columns=mapping, inplace=True)
        self._obj._mask.rename(columns=mapping, inplace=True)
        self._obj._setas._index.rename(index=mapping, inplace=True)
