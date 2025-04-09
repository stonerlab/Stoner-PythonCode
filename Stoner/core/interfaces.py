# -*- coding: utf-8 -*-
"""Provide a mixin class the provides sequence and mapping like interfaces for Data."""

__all__ = ["DataFileInterfacesMixin"]

import numpy as np

from ..compat import _pattern_type, string_types
from ..tools import isiterable


class DataFileInterfacesMixin:
    """Implement the required methods for a sequence and mapping type object."""

    def __contains__(self, item):
        """Operator function for membertship tests - used to check metadata contents.

        Args:
            item(string):
                name of metadata key

        Returns:
            (bool):
                True if item in self.metadata
        """
        return item in self.metadata

    def __delitem__(self, item):
        """Implement row or metadata deletion.

        Args:
            item (ingteger or string):
                row index or name of metadata to delete
        """
        if isinstance(item, string_types):
            del self.metadata[item]
        else:
            self.del_rows(item)

    def __getitem__(self, name):
        """Return either a row or iterm of metadata.

        Args:
            name (string or slice or int):
                The name, slice or number of the part of the
            :py:class:`DataFile` to be returned.

        Returns:
            mixed: an item of metadata or row(s) of data.

        -   If name is an integer then the corresponding single row will be returned
        -   if name is a slice, then the corresponding rows of data will be returned.
        -   If name is a string then the metadata dictionary item             with the correspondoing key will be
            returned.
        -   If name is a numpy array then the corresponding rows of the data are returned.
        -   If a tuple is supplied as the argument then there are a number of possible behaviours.
            -   If the first element of the tuple is a string, then it is assumed that it is the nth element of the
                named metadata is required.
            -   Otherwise itis assumed that it is a particular element within a column determined by the second
                part of the tuple that is required.

        Examples:
            DataFile['Temp',5] would return the 6th element of the
            list of elements in the metadata called 'Temp', while

            DataFile[5,'Temp'] would return the 6th row of the data column
            called 'Temp'

            and DataFile[5,3] would return the 6th element of the
            4th column.
        """
        match name:
            case str() | _pattern_type():
                try:
                    return self.metadata[name]
                except KeyError:
                    try:
                        return self.data[name]
                    except KeyError as err:
                        raise KeyError(
                            f"{name} was neither a key in the metadata nor a column in the main data."
                        ) from err
            case tuple() if name in self.metadata:
                return self.metadata[name]
            case (str(), *rest):
                try:
                    ret = self.metadata[name[0]]
                    return ret.__getitem__(*rest)
                except KeyError:
                    try:
                        return self.data[name]
                    except KeyError as err:
                        raise KeyError(
                            f"{name} was neither a key in the metadata nor a column in the main data."
                        ) from err
            case _:
                return self.data[name]

    def __iter__(self):
        """Provide agenerator for iterating.

        Pass through to :py:meth:`DataFile.rows` for the actual work.

        Returns:
            Next row
        """
        for r in self.rows(False):
            yield r

    def __len__(self):
        """Return the length of the data.

        Returns: Returns the number of rows of data
        """
        if np.prod(self.data.shape) > 0:
            return np.shape(self.data)[0]
        return 0

    def __setitem__(self, name, value):
        """Handle direct setting of with metadata items or data array elements.

        Args:
            name (string, tuple):
                The string key used to access the metadata or a tuple index into data
            value (any):
                The value to be written into the metadata or data/

        Notes:
            If name is a string or already exists as key in metadata, this setitem will set metadata values,
            otherwise if name is a tuple then if the first elem,ent in a string it checks to see if that is an
            existing metadata item that is iterable, and if so, sets the metadta. In all other circumstances,
            it attempts to set an item in the main data array.
        """
        match name:
            case str():
                self.metadata[name] = value
            case _ if name in self.metadata:
                self.metadata[name] = value
            case (str(), *_) if name[0] in self.metadata and isiterable(self.metadata[name[0]]):
                if len(name) == 2:
                    key = name[0]
                    name = name[1]
                else:
                    key = name[0]
                    name = tuple(name[1:])
                self.metadata[key][name] = value
            case _:
                self.data[name] = value

    def count(self, value=None, axis=0, col=None):
        """Count the number of un-masked elements in the :py:class:`DataFile`.

        Keywords:
            valiue (float):
                Value to count for
            axis (int):
                Which axis to count the unmasked elements along.
            col (index, None):
                Restrict to counting in a specific column. If left None, then the current 'y' column is used.

        Returns:
            (int):
                Number of unmasked elements.
        """
        _ = self._col_args(ycol=col)
        if _.ycol is not None:
            tmp = self.column(_.ycol)
        else:
            tmp = self.data
        if value is not None:
            args = np.argwhere(tmp == value)
            return args.size
        return tmp.count(axis)

    def insert(self, index, obj):
        """Implement the insert method."""
        self.data = np.insert(self.data, index, obj, axis=0).view(type(self.data))
