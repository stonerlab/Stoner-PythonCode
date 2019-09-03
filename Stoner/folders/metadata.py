# -*- coding: utf-8 -*-
"""
:py:mod:`Stoner.folders.metadata` provides classes and functions to support the :py:attr:`Stoner.DataFolder.metadata` magic attribute.
"""
__all__ = ["proxy"]
from Stoner.compat import string_types
from Stoner.tools import islike_list, isiterable, all_type
from Stoner.Core import DataFile
import numpy as np
from collections import MutableMapping
from Stoner.core import typeHintedDict


class proxy(MutableMapping):

    """Provide methods to interact with a whole collection of metadataObjects' metadata."""

    def __init__(self, folder):

        self._folder = folder

    @property
    def all(self):
        """A l,ist of all the metadata dictionaries in the Folder."""
        if hasattr(self._folder, "_metadata"):  # Extra logic for Folders like Stack
            for item in self._folder._metadata.items():
                yield item
        else:
            for item in self._folder:
                yield item.metadata

    @all.setter
    def all(self, value):
        """A l,ist of all the metadata dictionaries in the Folder."""
        if hasattr(self._folder, "_metadata"):  # Direct support for metadata dictionary
            for new, old in zip(value, self._folder._metadata):
                old.update(new)
        else:
            for new, item in zip(value, self._folder):
                item.metadata.update(new)

    @property
    def common_keys(self):
        """Return the set of metadata keys common to all objects int he Folder."""
        if len(self._folder) > 0:
            keys = set(self._folder[0].metadata.keys())
            for d in self._folder:
                keys &= set(d.metadata.keys())
        else:
            keys = set()
        return sorted(list(keys))

    @property
    def common_metadata(self):
        """Return a dictionary of the common_keys that have common values."""
        output = typeHintedDict()
        for key in self.common_keys:
            val = self._folder[0][key]
            if np.all(self[key] == val):
                output[key] = val
        return output

    def __contains__(self, item):
        """Check for membership of all possible kes."""
        return item in self.all_keys()

    def __iter__(self):
        """Iterate over objects."""
        for k in self.common_keys:
            yield k

    def __len__(self):
        """Out length is our common_keys."""
        return len(self.common_keys)

    def __repr__(self):
        """Give an informative dispaly of the metadata represenation."""
        return "The {} {} has {} common keys of metadata in {} {} objects".format(
            self._folder.__class__.__name__, self._folder.key, len(self), len(self._folder), self._folder.type.__name__
        )

    def __delitem__(self, item):
        """Attempt to delte item from all members of the folder."""
        ok = False
        for entry in self._folder:
            try:
                del entry.metadata[item]
                ok = True
            except KeyError:
                pass
        if not ok:  # item was not a key in any data file
            raise KeyError("{} was not recognised as a metadata key in any object in the folder.".format(item))

    def __getitem__(self, value):
        """Return an array formed by getting a single key from each object in the Folder."""
        ret = self.slice(value, mask_missing=True, output="array")
        if ret.size == 0:
            raise KeyError("{} did not match any keys in any file".format(value))
        return ret

    def __setitem__(self, key, value):
        for d in self._folder:
            d[key] = value

    def all_keys(self):
        """Return the union of all the metadata keyus for all objects int he Folder."""
        if len(self._folder) > 0:
            keys = set(self._folder[0].metadata.keys())
            for d in self._folder:
                keys |= set(d.metadata.keys())
        else:
            keys = set()
        for k in sorted(keys):
            yield k

    def all_items(self):
        """Return the result of indexing the metadata with all_keys().

        Yields:
            key,self[key]
        """
        for k in self.all_keys():
            yield k, self[k]

    def all_values(self):
        """Return the result of indexing the metadata with all_keys().

        Yields:
            self[key]
        """
        for k in self.all_keys():
            yield self[k]

    def apply(self, key, func):
        """Evaluate a function for each item in the folder and store the return value in a metadata key.

        Args:
            key (str): The name of the key to store the result in.
            func(callable): The function to be evaluated.

        Returns:
            (self) a copy of the combined metadata object to allow routines to be strung together.

        Notes:
            The function should have  a protoptye of the form:

                def func(i,metadataObject):

            where i is a counter that runs from 0 to the length of the current Folder
            and metadataObject will be each object in the Folder in turn.
        """
        for i, d in enumerate(self._folder):
            d[key] = func(i, d)
        return self

    def slice(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Return a list of the metadata dictionaries for each item/file in the top level group

        Keyword Arguments:
            *args (string or list of strings):
                if given then only return the item(s) requested from the metadata
            values_only(bool):
                if given and *output* not set only return tuples of the dictionary values. Mostly useful
                when given a single key string
            output (str or type):
                Controls the output format from slice_metadata. Possible values are

                - "dict" or dict - return a list of dictionary subsets of the metadata from each image
                - "list" or list - return a list of values of each item pf the metadata
                - "array" or np.array - return a single array - like list above, but returns as a numpy array. This can create a 2D array from multiple keys
                - "data" or Stoner.Data - returns the metadata in a Stoner.Data object where the column headers are the metadata keys.
                - "frame" - returns the metadata as a Pandas DataFrame object
                - "smart" - switch between *dict* and *list* depending whether there is one or more keys.
            mask_missing (bool):
                If true, then metadata entries missing in members of the folder are returned as masked values (or None), If
                False, then an exception is raised if any entries are missing.

        Returns:
            ret(list of dict, tuple of values or :py:class:`Stoner.Data`):
                depending on *values_only* or (output* returns the sliced dictionaries or tuples/
                values of the items

        To do:
            this should probably be a func in baseFolder and should use have
            recursive options (build a dictionary of metadata values). And probably
            options to extract other parts of objects (first row or whatever).
        """
        values_only = kwargs.pop("values_only", False)
        output = kwargs.pop("output", None)
        mask_missing = kwargs.pop("mask_missing", False)
        if kwargs:
            raise SyntaxError("Unused keyword arguments : {}".format(kwargs))
        if output is None:  # Sort out a definitive value of output
            output = "dict" if not values_only else "smart"
        if isinstance(output, string_types):
            output = output.lower()
        if output not in [
            "dict",
            "list",
            "array",
            "data",
            "frame",
            "smart",
            dict,
            list,
            np.ndarray,
            DataFile,
        ]:  # Check for good output value
            raise SyntaxError("output of slice metadata must be either dict, list, or array not {}".format(output))
        keys = []
        for k in args:
            if isinstance(k, string_types):
                keys.append(k)
            elif isiterable(k) and all_type(k, string_types):
                keys.extend(k)
            else:
                raise KeyError("{} cannot be used as a key name or set of key names".format(type(k)))
        if not mask_missing:
            for k in keys:
                if k not in self.common_keys:
                    raise KeyError("{} is not a key in all members of the folder".format(k))
        results = []
        for d in self._folder:
            results.append({k: d[k] for k in keys if k in d})

        for r in results:  # Expand the results where a result contains a list
            for k in keys:
                if k in r and islike_list(r[k]) and len(r[k]) > 0:
                    v = r[k]
                    del r[k]
                    r.update({"{}[{}]".format(k, i): vi for i, vi in enumerate(v)})

        if output == "smart":
            if np.all([len(r) == 1 and list(r.keys())[0] == list(results[0].keys())[0] for r in results]):
                output = "list"
            else:
                output = "dict"
        if output in ["list", list]:
            keys = set()
            for r in results:
                keys |= set(r.keys())
            keys = list(keys)
            if len(keys) == 1:
                ret = [r.get(keys[0], None) for r in results]
            else:
                ret = []
                for r in results:
                    ret.append(tuple(r.get(k, None) for k in keys))
        elif output == "dict":
            ret = results
        else:
            from pandas import DataFrame
            from Stoner import Data

            frame = DataFrame(results)
            mask = frame.isna()
            if output == "frame":
                ret = frame
            else:
                ret = Data(frame)
                ret.mask = mask
                if output in ["array", np.ndarray]:
                    ret = ret.data
        return ret
