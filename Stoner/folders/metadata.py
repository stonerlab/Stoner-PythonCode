# -*- coding: utf-8 -*-
"""Provides classes and functions to support the :py:attr:`Stoner.DataFolder.metadata` magic attribute."""

__all__ = ["MetadataProxy"]
import fnmatch
from collections.abc import MutableMapping

from lmfit import Model
import numpy as np

from ..core import TypeHintedDict, metadataObject
from ..compat import string_types
from ..tools import isLikeList, isiterable, make_Data
from ..Core import DataFile


def _fmt_as_list(results):
    """Convert the results list of slicing to a simple list."""
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

    return ret


def _fmt_as_dict(results):
    """Non-opt, results is already adictionary."""
    return results


def _fmt_as_dataframe(results):
    """Format the return results as a DataFrame."""
    from pandas import DataFrame

    frame = DataFrame(results)
    return frame


def _fmt_as_Data(results):
    """Format the results as a Data() object."""
    ret = make_Data(_fmt_as_dataframe(results))
    mask = np.zeros(ret.shape, dtype=bool)
    for ix, col in enumerate(ret.data.T):
        try:
            mask[:, ix] = np.isnan(col)
        except TypeError:
            pass
    ret.mask = mask
    return ret


def _fmt_as_array(results):
    """Format the results as an array."""
    ret = _fmt_as_Data(results)
    if ret.data.shape[1] != 1:
        ret = ret.data
    else:
        ret = ret.data[:, 0]
    return ret


def _fmt_as_smart(results):
    """Decide what formatting of results to do."""
    if np.all([len(r) == 1 and list(r.keys())[0] == list(results[0].keys())[0] for r in results]):
        return _fmt_as_list(results)
    return _fmt_as_dict(results)


def _slice_keys(args, possible=None):
    """Work through the arguments to slice() and construct a list of keys."""
    keys = []
    for k in args:
        if isinstance(k, string_types):
            if k not in possible:
                sub_k = fnmatch.filter(possible, k)
                if len(sub_k) > 0:
                    keys.extend(_slice_keys(sub_k, possible))
                else:
                    raise KeyError(f"No matching keys for {sub_k} in metadata")
            else:
                keys.append(k)
        elif isinstance(k, type) and issubclass(k, Model):
            model = k.__name__
            for name in k().param_names:
                for sub_k in [f"{model}:{name}", f"{model}:{name} err"]:
                    if sub_k not in possible:
                        raise KeyError(f"No matching keys for {sub_k} in metadata")
                    keys.append(sub_k)
        elif isinstance(k, Model):
            model = type(k).__name__
            for name in k.param_names:
                for sub_k in [f"{model}:{name}", f"{model}:{name} err"]:
                    if sub_k not in possible:
                        raise KeyError(f"No matching keys for {sub_k} in metadata")
                    keys.append(sub_k)
        elif isiterable(k):
            keys.extend(_slice_keys(k, possible))
        else:
            raise KeyError(f"{type(k)} cannot be used as a key name or set of key names")
    return keys


class MetadataProxy(MutableMapping):
    """Provide methods to interact with a whole collection of metadataObjects' metadata."""

    def __init__(self, folder):
        """Note our parent folder object."""
        self._folder = folder

    @property
    def all(self):
        """List all the metadata dictionaries in the Folder."""
        if hasattr(self._folder, "_metadata"):  # Extra logic for Folders like Stack
            for item in self._folder._metadata.items():
                yield item
        else:
            for item in self._folder:
                yield item.metadata

    @all.setter
    def all(self, value):
        """List all the metadata dictionaries in the Folder."""
        if hasattr(self._folder, "_metadata"):  # Direct support for metadata dictionary
            for new, old in zip(value, self._folder._metadata):
                old.update(new)
        else:
            for new, item in zip(value, self._folder):
                item.metadata.update(new)

    @property
    def all_by_keys(self):
        """Return the set of metadata keys common to all objects int he Folder."""
        if len(self._folder) > 0:
            keys = set(self._folder[0].metadata.keys())
            for d in self._folder:
                keys &= set(d.metadata.keys())
        else:
            keys = set()
        ret = TypeHintedDict()
        for k in sorted(list(keys)):
            ret[k] = self[k].view(np.ndarray)
        return ret

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
        output = TypeHintedDict()
        for key in self.common_keys:
            vals = self.slice(key, output="list")
            if np.all(vals == vals[0]):
                output[key] = vals[0]
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
        """Give an informative display of the metadata representation."""
        return (
            f"The {type(self._folder).__name__} {self._folder.key} has"
            + f" {len(self)} common keys of metadata in {len(self._folder)} {self._folder.type.__name__} objects"
        )

    def __delitem__(self, item):
        """Attempt to delete item from all members of the folder."""
        ok = False
        for entry in self._folder:
            try:
                del entry.metadata[item]
                ok = True
            except KeyError:
                pass
        if not ok:  # item was not a key in any data file
            raise KeyError(f"{item} was not recognised as a metadata key in any object in the folder.")

    def __getitem__(self, value):
        """Return an array formed by getting a single key from each object in the Folder."""
        ret = self.slice(value, mask_missing=True, output="array")
        if ret.size == 0:
            raise KeyError(f"{value} did not match any keys in any file")
        return ret

    def __setitem__(self, key, value):
        """Proxy to set an item on all the entries in the folder."""
        for d in self._folder:
            d[key] = value

    def __xor__(self, other):
        """Implement an XOR operator that gives differences between metadata dictionaries."""
        if isinstance(other, type(self._folder)):
            other = other.metadata
        if isinstance(other, MetadataProxy):
            other = other.all_by_keys
        elif isinstance(other, metadataObject):
            other = other.metadata
        else:
            return NotImplemented
        return self.all_by_keys ^ other

    def __eq__(self, other):
        """Equality test operator."""
        ret = self ^ other
        if not isinstance(ret, dict):
            return NotImplemented
        return len(ret) == 0

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
        """Return a list of the metadata dictionaries for each item/file in the top level group.

        Keyword Arguments:
            *args (string, lmfit.Model class or instance  or iterable of string, lmfit Models):
                if given then only return the item(s) requested from the metadata
            values_only(bool):
                if given and *output* not set only return tuples of the dictionary values. Mostly useful
                when given a single key string
            output (str or type):
                Controls the output format from slice_metadata. Possible values are

                - "dict" or dict - return a list of dictionary subsets of the metadata from each image
                - "list" or list - return a list of values of each item pf the metadata
                - "array" or np.array - return a single array - like list above, but returns as a numpy array.
                  This can create a 2D array from multiple keys
                - "data" or Stoner.Data - returns the metadata in a Stoner.Data object where the column headers
                  are the metadata keys.
                - "frame" - returns the metadata as a Pandas DataFrame object
                - "smart" - switch between *dict* and *list* depending whether there is one or more keys.
            mask_missing (bool):
                If true, then metadata entries missing in members of the folder are returned as masked values (or
                None), If False, then an exception is raised if any entries are missing.

        Returns:
            ret(list of dict, tuple of values or :py:class:`Stoner.Data`):
                depending on *values_only* or (output* returns the sliced dictionaries or tuples/
                values of the items

        """
        values_only = kwargs.pop("values_only", False)
        output = kwargs.pop("output", None)
        mask_missing = kwargs.pop("mask_missing", False)
        if kwargs:
            raise SyntaxError(f"Unused keyword arguments : {kwargs}")
        if output is None:  # Sort out a definitive value of output
            output = "dict" if not values_only else "smart"
        if isinstance(output, string_types):
            output = output.lower()
        outputs = {
            "list": _fmt_as_list,
            list: _fmt_as_list,
            "dict": _fmt_as_dict,
            dict: _fmt_as_dict,
            "frame": _fmt_as_dataframe,
            "data": _fmt_as_Data,
            DataFile: _fmt_as_Data,
            "array": _fmt_as_array,
            np.ndarray: _fmt_as_array,
            "smart": _fmt_as_smart,
        }
        if output not in outputs:  # Check for good output value
            raise TypeError(f"output of slice metadata must be either dict, list, or array not {output}")
        formatter = outputs[output]
        possible = list(self.all_keys()) if mask_missing else self.common_keys
        keys = _slice_keys(args, possible)
        results = []
        for d in self._folder:
            results.append({k: d[k] for k in keys if k in d})

        for r in results:  # Expand the results where a result contains a list
            for k in keys:
                if k in r and isLikeList(r[k]) and len(r[k]) > 0:
                    v = r[k]
                    del r[k]
                    r.update({f"{k}[{i}]": vi for i, vi in enumerate(v)})

        return formatter(results)
