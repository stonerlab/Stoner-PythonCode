# -*- coding: utf-8 -*-
"""
:py:mod:`Stoner.folders.metadata` provides classes and functions to support the :py:attr:`Stoner.DataFolder.metadata` magic attribute.
"""
__all__ = ["proxy"]
from Stoner.compat import string_types
from Stoner.tools import islike_list
from Stoner.Core import DataFile
import numpy as _np_
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
            if _np_.all(self[key] == val):
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
        for d in self._folder:
            try:
                del d.metadata[item]
                ok = True
            except KeyError:
                pass
        if not ok:  # item was not a key in any data file
            raise KeyError("{} was not recognised as a metadata key in any object in the folder.".format(item))

    def __getitem__(self, value):
        """Return an array formed by getting a single key from each object in the Folder."""
        if value not in self:
            raise KeyError("{} is not a key of any object in the Folder.".format(value))
        if value in self.common_keys:
            return _np_.array([d[value] for d in self._folder])
        tmp = []
        mask = []
        typ = type(None)
        for d in self._folder:
            try:
                tmp.append(d[value])
                mask.append(False)
                typ = type(d[value])
            except KeyError:
                tmp.append(None)
                mask.append(True)
        ret = _np_.zeros(len(tmp), dtype=typ)
        ret = _np_.where(mask, ret, _np_.array(tmp)).astype(typ).view(_np_.ma.MaskedArray)
        ret.mask = mask
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

    def slice(self, key=None, values_only=False, output=None):  # pylint: disable=arguments-differ
        """Return a list of the metadata dictionaries for each item/file in the top level group

        Keyword Arguments:
            key(string or list of strings):
                if given then only return the item(s) requested from the metadata
            values_only(bool):
                if given amd *output* not set only return tuples of the dictionary values. Mostly useful
                when given a single key string
            output (str or type):
                Controls the output format from slice_metadata. Possible values are

                - "dict" or dict - return a list of dictionary subsets of the metadata from each image
                - "list" or list - return a list of values of each item pf the metadata
                - "array" or np.array - return a single array - like list above, but returns as a numpy array. This can create a 2D array from multiple keys
                - "Data" or Stoner.Data - returns the metadata in a Stoner.Data object where the column headers are the metadata keys.
                - "smart" - switch between *dict* and *list* depending whether there is one or more keys.

        Returns:
            ret(list of dict, tuple of values or :py:class:`Stoner.Data`):
                depending on *values_only* or (output* returns the sliced dictionaries or tuples/
                values of the items

        To do:
            this should probably be a func in baseFolder and should use have
            recursive options (build a dictionary of metadata values). And probably
            options to extract other parts of objects (first row or whatever).
        """
        if output is None:  # Sort out a definitive value of output
            output = "dict" if not values_only else "smart"
        if output not in [
            "dict",
            "list",
            "array",
            "Data",
            "smart",
            dict,
            list,
            _np_.ndarray,
            DataFile,
        ]:  # Check for good output value
            raise SyntaxError("output of slice metadata must be either dict, list, or array not {}".format(output))
        metadata = [
            k.metadata for k in self._folder
        ]  # this can take some time if it's loading in the contents of the folder
        if isinstance(key, string_types):  # Single key given
            key = metadata[0].__lookup__(key, multiple=True)
            key = [key] if not islike_list(key) else key
        # Expand all keys in case of multiple metadata matches
        newkey = []
        for k in key:
            newkey.extend(metadata[0].__lookup__(k, multiple=True))
        key = newkey
        if len(
            set(key) - set(self.common_keys)
        ):  # Is the key in the common keys? # TODO: implement __getitem__'s masked array logic?
            raise KeyError("{} are missing from some items in the Folder.".format(set(key) - set(self.common_keys)))
        results = []
        for i, met in enumerate(metadata):  # Assemble a list of dictionaries of values
            results.append({k: v for k, v in metadata[i].items() if k in key})
        if output in ["list", "array", "Data", list, _np_.ndarray, DataFile] or (
            output == "smart" and len(results[0]) == 1
        ):  # Reformat output
            cols = []
            for k in key:  # Expand the columns of data we're going to need if some values are not scalar
                if islike_list(metadata[0][k]):
                    for i, _ in enumerate(metadata[0][k]):
                        cols.append("{}_{}".format(k, i))
                else:
                    cols.append(k)

            for i, met in enumerate(results):  # For each object in the Folder
                results[i] = []
                for k in key:  # and for each key in out list
                    v = met[k]
                    if islike_list(
                        v
                    ):  # extend or append depending if the value is scalar. # TODO: This will blowup for values with more than 1 D!
                        results[i].extend(v)
                    else:
                        results[i].append(v)
                if output in ["aaray", "Data", _np_.ndarray, DataFile]:  # Convert each row to an array
                    results[i] = _np_.array(results[i])
            if len(cols) == 1:  # single key
                results = [m[0] for m in results]
            if output in ["array", _np_.ndarray]:
                results = _np_.array(results)
            if output in ["Data", DataFile]:  # Build oour Data object
                from Stoner import Data

                tmp = Data()
                tmp.data = _np_.array(results)
                tmp.column_headers = cols
                results = tmp
        return results
