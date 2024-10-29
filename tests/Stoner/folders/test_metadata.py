#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:56:38 2018

@author: phygbu
"""
import sys
import os.path as path
import os
import numpy as np
from Stoner.compat import *
from Stoner import DataFolder
import pytest

from Stoner import Data
from pandas import DataFrame, Series


pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../../"))
sys.path.insert(0, pth)
datadir = path.join(pth, "sample-data")


def test_metadata_basic():
    os.chdir(datadir)
    fldr6 = DataFolder(".", pattern="QD*.dat", pruned=True)
    fldr6.sort()
    assert (
        repr(fldr6.metadata) == "The DataFolder . has 9 common keys of metadata in 4 Data objects"
    ), "Representation method of metadata wrong."
    assert len(fldr6.metadata) == 9, "Length of common metadata not right."
    assert list(fldr6.metadata.keys()) == [
        "Byapp",
        "Datatype,Comment",
        "Datatype,Time",
        "Fileopentime",
        "Loaded as",
        "Loaded from",
        "Startupaxis-X",
        "Startupaxis-Y1",
        "Stoner.class",
    ], "metadata.keys() not right."

    assert len(list(fldr6.metadata.all_keys())) == 49, "metadata.all_keys() the wrong length."


def test_metadata_slice():
    os.chdir(datadir)
    fldr6 = DataFolder(".", pattern="QD*.dat", pruned=True)
    fldr6.sort()
    assert isinstance(fldr6.metadata.slice("Loaded from")[0], dict), "metadata.slice not returtning a dictionary."
    assert isinstance(
        fldr6.metadata.slice("Loaded from", values_only=True), list
    ), "metadata.slice not returtning a list with values_only=True."
    assert isinstance(
        fldr6.metadata.slice("Loaded from", output="Data"), Data
    ), "metadata.slice not returtning Data with outpt='data'."
    for fmt, typ in zip(
        ["dict", "list", "array", "data", "frame", "smart"],
        [
            (list, dict, int),
            (list, tuple, int),
            (np.ndarray, np.ndarray, np.int64),
            (Data, np.ndarray, np.int64),
            (DataFrame, Series, np.int64),
            (list, dict, int),
        ],
    ):
        ret = fldr6.metadata.slice("Datatype,Comment", "Datatype,Time", output=fmt)
        for ix, t in enumerate(typ):
            assert isinstance(
                ret, t
            ), "Return from slice metadata for output={} and dimension {} had type {} and not {}".format(
                fmt, ix, type(ret), t
            )
            try:
                ret = ret[0]
            except KeyError:
                ret = ret[list(ret.keys())[0]]
            except (IndexError, TypeError):
                pass
    for k, typ in zip(
        ["Info.Sample_Holder", ("Info.Sample_Holder", "Datatype,Comment")],
        [(np.ndarray, str), (np.ndarray, np.ndarray)],
    ):
        ret = fldr6.metadata[k]
        for ix, t in enumerate(typ):
            assert isinstance(ret, t), "Indexing metadata for key={} and dimension {} had type {} and not {}".format(
                k, ix, type(ret), t
            )
            try:
                ret = ret[0]
            except KeyError:
                ret = ret[list(ret.keys())[0]]
            except AttributeError:
                ret = ret.data[0]
            except (IndexError, TypeError):
                pass
    del fldr6.metadata["Datatype,Comment"]
    with pytest.raises(KeyError):
        ret = fldr6.metadata["Datatype,Comment"]


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
