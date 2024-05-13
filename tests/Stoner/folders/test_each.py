# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:59:16 2018

@author: phygbu
"""

import sys
import os.path as path
import os
import pytest

from Stoner import DataFolder
from Stoner.Util import hysteresis_correct

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../../"))
sys.path.insert(0, pth)
datadir = path.join(pth, "sample-data")


def test_each_call():
    os.chdir(datadir)
    fldr6 = DataFolder(".", pattern="QD*.dat", pruned=True)
    shaper = lambda f: f.shape
    fldr6.sort()
    res = fldr6.each(shaper)
    assert res == [(6049, 88), (3026, 41), (1410, 57), (412, 72)], "__call__ on each fauiled."
    fldr6.each.del_column(0)
    res = fldr6.each(shaper)
    assert res == [(6049, 87), (3026, 40), (1410, 56), (412, 71)], "Proxy method call via each failed"
    paths = ["QD-MH.dat", "QD-PPMS.dat", "QD-PPMS2.dat", "QD-SQUID-VSM.dat"]
    filenames = [path.relpath(x, start=fldr6.directory) for x in fldr6.each.filename.tolist()]
    assert filenames == paths, "Reading attributes from each failed."
    meths = [x for x in dir(fldr6.each) if not x.startswith("_")]
    assert len(meths) == 138, "Dir of folders.each failed ({}).".format(len(meths))


def test_each_call_or_operator():
    os.chdir(datadir)
    fldr4 = DataFolder(datadir, pattern="QD-SQUID-VSM.dat")
    fldr5 = fldr4.clone
    (hysteresis_correct @ fldr4)(setas="3.xy", saturated_fraction=0.25)
    assert "Hc" in fldr4[0], "Matrix multiplication of callable by DataFolder failed test."
    fldr5.each(hysteresis_correct, setas="3.xy", saturated_fraction=0.25)
    assert "Hc" in fldr5[0], "Call on DataFolder.each() failed to apply function to folder"


def test_each_setas():
    os.chdir(datadir)
    fldr7 = DataFolder("NLIV", pattern="*.txt", pruned=True, setas="yx.")
    assert len(fldr7.each.setas) == 3, "Length of DataFolder.each.setas wrong"
    assert fldr7.each.setas.collapse() == ["y", "x", "."], "DataFolder collapsed each.setas wrong"
    del fldr7.each.setas[1]
    assert fldr7.each.setas.collapse() == ["y", ".", "."], "DataFolder.each.setas delitem failed"
    fldr7.each.setas[1] = "z"
    assert fldr7.each.setas[1] == ["z"] * len(fldr7), "DataFolder.each.setas setitem and getitem failed"
    fldr7.each.setas = "yx."
    assert fldr7.each.setas.collapse() == ["y", "x", "."], "DataFolder.each.setas assignment failed"


def test_each_attr():
    os.chdir(datadir)
    fldr6 = DataFolder(".", pattern="QD*.dat", pruned=True)
    with pytest.raises(AttributeError):
        _ = fldr6.each.bad_item
    fldr6.each.column_headers = ["X", "Y"]
    assert fldr6.each.column_headers.size == 4, "Read back of an attribute failed to return the right array length"
    res = [x[0] for x in fldr6.each.column_headers]
    assert res == ["X"] * 4, "Setting DataFolder.each.attr failed"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
