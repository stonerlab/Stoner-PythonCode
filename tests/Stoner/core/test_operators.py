# -*- coding: utf-8 -*-
"""Teest stoner.core.operators module"""

import pytest
from os import path
import re
import numpy as np

from Stoner import Data, __home__

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
datadir = path.join(pth, "sample-data")

selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")


def test_floordiv_operators():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")

    assert all(selfd // 0 == selfd.column(0)), "Failed the // operator with integer"
    assert all(selfd // "X-Data" == selfd.column(0)), "Failed the // operator with string"
    assert all((selfd // re.compile(r"^X\-"))[:, 0] == selfd.column(0)), "Failed the // operator with regexp"
    with pytest.raises(TypeError):
        selfd + 2.2


def test_eq_operator():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd2 = selfd.clone
    selfd3 = selfd.clone
    selfd3.data = selfd3.data[:, 1]
    assert selfd == selfd2
    assert not selfd == 1
    assert not selfd3 == selfd
    selfd3 = selfd.clone
    selfd3.column_headers[0] = "Different"
    assert not selfd == selfd3
    selfd3 = selfd.clone
    selfd3["Extra"] = False
    assert not selfd == selfd3


def test_modulo_operators():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    t = [selfd % 1, selfd % "Y-Data", (selfd % re.compile(r"Y\-"))]
    for ix, tst in enumerate(["integer", "string", "regexp"]):
        assert all(t[ix].data == np.atleast_2d(selfd.column(0)).T), f"Failed % operator with {tst} index"
    d = selfd % 1
    assert d.shape[1] == 1, "Column division failed"
    assert len(d.setas) == d.shape[1], "Column removal messed up setas"
    assert len(d.column_headers) == d.shape[1], "Column removal messed up column headers"


def test_and_operator():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd2 = Data(path.join(__home__, "..", "sample-data", "TDI_Format_RT.txt"))
    d = selfd & selfd.x
    assert d.shape[1] == 3, "& operator failed."
    assert len(d.setas) == 3, "& messed up setas"
    assert len(d.column_headers) == 3, "& messed up setas"
    d &= d.x
    assert d.shape[1] == 4, "Inplace & operator failed"
    e = selfd.clone
    f = selfd2.clone
    g = e & f
    h = f & e
    assert g.shape == h.shape, "Anding unequal column lengths faile!"


def test_add_operator():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd2 = Data(path.join(__home__, "..", "sample-data", "TDI_Format_RT.txt"))
    empty = Data()
    empty = empty + np.zeros(10)
    assert empty.shape == (1, 10), "Adding to an empty array failed"
    d = selfd + np.array([0, 0])
    assert len(d) == len(selfd) + 1, "+ operator failed."
    d += np.array([0, 0])
    assert len(d) == len(selfd) + 2, "Inplace + operator failed."
    e = selfd.clone
    f = selfd2.clone
    g = e + f
    assert g.shape == (1776, 2), "Add of 2 column and 3 column failed"
    g = f + e
    assert g.shape == (1776, 3), "Add of 3 column and 2 column failed."
    f = selfd2.clone
    f += {"Res": 0.3, "Temp": 1.2, "New_Column": 25}
    f1, f2 = f.shape
    assert f1 == selfd2.shape[0] + 1, "Failed to add a row by providing a dictionary"
    assert f2 == selfd2.shape[1] + 1, "Failed to add an extra columns by adding a dictionary with a new key"
    assert np.isnan(f[-1, 2]) and np.isnan(f[0, -1]), "Unset values when adding a dictionary not NaN"
    d = Data()
    d += np.ones(5)
    d += np.zeros(5)
    assert np.all(d // 1 == np.array([1, 0])), "Adding 1D arrays should add new rows"
    with pytest.raises(TypeError):
        d += True


def test_delete_operator():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd2 = Data(path.join(__home__, "..", "sample-data", "TDI_Format_RT.txt"))
    d = selfd + np.array([0, 0])
    d += np.array([0, 0])
    d = d - (-1)
    assert len(d) == len(selfd) + 1, f"Delete with integer index failed. {len(d)} vs {len(selfd)+1}"
    d -= -1
    assert len(d) == len(selfd), f"Inplace delete with integer index failed. {len(d)} vs {len(selfd)}"
    d -= slice(0, -1, 2)
    assert len(d) == len(selfd) / 2, f"Inplace delete with slice index failed. {len(d)} vs {len(selfd)/2}"

    e = selfd.clone
    f = selfd2.clone
    g = e + f
    assert g.shape == (1776, 2), "Add of 2 column and 3 column failed"
    g = f + e
    assert g.shape == (1776, 3), "Add of 3 column and 2 column failed."
    g = e & f
    h = f & e
    assert g.shape == h.shape, "Anding unequal column lengths faile!"


def test_invert_operator():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    e = ~selfd
    assert e.setas[0] == "y", "failed to invert setas columns"
    d = Data(np.array([[1, 2, 3, 4]]), setas="xdyz")
    assert (~d).setas == ["y", "e", "z", "x"], "Inverting multi-axis array failed"
    with pytest.raises(TypeError):
        Data() << 45.3


def test_lshoft_operator():
    with open(selfd.filename, "r") as data:
        lines = data.readlines()
    e = Data() << lines
    e.setas = selfd.setas
    e["Loaded as"] = selfd["Loaded as"]
    assert e == selfd, "Failed to read and create from << operator"
    with open(selfd.filename, "r") as data:
        e = Data() << data
    e.setas = selfd.setas
    e["Loaded as"] = selfd["Loaded as"]
    assert e == selfd, "Failed to read and create from << operator"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
