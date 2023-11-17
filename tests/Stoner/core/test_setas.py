# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import sys
import os.path as path
import pytest
import numpy as np
from numpy import all

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../../"))
sys.path.insert(0, pth)

from Stoner import Data, __home__

datadir = path.join(pth, "sample-data")
selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")


def test_setas_basics():
    selfd4 = Data(path.join(__home__, "..", "sample-data", "Cu_resistivity_vs_T.txt"))
    selfd4.setas = "2.y.x3."
    assert selfd4.setas.x == 4, "Failed to set set as from string with numbers."
    tmp = list(selfd4.setas)
    selfd4.setas = "..y.x..."
    assert list(selfd4.setas) == tmp, "Explicit expansion of setas not the same as with numbers."
    selfd4.setas(x="T (K)")
    selfd4.setas(y="rho", reset=False)
    assert list(selfd4.setas) == tmp, "setas from calls with and without reset failed"
    s = selfd4.setas.clone
    selfd4.setas = []
    assert list(selfd4.setas) == ["." for i in range(8)], "Failed to clear setas"
    selfd4.setas(s)
    assert list(selfd4.setas) == tmp, "setas from call with setas failed"
    selfd4.setas(reset="Yes")
    assert list(selfd4.setas) == ["." for i in range(8)], "Failed to clear setas"
    assert selfd4.setas._size == 8, "Size attribute failed."
    assert selfd4[0, :].setas._size == 8, "Size attribute for array row failed."
    assert selfd4[:, 0].setas._size == 1, "Size attribute for array column  failed."
    selfd4.column_headers[-3:] = ["Column", "Column", "Column"]
    assert selfd4.setas._unique_headers == [
        "Voltage",
        "Current",
        "$\\rho$ ($\\Omega$m)",
        "Resistance",
        "T (K)",
        "Column",
        6,
        7,
    ], "Unique Headers failed in setas."
    selfd4.setas("2.y.x3.")
    s = selfd4.setas.clone
    selfd4.setas.clear()
    selfd4.setas(s)
    assert selfd4.setas == "..y.x...", "setas set by call with setas argument failed"
    selfd4.setas(selfd4.setas)
    assert selfd4.setas == "..y.x...", "setas set by call with self argument failed"
    selfd4.setas()
    assert selfd4.setas == "..y.x...", "setas set by call with no argument failed"

    selfd4.setas -= ["x", "y"]
    assert selfd4.setas == "8.", "setas __sub__ with iterable failed"

    selfd4.setas("2.y.x3.")
    assert selfd4.setas == "..y.x...", "Equality test by string failed."
    assert selfd4.setas == "2.y.x3.", "Equality test by numbered string failed."
    assert selfd4.setas.to_string() == "..y.x...", "To_string() failed."
    assert selfd4.setas.to_string(encode=True) == "..y.x3.", "To_string() failed."
    selfd4.setas.clear()
    selfd4.setas += "2.y.x3."
    assert selfd4.setas == "2.y.x3.", "Equality test by numbered string failed."
    selfd4.setas -= "2.y"
    assert selfd4.setas == "4.x3.", "Deletion Operator failed."
    selfd4.setas = "2.y.x3."
    assert selfd4.setas["rho"] == "y", "Getitem by type failed."
    assert selfd4.setas["x"] == "T (K)", "Getitem by name failed."
    assert selfd4.setas.x == 4, "setas x attribute failed."
    assert selfd4.setas.y == [2], "setas y attribute failed."
    assert selfd4.setas.z == [], "setas z attribute failed."
    assert all(
        selfd4.setas.set == np.array([False, False, True, False, True, False, False, False])
    ), "setas.set attribute not working."
    assert all(
        selfd4.setas.not_set == np.array([True, True, False, True, False, True, True, True])
    ), "setas.set attribute not working."
    assert "x" in selfd4.setas, "set.__contains__ failed"
    del selfd4.setas["x"]
    assert "x" not in selfd4.setas, "setas del item by column type failed."
    del selfd4.setas["rho"]
    assert "y" not in selfd4.setas, "setas del by column named failed"
    selfd4.setas({"T (K)": "x", "y": "rho"})
    assert selfd4.setas == "..y.x...", "Setting setas by call with dictionary failed"


def test_failures():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    with pytest.raises(AttributeError):
        selfd.setas.shape = (1, 2, 1)
    with pytest.raises(SyntaxError):
        selfd.setas(1.234)
    with pytest.raises(IndexError):
        selfd.setas({"i": 4})
    with pytest.raises(KeyError):
        selfd.setas({"x": "Wobble"})
    with pytest.raises(IndexError):
        selfd.setas({(2, 2): 1})
    with pytest.raises(IndexError):
        selfd.setas({"x": (1, 2)})
    with pytest.raises(ValueError):
        selfd.setas("ii")
    with pytest.raises(IndexError):
        selfd.setas[None]
    with pytest.raises(IndexError):
        selfd.setas[None] = None
    with pytest.raises(TypeError):
        selfd.setas + 3
    with pytest.raises(IndexError):
        selfd.setas + {"i": 2}
    with pytest.raises(TypeError):
        selfd.setas - 3.2
    with pytest.raises(IndexError):
        selfd.setas - {"i": 2}


def test_operators():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")

    selfd.setas = "xy"
    assert not selfd.setas == "."
    assert not selfd.setas == "yx"
    assert not selfd.setas == "xyz"
    dd = selfd
    assert selfd.setas == dd.setas
    assert not selfd.setas != "xy"
    assert str(selfd.setas) == "xy"


def test_setas_dict_interface():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd4 = Data(path.join(__home__, "..", "sample-data", "Cu_resistivity_vs_T.txt"))

    assert list(selfd.setas.items()) == [("X-Data", "x"), ("Y-Data", "y")], "Items method of setas failed."
    assert list(selfd.setas.keys()) == ["X-Data", "Y-Data"], "setas keys method fails."

    assert list(selfd.setas.values()) == ["x", "y"], "setas values method fails."
    assert selfd.setas.get("x") == "X-Data", "Simple get with key on setas failed."
    assert selfd.setas.get("e", "Help") == "Help", "get on setas with non-existent key failed."
    assert selfd.setas.get("X-Data") == "x", "get on setas by column named failed."
    assert selfd.setas.pop("x") == "X-Data", "pop on setas failed."
    assert selfd.setas == ".y", "Residual setas after pop wrong."
    selfd.setas.clear()
    assert selfd.setas == "", "setas clear failed."
    selfd.setas.update({"x": 0, "y": 1})
    assert selfd.setas.popitem() == ("x", "X-Data"), "Popitem method failed."
    assert selfd.setas, ".y" == "residual after popitem wrong on setas."
    assert selfd.setas.setdefault("x", "X-Data") == "X-Data", "setas setdefault failed."
    assert selfd.setas == "xy", "Result after set default wrong."
    selfd4.setas = "2.y.x3."
    assert selfd4.setas["#x"] == 4, "Indexing setas by #type failed."
    assert selfd4.setas[1::2] == [".", ".", ".", "."], "Indexing setas with a slice failed."
    assert selfd4.setas[[1, 3, 5, 7]] == [".", ".", ".", "."], "Indexing setas with a slice failed."
    selfd4.setas.clear()
    selfd4.setas += {"x": "T (K)", "rho": "y"}
    assert selfd4.setas == "2.y.x3.", "Adding dictionary with type:column and column:type to setas failed."
    assert selfd4.setas.pop("x") == "T (K)" and selfd4.setas == "2.y5.", "Pop from setas failed."
    assert not selfd4.setas.pop("z", False), "Pop with non existent key and default value failed"
    selfd4.setas.update({"x": "T (K)", "rho": "y"})
    assert selfd4.setas == "2.y.x3.", "setas.update failed."


def test_setas_metadata():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd2 = Data(path.join(__home__, "..", "sample-data", "TDI_Format_RT.txt"))
    selfd4 = Data(path.join(__home__, "..", "sample-data", "Cu_resistivity_vs_T.txt"))

    d2 = selfd2.clone
    d2.setas += {"x": 0, "y": 1}
    assert d2.setas.to_dict() == {
        "x": "Temperature",
        "y": "Resistance",
    }, f"__iadd__ failure {repr(d2.setas.to_dict())}"
    d3 = selfd2.clone
    s = d3.setas.clone
    assert d3.setas + s == s, "Addition operator to itself is not equal to itself in setas."
    assert d3.setas - s == "...", "Subtraction operator of setas on itself not equal to empty setas."
    d3.setas = "..e"
    d2.setas.update(d3.setas)
    assert d2.setas.to_dict() == {
        "x": "Temperature",
        "y": "Resistance",
        "e": "Column 2",
    }, f"__iadd__ failure {repr(d2.setas.to_dict())}"
    auto_setas = {
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
        "has_xcol": True,
        "has_xerr": False,
        "has_ycol": True,
        "has_yerr": True,
        "has_zcol": False,
        "has_zerr": False,
        "has_ucol": False,
        "has_vcol": False,
        "has_wcol": False,
        "has_axes": True,
        "has_uvw": False,
    }
    d2.setas = ""
    assert sorted(d2.setas._get_cols().items()) == sorted(auto_setas.items()), "Automatic guessing of setas failed!"
    d2.setas.clear()
    assert list(d2.setas) == ["."] * 3, "Failed to clear() setas"
    d2.setas[[0, 1, 2]] = "x", "y", "z"
    assert "".join(d2.setas) == "xyz", "Failed to set setas with a list"
    d2.setas[[0, 1, 2]] = "x"
    assert "".join(d2.setas) == "xxx", "Failed to set setas with an element from a list"
    d = selfd.clone
    d.setas = "xyz"
    assert repr(d.setas), "['x', 'y']" == "setas __repr__ failure {}".format(repr(d.setas))
    assert d.find_col(slice(5)) == [0, 1], f"findcol with a slice failed {d.find_col(slice(5))}"
    d = selfd2.clone
    d.setas = "xyz"
    assert d["Column 2", 5] == d[5, "Column 2"], "Indexing with mixed integer and string failed."
    assert selfd2.metadata.type(["User", "Timestamp"]) == [
        "String",
        "Timestamp",
    ], f"Metadata reading error {selfd2.metadata}"
    assert d.metadata.type(["User", "Timestamp"]) == ["String", "Timestamp"], "Metadata.type with slice failed"
    d.data["Column 2", :] = np.zeros(len(d))  # TODO make this work with d["Column 2",:] as well
    assert d.z.max() == 0.0 and d.z.min() == 0.0, "Failed to set Dataarray using string indexing"
    assert d.setas.x == 0 and d.setas.y == [1] and d.setas.z == [2]
    d.setas(x=1, y="Column 2")
    assert d.setas.x == 1 and d.setas.y == [2]
    selfd4.setas = "xyxy4."
    m1 = selfd4.setas._get_cols()
    m2 = selfd4.setas._get_cols(startx=2)
    assert "{xcol} {ycol}".format(**m1) == "0 [1]", f"setas._get_cols without startx failed.\n{m1}"
    assert "{xcol} {ycol}".format(**m2) == "2 [3]", f"setas._get_cols without startx failed.\n{m1}"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
