#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:20:15 2018

@author: phygbu
"""

import pytest
import os.path as path
from Stoner.core.base import TypeHintedDict

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
# sys.path.insert(0,pth)


def test_ops():
    d = TypeHintedDict([("el1", 1), ("el2", 2), ("el3", 3), ("other", 4)])
    d.filter("el")
    assert len(d) == 3
    d.filter(lambda x: x.endswith("3"))
    assert len(d) == 1
    assert d["el3"] == 3
    d["munge"] = None
    assert d.types["munge"] == "Void", "Setting type for None value failed."
    d["munge"] = 1
    assert d["munge{String}"] == "1", "Munging return type in getitem failed."
    assert (
        repr(d)
        == """'el3':I32:3
'munge':I32:1"""
    ), "Repr failed \n{}".format(d)
    assert len(d | {"test": 2}) == len(d) + 1
    assert len(d | {"munge": 2}) == len(d)
    e = d.copy()
    e |= {"test": 4}
    assert d != e
    d.update({"test": 4})
    assert d == e


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
