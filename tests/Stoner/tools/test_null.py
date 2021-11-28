# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:03:55 2021

@author: phygbu
"""
import pytest

from Stoner.tools import null

def test_null():
    assert null.null() is None

if __name__ == "__main__":
    pytest.main(["--pdb",__file__])
