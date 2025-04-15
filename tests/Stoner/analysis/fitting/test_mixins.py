#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fitting mixin classes. Much of the code is also tested by the doc/samples scripts - so this is just
filling in some gaps.

Created on Sat Aug 24 22:56:59 2019

@author: phygbu
"""
import pytest
import sys
import os.path as path
import numpy as np


pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
sys.path.insert(0, pth)
from Stoner import Data

from Stoner.analysis.fitting.functions import _curve_fit_result


def fit(x, a, b, c):
    """Fitting function"""
    return a * x**2 + b * x + c


"""Path to sample Data File"""

datadir = path.join(pth, "sample-data")



x_data = np.linspace(-10, 10, 101)
y_data = 0.01 * x_data**2 + 0.3 * x_data - 2

y_data *= np.random.normal(size=101, loc=1.0, scale=0.01)
x_data += np.random.normal(size=101, scale=0.02)

selfd = Data(x_data, y_data, column_headers=["X", "Y"])
selfd.setas = "xy"


def test_cuve_fit():
    for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
        res = selfd.curve_fit(fit, p0=[0.02, 0.2, 2], output=output)
        assert isinstance(res, fmt), f"Failed to get expected output from curve_fit for {output} (got {type(res)})"


def test_lmfit():
    for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
        res = selfd.lmfit(fit, p0=[0.02, 0.2, 2], output=output)
        assert isinstance(res, fmt), f"Failed to get expected output from curve_fit for {output} (got {type(res)})"


def test_odr():
    for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
        res = selfd.odr(fit, p0=[0.02, 0.2, 2], output=output)
        assert isinstance(res, fmt), f"Failed to get expected output from curve_fit for {output} (got {type(res)})"


def test_differential_evolution():
    for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
        res = selfd.differential_evolution(fit, p0=[0.02, 0.2, 2], output=output)
        assert isinstance(res, fmt), f"Failed to get expected output from curve_fit for {output} (got {type(res)})"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
