#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fitting mixin classes. Much of the code is also tested by the doc/samples scripts - so this is just
filling in some gaps.

Created on Sat Aug 24 22:56:59 2019

@author: phygbu
"""
import unittest
import sys
import os.path as path
import numpy as np


pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
sys.path.insert(0, pth)
from Stoner import Data

from Stoner.analysis.fitting.mixins import _curve_fit_result


def fit(x, a, b, c):
    """Fitting function"""
    return a * x**2 + b * x + c


class AnalysisMixins_test(unittest.TestCase):

    """Path to sample Data File"""

    datadir = path.join(pth, "sample-data")

    def setUp(self):
        x_data = np.linspace(-10, 10, 101)
        y_data = 0.01 * x_data**2 + 0.3 * x_data - 2

        y_data *= np.random.normal(size=101, loc=1.0, scale=0.01)
        x_data += np.random.normal(size=101, scale=0.02)

        self.data = Data(x_data, y_data, column_headers=["X", "Y"])
        self.data.setas = "xy"

    def test_cuve_fit(self):
        for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
            res = self.data.curve_fit(fit, p0=[0.02, 0.2, 2], output=output)
            self.assertTrue(
                isinstance(res, fmt),
                "Failed to get expected output from curve_fit for {} (got {})".format(output, type(res)),
            )

    def test_lmfit(self):
        for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
            res = self.data.lmfit(fit, p0=[0.02, 0.2, 2], output=output)
            self.assertTrue(
                isinstance(res, fmt),
                "Failed to get expected output from lmfit for {} (got {})".format(output, type(res)),
            )

    def test_odr(self):
        for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
            res = self.data.odr(fit, p0=[0.02, 0.2, 2], output=output)
            self.assertTrue(
                isinstance(res, fmt),
                "Failed to get expected output from idr for {} (got {})".format(output, type(res)),
            )

    def test_differential_evolution(self):
        for output, fmt in zip(["fit", "row", "full", "dict", "data"], [tuple, np.ndarray, tuple, dict, Data]):
            res = self.data.differential_evolution(fit, p0=[0.02, 0.2, 2], output=output)
            self.assertTrue(
                isinstance(res, fmt),
                "Failed to get expected output from differential_evolution for {} (got {})".format(output, type(res)),
            )


if __name__ == "__main__":  # Run some tests manually to allow debugging
    test = AnalysisMixins_test("test_cuve_fit")
    test.setUp()
    unittest.main()
    # test.test_apply()
