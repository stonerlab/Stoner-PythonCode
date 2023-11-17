#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fitting mixin classes. Much of the code is also tested by the doc/samples scripts - so this is just
filling in some gaps.

Created on Sat Aug 24 22:56:59 2019

@author: phygbu
"""
import unittest
import numpy as np


from Stoner import Data


class ColumnOps_test(unittest.TestCase):
    def setUp(self):
        """Create a test data set."""
        x = np.linspace(1, 10, 10)
        y = 2 * x - 3
        dy = np.abs(y / 100)
        z = x + 4
        dz = np.abs(z / 100)
        self.data = Data(
            np.column_stack((x, y, dy, z, dz)), column_headers=["Tine", "Signal 1", "d1", "Signal 2", "d2"]
        )

    def test_add(self):
        self.data.add(1, 3, header="Add")
        self.assertTrue(
            np.all(self.data // "Add" == (self.data // "Signal 1" + self.data // "Signal 2")),
            "Failed to add column correctly",
        )
        self.data.add((1, 2), (3, 4), header="Add", index="Add", replace=True)
        d_man = np.sqrt((self.data // "d1") ** 2 + (self.data // "d2") ** 2)
        self.assertTrue(np.allclose(self.data // -1, d_man), "Failed to calculate error in add")
        self.data.add(1, 3.0, index="Add", replace=True)
        self.assertTrue(np.all(self.data // 5 == self.data // 1 + 3), "Failed to add with a constant")
        self.data.add(1, np.ones(10), index=5, replace=True)
        self.assertTrue(np.all(self.data // 5 == self.data // 1 + 1), "Failed to add with a array")

    def test_subtract(self):
        self.data.subtract(1, 3, header="Subtract")
        self.assertTrue(
            np.all(self.data // "Subtract" == (self.data // "Signal 1" - self.data // "Signal 2")),
            "Failed to add column correctly",
        )
        self.data.subtract((1, 2), (3, 4), header="Subtract", index="Subtract", replace=True)
        d_man = np.sqrt((self.data // "d1") ** 2 + (self.data // "d2") ** 2)
        self.assertTrue(np.allclose(self.data // -1, d_man), "Failed to calculate error in add")
        self.data.subtract(1, 3.0, index="Subtract", replace=True)
        self.assertTrue(np.all(self.data // 5 == self.data // 1 - 3), "Failed to subtract with a constant")
        self.data.subtract(1, np.ones(10), index=5, replace=True)
        self.assertTrue(np.all(self.data // 5 == self.data // 1 - 1), "Failed to subtract with a array")

    def test_multiply(self):
        self.data.multiply(1, 3, header="Multiply")
        self.assertTrue(
            np.all((self.data // "Multiply") == ((self.data // "Signal 1") * (self.data // "Signal 2"))),
            "Failed to add column correctly",
        )
        self.data.multiply((1, 2), (3, 4), header="Multiply", index="Multiply", replace=True)
        d_man = np.sqrt(2) * 0.01 * np.abs(self.data // -2)
        self.assertTrue(np.allclose(self.data // -1, d_man), "Failed to calculate error in add")
        self.data.multiply(1, 3.0, index="Multiply", replace=True)
        self.assertTrue(np.all(self.data // 5 == (self.data // 1) * 3), "Failed to multiply with a constant")
        self.data.multiply(1, 2 * np.ones(10), index=5, replace=True)
        self.assertTrue(np.all(self.data // 5 == (self.data // 1) * 2), "Failed to multiply with a array")

    def test_divide(self):
        self.data.divide(1, 3, header="Divide")
        self.assertTrue(
            np.all((self.data // "Divide") == ((self.data // "Signal 1") / (self.data // "Signal 2"))),
            "Failed to add column correctly",
        )
        self.data.divide((1, 2), (3, 4), header="Divide", index="Divide", replace=True)
        d_man = np.sqrt(2) * 0.01 * np.abs(self.data // -2)
        self.assertTrue(np.allclose(self.data // -1, d_man), "Failed to calculate error in diffsum")
        self.data.divide(1, 3.0, index="Divide", replace=True)
        self.assertTrue(np.all(self.data // 5 == (self.data // 1) / 3), "Failed to add with a constant")
        self.data.divide(1, 2 * np.ones(10), index=5, replace=True)
        self.assertTrue(np.all(self.data // 5 == (self.data // 1) / 2), "Failed to add with a array")

    def test_diffsum(self):
        self.data.diffsum(1, 3, header="Diffsum")
        a = self.data // 1
        b = self.data // 3
        man = (a - b) / (a + b)
        self.assertTrue(np.all((self.data // "Diffsum") == man), "Failed to diffsum column correctly")
        self.data.diffsum((1, 2), (3, 4), header="Diffsum", index="Diffsum", replace=True)

    def test_limits(self):
        self.data.setas = "x..ye"
        self.assertEqual(self.data.min(1), (-1, 0), "Minimum method failed")
        self.assertEqual(self.data.min(), (5.0, 0), "Minimum method failed")
        self.assertEqual(self.data.min(1, bounds=lambda r: r[0] > 2), (3.0, 2.0), "Max with bounds failed")
        self.assertEqual(self.data.max(), (14, 9), "Max method failed")
        self.assertEqual(self.data.max(1), (17, 9), "Max method failed")
        self.assertEqual(self.data.max(1, bounds=lambda r: r[0] < 5), (5.0, 3.0), "Max with bounds failed")
        self.assertEqual(self.data.span(), (5.0, 14.0), "span method failed")
        self.assertEqual(self.data.span(1), (-1.0, 17.0), "span method failed")
        self.assertEqual(self.data.span(1, bounds=lambda r: 2 < r.i < 8), (5.0, 13.0), "span with bounds failed")

    def test_stats(self):
        self.assertEqual(self.data.mean(1), 8.0, "Simple channel mean failed")
        self.assertTrue(
            np.allclose(self.data.mean(1, sigma=2), (0.048990998729652346, 0.031144823004794875)),
            "Channel mean with sigma failed",
        )
        self.assertAlmostEqual(self.data.std(1), 6.0553007081949835, msg="Simple Standard Deviation failed")
        self.assertAlmostEqual(self.data.std(1, 2), 2.7067331877422456, msg="Simple Standard Deviation failed")


if __name__ == "__main__":  # Run some tests manually to allow debugging
    test = ColumnOps_test("test_add")
    test.setUp()
    unittest.main()
    # test.test_add()
    # test.setUp()
    # test.test_subtract()
    # test.setUp()
    # test.test_multiply()
