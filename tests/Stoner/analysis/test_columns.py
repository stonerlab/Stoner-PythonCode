#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fitting mixin classes. Much of the code is also tested by the doc/samples scripts - so this is just
filling in some gaps.

Created on Sat Aug 24 22:56:59 2019

@author: phygbu
"""
import numpy as np
import pytest

from Stoner import Data

x = np.linspace(1, 10, 10)
y = 2 * x - 3
dy = np.abs(y / 100)
z = x + 4
dz = np.abs(z / 100)
selfd_master = Data(np.column_stack((x, y, dy, z, dz)), column_headers=["Tine", "Signal 1", "d1", "Signal 2", "d2"])


def test_add():
    selfd = selfd_master.clone
    selfd.add(1, 3, header="Add")
    assert np.all(selfd // "Add" == (selfd // "Signal 1" + selfd // "Signal 2")), "Failed to add column correctly"
    selfd.add((1, 2), (3, 4), header="Add", index="Add", replace=True)
    d_man = np.sqrt((selfd // "d1") ** 2 + (selfd // "d2") ** 2)
    assert np.allclose(selfd // -1, d_man), "Failed to calculate error in add"
    selfd.add(1, 3.0, index="Add", replace=True)
    assert np.all(selfd // 5 == selfd // 1 + 3), "Failed to add with a constant"
    selfd.add(1, np.ones(10), index=5, replace=True)
    assert np.all(selfd // 5 == selfd // 1 + 1), "Failed to add with a array"


def test_subtract():
    selfd = selfd_master.clone
    selfd.subtract(1, 3, header="Subtract")
    assert np.all(selfd // "Subtract" == (selfd // "Signal 1" - selfd // "Signal 2")), "Failed to add column correctly"
    selfd.subtract((1, 2), (3, 4), header="Subtract", index="Subtract", replace=True)
    d_man = np.sqrt((selfd // "d1") ** 2 + (selfd // "d2") ** 2)
    assert np.allclose(selfd // -1, d_man), "Failed to calculate error in add"
    selfd.subtract(1, 3.0, index="Subtract", replace=True)
    assert np.all(selfd // 5 == selfd // 1 - 3), "Failed to subtract with a constant"
    selfd.subtract(1, np.ones(10), index=5, replace=True)
    assert np.all(selfd // 5 == selfd // 1 - 1), "Failed to subtract with a array"


def test_multiply():
    selfd = selfd_master.clone
    selfd.multiply(1, 3, header="Multiply")
    assert np.all(
        (selfd // "Multiply") == ((selfd // "Signal 1") * (selfd // "Signal 2"))
    ), "Failed to add column correctly"
    selfd.multiply((1, 2), (3, 4), header="Multiply", index="Multiply", replace=True)
    d_man = np.sqrt(2) * 0.01 * np.abs(selfd // -2)
    assert np.allclose(selfd // -1, d_man), "Failed to calculate error in add"
    selfd.multiply(1, 3.0, index="Multiply", replace=True)
    assert np.all(selfd // 5 == (selfd // 1) * 3), "Failed to multiply with a constant"
    selfd.multiply(1, 2 * np.ones(10), index=5, replace=True)
    assert np.all(selfd // 5 == (selfd // 1) * 2), "Failed to multiply with a array"


def test_divide():
    selfd = selfd_master.clone
    selfd.divide(1, 3, header="Divide")
    assert np.all(
        (selfd // "Divide") == ((selfd // "Signal 1") / (selfd // "Signal 2"))
    ), "Failed to add column correctly"
    selfd.divide((1, 2), (3, 4), header="Divide", index="Divide", replace=True)
    d_man = np.sqrt(2) * 0.01 * np.abs(selfd // -2)
    assert np.allclose(selfd // -1, d_man), "Failed to calculate error in diffsum"
    selfd.divide(1, 3.0, index="Divide", replace=True)
    assert np.all(selfd // 5 == (selfd // 1) / 3), "Failed to add with a constant"
    selfd.divide(1, 2 * np.ones(10), index=5, replace=True)
    assert np.all(selfd // 5 == (selfd // 1) / 2), "Failed to add with a array"


def test_diffsum():
    selfd = selfd_master.clone
    selfd.diffsum(1, 3, header="Diffsum")
    a = selfd // 1
    b = selfd // 3
    man = (a - b) / (a + b)
    assert np.all((selfd // "Diffsum") == man), "Failed to diffsum column correctly"
    selfd.diffsum((1, 2), (3, 4), header="Diffsum", index="Diffsum", replace=True)


def test_limits():
    selfd = selfd_master.clone
    selfd.setas = "x..ye"
    assert selfd.min(1) == (-1, 0), "Minimum method failed"
    assert selfd.min() == (5.0, 0), "Minimum method failed"
    assert selfd.min(1, bounds=lambda r: r[0] > 2) == (3.0, 2.0), "Max with bounds failed"
    assert selfd.max() == (14, 9), "Max method failed"
    assert selfd.max(1) == (17, 9), "Max method failed"
    assert selfd.max(1, bounds=lambda r: r[0] < 5) == (5.0, 3.0), "Max with bounds failed"
    assert selfd.span() == (5.0, 14.0), "span method failed"
    assert selfd.span(1) == (-1.0, 17.0), "span method failed"
    assert selfd.span(1, bounds=lambda r: 2 < r.i < 8) == (5.0, 13.0), "span with bounds failed"


def test_stats():
    selfd = selfd_master.clone
    assert selfd.mean(1) == 8.0, "Simple channel mean failed"
    assert np.allclose(
        selfd.mean(1, sigma=2), (0.048990998729652346, 0.031144823004794875)
    ), "Channel mean with sigma failed"
    assert np.allclose(selfd.std(1), 6.0553007081949835), "Simple Standard Deviation failed"
    assert np.allclose(selfd.std(1, 2), 2.7067331877422456), "Simple Standard Deviation failed"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
