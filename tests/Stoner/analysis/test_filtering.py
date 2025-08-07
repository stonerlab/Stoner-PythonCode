# -*- coding: utf-8 -*-
"""Test for Stoner.analysis.filtering"""

import pytest
import numpy as np
from Stoner import Data
import warnings

warnings.filterwarnings("error")

testd = None
np.random.seed(12345)


def setup_function(function):
    global testd
    x = np.linspace(0, 2 * np.pi, 721)
    np.random.seed(87654321)
    y = np.sin(2 * x) + np.random.normal(scale=0.02, size=x.size)
    testd = Data(x, y, column_headers=["Time", "Ramp", "sine"])


def test_sgfilter():
    global testd
    testd.setas = ".y"
    testd.SG_Filter(result=True, header="Smooth")
    testd.setas = "xyy"
    res = testd // 1 - testd // 2
    assert res.mean() < 1e-4
    testd.del_column(2)
    testd.setas = ".y"
    testd.SG_Filter(result=True, header="Smooth", pad=0.0)
    testd.setas = "xyy"
    res = testd // 1 - testd // 2
    assert res.mean() < 1e-4
    testd.del_column(2)
    testd.SG_Filter(order=1, result=True, header="cos", pad=False)
    testd.setas = "xyz"
    testd.z /= 2
    testd.add_column(testd.z**2 + testd.y**2, header="sin*cos")
    testd.setas = "xyyy"
    testd.SG_Filter(col=3, order=0, points=51, replace=True, result=True, pad=1.0)
    testd.del_column(3)
    testd.setas = "xyyy"
    assert np.abs(testd // 3 - 1.0).max() < 0.075


def test_extrapolate():
    global testd
    testd.setas = "xy"
    testd.x -= 3
    with pytest.raises(TypeError):
        testd.extrapolate(kind=3.0)
    with pytest.raises(TypeError):
        testd.extrapolate(kind=lambda x, m, c: m * x + c)


def test_bins():
    global testd
    with pytest.raises(ValueError):
        testd.make_bins(0, 10, mode="bad")
    with pytest.raises(ValueError):
        testd.make_bins(0, 10.0, mode="log")
    with pytest.raises(ValueError):
        testd.make_bins(0, 0.5, mode="log")
    with pytest.raises(ValueError):
        testd.make_bins(0, 0.5, mode="bad")
    bins = np.linspace(testd.x.min(), testd.x.max(), 11)
    with pytest.raises(ValueError):
        testd.make_bins(0, bins=bins, mode="bad")

    b1 = testd.make_bins(0, bins=bins, mode="lin")
    for element,length in zip(b1,[11,10]):
        assert len(element) == length
    b2 = testd.make_bins(0, bins=bins, mode="log")
    for element,length in zip(b2,[11,10]):
        assert len(element) == length
    b3 = testd.make_bins(0, bins=10, mode="spacing")
    for element,length in zip(b3,[11,10]):
        assert len(element) == length
    with pytest.raises(TypeError):
        testd.make_bins(0, "10", mode="bad")
    with pytest.raises(ValueError):
        testd.make_bins(0, np.linspace(0, 6, 1000), mode="lin")


def test_outlier_detect():
    global testd
    testd.add_column(np.zeros_like(testd.x), header="zeros")
    testd.data[[90, 270, 450, 630], 1] = 0
    d1 = testd.clone
    with pytest.raises(ValueError):
        d1.outlier_detection(action="bad")

    d1.outlier_detection(certainty=20)
    assert d1.count() in (717, 716)
    assert not np.any(d1.mask[:, 2])
    d1 = testd.clone
    d1.outlier_detection(certainty=20, action="mask row")
    assert d1.count() == 717
    assert np.all(d1.mask[90, :] == [True, True, True])
    d1 = testd.clone
    d1.outlier_detection(certainty=20, action="delete")
    assert len(d1) in (717, 716)
    with pytest.raises(SyntaxError):
        d1.outlier_detection(action_args=(True, False))

    def action(i, column, data):
        data[i, column] = (data[i - 1, column] + data[i + 1, column]) / 2

    d1 = testd.clone
    d1.outlier_detection(certainty=20, action=action)
    d1.outlier_detection(certainty=20, action="mask row")
    assert not np.any(d1.mask)


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
