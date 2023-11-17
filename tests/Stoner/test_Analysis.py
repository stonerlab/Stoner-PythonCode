# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

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


"""Path to sample Data File"""
slfdatadir = path.join(pth, "sample-data")

np.random.seed(12345)
slfd1 = Data(path.join(slfdatadir, "OVF1.ovf"))
slfd2 = Data(path.join(slfdatadir, "TDI_Format_RT.txt"))
slfd3 = Data(path.join(slfdatadir, "New-XRay-Data.dql"))
slfd4 = Data(np.column_stack([np.ones(100), np.ones(100) * 2]), setas="xy")


def test_functions():
    # Test section:
    _ = slfd1.section(z=(12, 13))
    f = slfd2.split(lambda r: r["Temp"] < 150)
    assert len(f[0]) == 838, "Split failed to work."
    assert len(slfd3.threshold(2000, rising=True, falling=True, all_vals=True)) == 5, "Threshold failure."


def test_peaks():
    d = slfd3.clone
    d.peaks(width=8, poly=4, significance=100, modify=True)
    assert len(d) == 11, "Failed on peaks test."


def test_threshold():
    # set up some zigzag data
    # mins at 0,100,200,300,400, max at 50, 150, 250, 350 and zeros in between
    ar = np.zeros((400, 2))
    ar[:, 0] = np.arange(0, len(ar))
    for i in range(4):
        ar[i * 100 : i * 100 + 50, 1] = np.linspace(-1, 1, 50)
    for i in range(4):
        ar[i * 100 + 50 : i * 100 + 100, 1] = np.linspace(1, -1, 50)
    d = Data(ar, setas="xy")
    assert len(d.threshold(0, rising=True, falling=False, all_vals=True) == 4)
    assert len(d.threshold(0, rising=False, falling=True, all_vals=True) == 4)
    assert len(d.threshold(0, interpolate=False, rising=False, falling=True, all_vals=True) == 4)
    assert d.threshold(0, all_vals=True)[1] == 124.5
    _ = d
    assert (
        np.sum(d.threshold([0.0, 0.5, 1.0]) - np.array([[24.5, 36.74999999, 49.0]])) < 1e-6
    ), "Multiple threshold failed."
    assert np.isclose(
        d.threshold(0, interpolate=False, all_vals=True)[1], 124.5, 6
    ), "Threshold without interpolation failed."
    result = d.threshold(0, interpolate=False, all_vals=True, xcol=False)
    assert np.allclose(
        result, np.array([[24.5, 0.0], [124.5, 0.0], [224.5, 0.0], [324.5, 0.0]])
    ), "Failed threshold with False scol - result was {}".format(result)


def test_apply():
    slfapp = Data(np.zeros((100, 1)), setas="y")
    slfapp.apply(lambda r: r.i[0], header="Counter")

    def calc(r, omega=1.0, k=1.0):
        return np.sin(r.y * omega)

    slfapp.apply(calc, replace=False, header="Sin", _extra={"omega": 0.1}, k=1.0)
    slfapp.apply(lambda r: r.__class__([r[1], r[0]]), replace=True, header=["Index", "Sin"])
    slfapp.setas = "xy"
    assert np.isclose(slfapp.integrate(output="result"), 18.87616564214), "Integrate after applies failed."


def test_scale():
    x = np.linspace(-5, 5, 101)
    y = np.sin(x)
    orig = Data(x + np.random.normal(size=101, scale=0.025), y + np.random.normal(size=101, scale=0.01))
    orig.setas = "xy"

    XTests = [[(0, 0, 0.5), (0, 2, -0.1)], [(0, 0, 0.5)], [(0, 2, -0.2)]]
    YTests = [[(1, 1, 0.5), (1, 2, -0.1)], [(1, 1, 0.5)], [(1, 2, -0.2)]]
    for xmode, xdata, xtests in zip(["linear", "scale", "offset"], [x * 2 + 0.2, x * 2, x + 0.2], XTests):
        for ymode, ydata, ytests in zip(["linear", "scale", "offset"], [y * 2 + 0.2, y * 2, y + 0.2], YTests):
            to_scale = Data(
                xdata + np.random.normal(size=101, scale=0.025), ydata + np.random.normal(size=101, scale=0.01)
            )
            to_scale.setas = "xy"
            to_scale.scale(orig, xmode=xmode, ymode=ymode)
            transform = to_scale["Transform"]
            t_err = to_scale["Transform Err"]
            for i, j, v in xtests + ytests:
                assert (
                    np.abs(transform[i, j] - v) <= 5 * t_err[i, j]
                ), "Failed to get correct trandorm factor for {}:{} ({} vs {}".format(
                    xmode, ymode, transform[i, j], v
                )

    to_scale = Data(
        x * 5 + 0.1 + np.random.normal(size=101, scale=0.025),
        y * 0.5 + 0.1 + 0.5 * x + np.random.normal(size=101, scale=0.01),
    )
    to_scale.setas = "xy"
    to_scale.scale(orig, xmode="affine")
    a_tranform = np.array([[0.2, 0.0, -0.02], [-0.2, 2.0, -0.17]])
    t_delta = np.abs(to_scale["Transform"] - a_tranform)
    t_in_range = t_delta < to_scale["Transform Err"] * 5
    assert np.all(t_in_range), "Failed to produce correct affine scaling {} vs {}".format(
        to_scale["Transform"], a_tranform
    )


def test_clip():
    x = np.linspace(0, np.pi * 10, 1001)
    y = np.sin(x)
    z = np.cos(x)
    d = Data(x, y, z, setas="xyz")
    d.clip((-0.1, 0.2), "Column 2")
    assert (d.z.min() >= -0.1) and (d.z.max() <= 0.2), "Clip with a column specified failed."
    d = Data(x, y, z, setas="xyz")
    d.clip((-0.5, 0.7))
    assert (d.y.min() >= -0.5) and (d.y.max() <= 0.7), "Clip with no column specified failed."


def test_integrate():
    d = Data(path.join(slfdatadir, "SLD_200919.dat"))
    d.setas = "x..y"
    d.integrate(result=True, header="Total_M")
    result = d["Total_M"]
    assert np.isclose(result, 4.19687459365, 7), "Integrate returned the wrong result!"
    d.setas[-1] = "y"
    d.plot(multiple="y2")
    assert len(d.axes) == 2, "Failed to produce plot with double y-axis"
    d.close("all")
    d.setas = "x..y"
    fx = d.interpolate(None)
    assert fx(np.linspace(1, 1500, 101)).shape == (101, 7), "Failed to get the interpolated shape right"


def test_sg_filter():
    x = np.linspace(0, 10 * np.pi, 1001)
    y = np.sin(x) + np.random.normal(size=1001, scale=0.05)
    d = Data(x, y, column_headers=["Time", "Signal"], setas="xy")
    d.SG_Filter(order=1, result=True)
    d.setas = "x.y"
    d.y = d.y - np.cos(x)
    assert np.isclose(d.y[5:-5].mean(), 0, atol=0.01), "Failed to differentiate correctly"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
