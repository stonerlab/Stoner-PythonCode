# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import pytest
import sys
import  os.path as path
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../../"))
sys.path.insert(0, pth)
from Stoner import Data, __home__, Options
from Stoner.Core import TypeHintedDict
from Stoner.plot.core import hsl2rgb
from Stoner.Image import ImageFile

from Stoner.plot.formats import DefaultPlotStyle

datadir = path.join(pth, "sample-data")

selfd = Data(path.join(__home__, "..", "sample-data", "New-XRay-Data.dql"))


warnings.filterwarnings("error")

def test_set_no_figs():
    global selfd
    assert Options.no_figs, "Default setting for no_figs option is incorrect."
    Options.no_figs = True
    e = selfd.clone
    ret = e.plot()
    assert ret is None, "Output of Data.plot() was not None when no_figs is True  and showfig is not set({})".format(
        type(ret)
    )
    Options.no_figs = False
    e.showfig = False
    ret = e.plot()
    assert isinstance(ret, Data), "Return value of Data.plot() was not self when Data.showfig=False ({})".format(
        type(ret)
    )
    e.showfig = True
    ret = e.plot()
    assert isinstance(ret, Figure), "Return value of Data.plot() was not Figure when Data.showfig=False({})".format(
        type(ret)
    )
    e.showfig = None
    ret = e.plot()
    assert ret is None, "Output of Data.plot() was not None when Data.showfig is None ({})".format(type(ret))
    Options.no_figs = True
    assert Options.no_figs, "set_option no_figs failed."
    selfd = Data(path.join(__home__, "..", "sample-data", "New-XRay-Data.dql"))
    selfd.showfig = False
    ret = selfd.plot()
    assert ret is None, "Output of Data.plot() was not None when no_figs is True ({})".format(type(ret))
    Options.no_figs = True
    plt.close("all")


@pytest.mark.filterwarnings("ignore:.*:matplotlib.MatplotlibDeprecationWarning")
def test_template_settings():
    template = DefaultPlotStyle(font__weight="bold")
    assert template["font.weight"] == "bold", "Setting ytemplate parameter in init failed."
    template(font__weight="normal")
    assert template["font.weight"] == "normal", "Setting ytemplate parameter in call failed."
    template["font.weight"] = "bold"
    assert template["font.weight"] == "bold", "Setting ytemplate parameter in setitem failed."
    del template["font.weight"]
    assert template["font.weight"] == "normal", "Resettting template parameter failed."
    keys = sorted([x for x in template])
    assert sorted(template.keys()) == keys, "template.keys() and template.iter() disagree."
    attrs = [x for x in dir(template) if template._allowed_attr(x)]
    length = len(dict(plt.rcParams)) + len(attrs)
    assert len(template) == length, "templa length wrong."


def test_plot_magic():
    selfd.figure()
    dpi = selfd.fig_dpi
    selfd.fig_dpi = dpi * 2
    assert selfd.fig.dpi == dpi * 2, "Failed to get/set attributes on current figure"
    vis = selfd.fig_visible
    selfd.fig_visible = not vis
    assert not selfd.fig_visible, "Setting/Getting figure.visible failed"
    plt.close("all")
    plt.figure()
    fn = plt.get_fignums()[0]
    selfd.fig = fn
    selfd.plot()
    assert len(plt.get_fignums()) == 1, "Setting Data.fig by integer failed."
    plt.close("all")
    selfd.plot(plotter=plt.semilogy)
    assert selfd.ax_lines[0].get_c() == "k", "Auto formatting of plot failed"
    selfd.plot(figure=False)
    selfd.plot(figure=1)
    assert len(plt.get_fignums()) == 2, "Plotting setting figure failed"
    assert len(selfd.ax_lines) == 2, "Plotting setting figure failed"
    selfd.figure(2)
    selfd.plot()
    selfd.plot(figure=True)
    assert len(plt.get_fignums()) == 2, "Plotting setting figure failed"
    assert len(selfd.ax_lines) == 3, "Plotting setting figure failed"
    plt.close("all")


def test_extra_plots():
    x = np.random.uniform(-np.pi, np.pi, size=5001)
    y = np.random.uniform(-np.pi, np.pi, size=5001)
    z = (np.cos(4 * np.sqrt(x**2 + y**2)) * np.exp(-np.sqrt(x**2 + y**2) / 3.0)) ** 2
    selfd2 = Data(x, y, z, column_headers=["X", "Y", "Z"], setas="xyz")
    selfd2.contour_xyz(projection="2d")  #
    assert len(plt.get_fignums()) == 1, "Setting Data.fig by integer failed."
    plt.close("all")
    X, Y, Z = selfd2.griddata(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
    plt.imshow(Z)
    assert len(plt.get_fignums()) == 1, "Setting Data.fig by integer failed."
    plt.imshow(Z)
    plt.close("all")
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 10), np.linspace(-np.pi, np.pi, 10))
    z = np.zeros_like(x)
    w = np.cos(np.sqrt(x**2 + y**2))
    q = np.arctan2(x, y)
    u = np.abs(w) * np.cos(q)
    v = np.abs(w) * np.sin(q)
    selfd3 = Data(x.ravel(), y.ravel(), z.ravel(), u.ravel(), v.ravel(), w.ravel(), setas="xyzuvw")
    selfd3.plot()
    assert len(plt.get_fignums()) == 1, "Setting Data.fig by integer failed."
    plt.close("all")
    # i=ImageFile(path.join(__home__,"..","sample-data","Kermit.png"))
    # selfd3=Data(i)
    # selfd3.data=i.data
    # selfd3.plot_matrix()


def test_misc_funcs():
    assert np.all(hsl2rgb(0.5, 0.5, 0.5) == np.array([[63, 191, 191]]))
    selfd.load(selfd.filename, Q=True)
    selfd.plot()
    selfd.x2()
    selfd.setas = ".yx"
    selfd.plot()
    assert len(selfd.fig_axes) == 2, "Creating a second X axis failed"
    plt.close("all")
    for i in range(4):
        selfd.subplot(2, 2, i + 1)
        selfd.plot()
    assert len(selfd.fig_axes) == 4, "Creating subplots failed"
    selfd.close()


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
