# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:33:08 2018

@author: phygbu
"""

import sys
import os.path as path
import pytest


pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../../"))
sys.path.insert(0, pth)

from Stoner import Data, __home__, Options
from Stoner.Folders import PlotFolder
from Stoner.plot.formats import TexEngFormatter, DefaultPlotStyle
import matplotlib.pyplot as plt


def extra(i, j, d):
    """Cleanup plot window."""
    d.title = "{},{}".format(i, j)
    d.xlabel("$V$")
    d.ylabel("$I$")


datadir = path.join(pth, "sample-data")


def test_plotting():
    selffldr = PlotFolder(path.join(datadir, "NLIV"), pattern="*.txt", setas="yx")
    Options.multiprocessing = False
    selffldr.plots_per_page = len(selffldr)
    selffldr.plot(figsize=(18, 12), title="{iterator}")
    assert len(plt.get_fignums()) == 1, "Plotting to a single figure in PlotFolder failed."
    selffldr.plot(extra=extra)
    assert len(plt.get_fignums()) == 2, "Plotting to a single figure in PlotFolder failed."
    selfax = selffldr[0].subplots
    assert len(selfax) == 16, "Subplots check failed."

    plt.close("all")
    Options.multiprocessing = False


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
