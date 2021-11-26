# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:33:08 2018

@author: phygbu
"""

import pytest
import sys
import os, os.path as path

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)

from Stoner import Data,__home__,Options
from Stoner.folders import PlotFolder
from Stoner.plot.formats import TexEngFormatter,DefaultPlotStyle
import matplotlib.pyplot as plt

datadir=path.join(pth,"sample-data")

def extra(i,j,d):
    """Cleanup plot window."""
    d.title="{},{}".format(i,j)
    d.xlabel("$V$")
    d.ylabel("$I$")


def test_plotting():
    sfldr=PlotFolder(path.join(datadir,"NLIV"),pattern="*.txt",setas="yx")
    sfldr.template=DefaultPlotStyle()
    sfldr.template.xformatter=TexEngFormatter
    sfldr.template.yformatter=TexEngFormatter
    Options.multiprocessing=False
    sfldr.plots_per_page=len(sfldr)
    sfldr.plot(figsize=(18,12),title="{iterator}")
    assert len(plt.get_fignums())==1,"Plotting to a single figure in PlotFolder failed."
    sfldr.figure(figsize=(18,12))
    sfldr.plot(extra=extra)
    assert len(plt.get_fignums())==2,"Plotting to a single figure in PlotFolder failed."
    sax=sfldr[0].subplots
    assert len(sax)==16,"Subplots check failed."

    plt.close("all")
    Options.multiprocessing=False


if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb",__file__])