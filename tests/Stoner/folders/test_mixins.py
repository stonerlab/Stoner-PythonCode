# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:33:08 2018

@author: phygbu
"""

import unittest
import sys
import os, os.path as path
import numpy as np
import re
from numpy import any,all,sqrt,nan

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)

from Stoner import Data,__home__,Options
from Stoner.Folders import PlotFolder
from Stoner.plot.formats import TexEngFormatter,DefaultPlotStyle
import matplotlib.pyplot as plt

def extra(i,j,d):
    """Cleanup plot window."""
    d.title="{},{}".format(i,j)
    d.xlabel("$V$")
    d.ylabel("$I$")

class folders_mixins_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.fldr=PlotFolder(path.join(self.datadir,"NLIV"),pattern="*.txt",setas="yx")
        self.fldr.template=DefaultPlotStyle()
        self.fldr.template.xformatter=TexEngFormatter
        self.fldr.template.yformatter=TexEngFormatter


    def test_plotting(self):
        Options.multiprocessing=False
        self.fldr.figure(figsize=(9,6))
        self.fldr.each.plot()
        self.assertEqual(len(plt.get_fignums()),1,"Plotting to a single figure in PlotFolder failed.")
        plt.close("all")
        self.fldr.plot(extra=extra)
        self.assertEqual(len(plt.get_fignums()),2,"Plotting to a single figure in PlotFolder failed.")
        self.ax=self.fldr[0].subplots
        self.assertEqual(len(self.ax),12,"Subplots check failed.")

        plt.close("all")
        Options.multiprocessing=True


if __name__=="__main__": # Run some tests manually to allow debugging
    test=folders_mixins_test("test_plotting")
    test.setUp()
    test.test_plotting()
    #unittest.main()