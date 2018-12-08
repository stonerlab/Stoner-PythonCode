# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import unittest
import sys
import os, os.path as path
import numpy as np
import re
from numpy import any,all,sqrt,nan
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data,__home__,Options
from Stoner.Core import typeHintedDict

class Plottest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))

    def test_set_no_figs(self):
        self.assertTrue(Options.no_figs,"Default setting for no_figs option is incorrect.")
        Options.no_figs=True
        e=self.d.clone
        ret=e.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when no_figs is True  and showfig is not set({})".format(type(ret)))
        Options.no_figs=False
        e.showfig=False
        ret=e.plot()
        self.assertTrue(isinstance(ret,Data),"Return value of Data.plot() was not self when Data.showfig=False ({})".format(type(ret)))
        e.showfig=True
        ret=e.plot()
        self.assertTrue(isinstance(ret,Figure),"Return value of Data.plot() was not Figure when Data.showfig=False({})".format(type(ret)))
        e.showfig=None
        ret=e.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when Data.showfig is None ({})".format(type(ret)))
        Options.no_figs=True
        self.assertTrue(Options.no_figs,"set_option no_figs failed.")
        self.d=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        self.d.showfig=False
        ret=self.d.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when no_figs is True ({})".format(type(ret)))
        Options.no_figs=True
        plt.close("all")





if __name__=="__main__": # Run some tests manually to allow debugging
    test=Plottest("test_set_no_figs")
    test.setUp()
    unittest.main()
