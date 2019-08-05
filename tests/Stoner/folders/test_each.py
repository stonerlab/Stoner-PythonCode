#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:59:16 2018

@author: phygbu
"""

import unittest
import sys
import os.path as path
import os
import numpy as np
import re
import fnmatch
from numpy import ceil
from Stoner.compat import *
import Stoner.Folders as SF

from Stoner import Data,set_option
import Stoner.HDF5, Stoner.Zip
from Stoner.Util import hysteresis_correct

import matplotlib.pyplot as plt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)

class folders_each_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_each(self):
        os.chdir(self.datadir)
        fldr6=SF.DataFolder(".",pattern="QD*.dat",pruned=True)
        print("X")
        fldr4=SF.DataFolder(self.datadir,pattern="QD-SQUID-VSM.dat")
        fldr5=fldr4.clone
        shaper=lambda f:f.shape
        fldr6.sort()
        res=fldr6.each(shaper)
        self.assertEqual(res,[(6048, 88), (3025, 41), (1409, 57), (411, 72)],"__call__ on each fauiled.")
        fldr6.each.del_column(0)
        res=fldr6.each(shaper)
        self.assertEqual(res,[(6048, 87), (3025, 40), (1409, 56), (411, 71)],"Proxy method call via each failed")
        paths=['QD-MH.dat', 'QD-PPMS.dat', 'QD-PPMS2.dat','QD-SQUID-VSM.dat']
        filenames=[path.relpath(x,start=fldr6.directory) for x in fldr6.each.filename.tolist()]
        self.assertEqual(filenames,paths,"Reading attributes from each failed.")
        if python_v3:
            eval('(hysteresis_correct@fldr4)(setas="3.xy",saturated_fraction=0.25)')
            self.assertTrue("Hc" in fldr4[0],"Matrix multiplication of callable by DataFolder failed test.")
        fldr5.each(hysteresis_correct,setas="3.xy",saturated_fraction=0.25)
        self.assertTrue("Hc" in fldr5[0],"Call on DataFolder.each() failed to apply function to folder")
        meths=[x for x in dir(fldr6.each) if not x.startswith("_")]
        self.assertEqual(len(meths),126 if python_v3 else 129,"Dir of folders.each failed ({}).".format(len(meths)))

    def test_attr_access(self):
        self.fldr=SF.PlotFolder(path.join(self.datadir,"NLIV"),pattern="*.txt",setas="yx")



if __name__=="__main__": # Run some tests manually to allow debugging
    test=folders_each_test("test_each")
    #unittest.main()
    test.test_attr_access()
