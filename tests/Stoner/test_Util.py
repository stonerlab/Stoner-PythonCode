# -*- coding: utf-8 -*-
"""
Test_Util.py

Created on Mon Jul 18 14:13:39 2016

@author: phygbu"""


import unittest
import sys
import os.path as path
import os
import numpy as np
import re
from Stoner.compat import *
import Stoner.Util as SU

from Stoner import Data

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)

def is_2tuple(x):
    """Return tru if x is a length two tuple of floats."""
    return isinstance(x,tuple) and len(x)==2 and isinstance(x[0],float) and isinstance(x[1],float)


class Utils_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_hysteresis(self):
        """Test the hysteresis analysis code."""
        x=SU.hysteresis_correct(path.join(pth,"./sample-data/QD-SQUID-VSM.dat"),setas="3.xy")
        self.assertTrue("Hc" in x and "Area" in x and
                       "Hsat" in x and "BH_Max" in x and
                       "BH_Max_H" in x,"Hystersis loop analysis keys not present.")

        self.assertTrue(is_2tuple(x["Hc"]) and x["Hc"][0]+578<1.0,"Failed to find correct Hc in a SQUID loop")
        self.assertTrue(isinstance(x["Area"],float) and 0.0136<x["Area"]<0.0137,"Incorrect calculation of area under loop")

if __name__=="__main__": # Run some tests manually to allow debugging
    test=Utils_test("test_hysteresis")
    test.setUp()
    test.test_hysteresis()