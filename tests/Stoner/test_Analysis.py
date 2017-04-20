# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import unittest
import sys
import os.path as path
import numpy as np
from numpy import any,all,sqrt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data

class Analysis_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d1=Data(path.join(self.datadir,"OVF1.ovf"))
        self.d2=Data(path.join(self.datadir,"TDI_Format_RT.txt"))
        self.d3=Data(path.join(self.datadir,"New-XRay-Data.dql"))
        self.d4=Data(np.column_stack([np.ones(100),np.ones(100)*2]),setas="xy")
        
    def test_functions(self):
        #Test section:
        self.s1=self.d1.section(z=(12,13))
        self.assertTrue(142.710<self.d2.mean("Temp")<142.711,"Failed on the mean test.")
        self.assertTrue(round(self.d2.span("Temp")[0],1)==4.3 and round(self.d2.span("Temp")[1],1)==291.6,"Span test failed.")
        f=self.d2.split("Temp",lambda x,r:x<150)
        self.assertTrue(len(f[0])==838,"Split failed to work.")
        self.assertEqual(len(self.d3.threshold(2000,rising=True,falling=True,all_vals=True)),5,"Threshold failure.")
        self.d4.add(0,1,"Add")
        self.d4.subtract(1,0,header="Subtract")
        self.d4.multiply(0,1,header="Multiply")
        self.d4.divide(0,1,header="Divide")
        self.d4.diffsum(0,1,header="Diffsum")
        self.assertTrue(np.all(self.d4[0]==np.array([-0.5,-1,-3,3,-1,2])),"Test column ops failed.")
        
    def test_peaks(self):
        d=self.d3.clone
        d.peaks(width=8,poly=4,significance=40,modify=True)
        self.assertEqual(len(d),10,"Failed on peaks test.")
    
if __name__=="__main__": # Run some tests manually to allow debugging
    test=Analysis_test("test_functions")
    test.setUp()
    test.test_functions()   
    test.test_peaks()
