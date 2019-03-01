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
        f=self.d2.split(lambda r:r["Temp"]<150)
        self.assertTrue(len(f[0])==838,"Split failed to work.")
        self.assertEqual(len(self.d3.threshold(2000,rising=True,falling=True,all_vals=True)),5,"Threshold failure.")
        self.d4.add(0,1,"Add")
        self.d4.subtract(1,0,header="Subtract")
        self.d4.multiply(0,1,header="Multiply")
        self.d4.divide(0,1,header="Divide")
        self.d4.diffsum(0,1,header="Diffsum")
        self.assertTrue(np.all(self.d4[0]==np.array([-0.5,-1,-3,3,-1,2])),"Test column ops failed.")
        d=Data(np.zeros((100,1)))
        d.add(0,1.0)
        self.assertEqual(np.sum(d[:,0]),100.,"Add with a flot didn't work")
        d.add(0,np.ones(100))
        self.assertEqual(np.sum(d[:,0]),200.,"Add with an array failed.")

    def test_peaks(self):
        d=self.d3.clone
        d.peaks(width=8,poly=4,significance=100,modify=True)
        self.assertEqual(len(d),11,"Failed on peaks test.")

    def test_threshold(self):
        #set up some zigzag data
        #mins at 0,100,200,300,400, max at 50, 150, 250, 350 and zeros in between
        ar = np.zeros((400,2))
        ar[:,0]=np.arange(0,len(ar))
        for i in range(4):
            ar[i*100:i*100+50,1] = np.linspace(-1,1,50)
        for i in range(4):
            ar[i*100+50:i*100+100,1] = np.linspace(1,-1,50)
        d = Data(ar, setas='xy')
        self.assertTrue(len(d.threshold(0,rising=True,falling=False,all_vals=True)==4))
        self.assertTrue(len(d.threshold(0,rising=False,falling=True,all_vals=True)==4))
        self.assertTrue(len(d.threshold(0,interpolate=False,rising=False,falling=True,all_vals=True)==4))
        self.assertTrue(d.threshold(0,all_vals=True)[1]==124.5)
        self.thresh=d
        self.assertTrue(np.sum(d.threshold([0.0,0.5,1.0])-np.array([[24.5,36.74999999, 49.]]))<1E-6,"Multiple threshold failed.")
        self.assertAlmostEqual(d.threshold(0,interpolate=False,all_vals=True)[1],124.5,6,"Threshold without interpolation failed.")
        result=d.threshold(0,interpolate=False,all_vals=True,xcol=False)
        self.assertTrue(np.allclose(result,np.array([[ 24.5,   0. ],[124.5,   0. ],[224.5,   0. ],[324.5,   0. ]])),
                        "Failed threshold with False scol - result was {}".format(result))

    def test_apply(self):
        self.app=Data(np.zeros((100,1)),setas="y")
        self.app.apply(lambda r:r.i[0],header="Counter")
        def calc(r,omega=1.0,k=1.0):
            return np.sin(r.y*omega)
        self.app.apply(calc,replace=False,header="Sin",_extra={"omega":0.1},k=1.0)
        self.app.apply(lambda r:r.__class__([r[1],r[0]]),replace=True,header=["Index","Sin"])
        self.app.setas="xy"
        self.assertAlmostEqual(self.app.integrate(),-64.1722191259037,msg="Integrate after aplies failed.")



if __name__=="__main__": # Run some tests manually to allow debugging
    test=Analysis_test("test_functions")
    test.setUp()
    unittest.main()
    #test.test_apply()
