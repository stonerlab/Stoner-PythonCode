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
        f=self.d2.split(lambda r:r["Temp"]<150)
        self.assertTrue(len(f[0])==838,"Split failed to work.")
        self.assertEqual(len(self.d3.threshold(2000,rising=True,falling=True,all_vals=True)),5,"Threshold failure.")

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
        self.assertAlmostEqual(self.app.integrate(output="result"),18.87616564214,msg="Integrate after aplies failed.")

    def test_scale(self):
        x=np.linspace(-5,5,101)
        y=np.sin(x)
        orig=Data(x+np.random.normal(size=101,scale=0.025),y+np.random.normal(size=101,scale=0.01))
        orig.setas="xy"

        XTests=[[(0,0,0.5),(0,2,-0.1)],
                 [(0,0,0.5)],
                 [(0,2,-0.2)]]
        YTests=[[(1,1,0.5),(1,2,-0.1)],
                 [(1,1,0.5)],
                 [(1,2,-0.2)]]
        for xmode,xdata,xtests in zip(["linear","scale","offset"],[x*2+0.2,x*2,x+0.2],XTests):
            for ymode,ydata,ytests in zip(["linear","scale","offset"],[y*2+0.2,y*2,y+0.2],YTests):
                to_scale=Data(xdata+np.random.normal(size=101,scale=0.025),ydata+np.random.normal(size=101,scale=0.01))
                to_scale.setas="xy"
                to_scale.scale(orig,xmode=xmode,ymode=ymode)
                transform=to_scale["Transform"]
                t_err=to_scale["Transform Err"]
                for i,j,v in xtests+ytests:
                    self.assertLessEqual(np.abs(transform[i,j]-v),5*t_err[i,j],"Failed to get correct trandorm factor for {}:{} ({} vs {}".format(xmode,ymode,transform[i,j],v))

        to_scale=Data(x*5+0.1+np.random.normal(size=101,scale=0.025),y*0.5+0.1+0.5*x+np.random.normal(size=101,scale=0.01))
        to_scale.setas="xy"
        to_scale.scale(orig,xmode="affine")
        a_tranform=np.array([[0.2,0.,-0.02],[-0.2, 2.,-0.17]])
        t_delta=np.abs(to_scale["Transform"]-a_tranform)
        t_in_range=t_delta<to_scale["Transform Err"]*5
        self.assertTrue(np.all(t_in_range),"Failed to produce correct affine scaling {} vs {}".format(to_scale["Transform"],a_tranform))

    def test_clip(self):
        x=np.linspace(0,np.pi*10,1001)
        y=np.sin(x)
        z=np.cos(x)
        d=Data(x,y,z,setas="xyz")
        d.clip((-0.1,0.2),"Column 2")
        self.assertTrue((d.z.min()>=-0.1) and (d.z.max()<=0.2),"Clip with a column specified failed.")
        d=Data(x,y,z,setas="xyz")
        d.clip((-0.5,0.7))
        self.assertTrue((d.y.min()>=-0.5) and (d.y.max()<=0.7),"Clip with no column specified failed.")

    def test_integrate(self):
        d=Data(path.join(self.datadir,"SLD_200919.dat"))
        d.setas="x..y"
        d.integrate(result=True,header="Total_M")
        result=d["Total_M"]
        self.assertAlmostEqual(result,4.19687459365,7,"Integrate returned the wrong result!")
        d.setas[-1]="y"
        d.plot(multiple="y2")
        self.assertEqual(len(d.axes),2,"Failed to produce plot with double y-axis")
        d.close("all")
        d.setas="x..y"
        fx=d.interpolate(None)
        self.assertEqual(fx(np.linspace(1,1500,101)).shape,(101,7),"Failed to get the interpolated shape right")

    def test_sg_filter(self):
        x=np.linspace(0,10*np.pi,1001)
        y=np.sin(x)+np.random.normal(size=1001,scale=0.05)
        d=Data(x,y,column_headers=["Time","Signal"],setas="xy")
        d.SG_Filter(order=1,result=True)
        d.setas="x.y"
        d.y=d.y-np.cos(x)
        self.assertAlmostEqual(d.y[5:-5].mean(), 0,places=2,msg="Failed to differentiate correctly")


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Analysis_test("test_functions")
    test.setUp()
    #test.test_peaks()
    unittest.main()
    #test.test_integrate()
