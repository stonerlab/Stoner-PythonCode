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

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)

from Stoner import Data,__home__
from Stoner.Core import typeHintedDict

class SetasTest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(path.dirname(__file__),"..","CoreTest.dat"),setas="xy")
        self.d2=Data(path.join(__home__,"..","sample-data","TDI_Format_RT.txt"))
        self.d3=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        self.d4=Data(path.join(__home__,"..","sample-data","Cu_resistivity_vs_T.txt"))

    def test_setas_dict_interface(self):
        self.assertEqual(list(self.d.setas.items()),[('X-Data', 'x'), ('Y-Data', 'y')],"Items method of setas failed.")
        self.assertEqual(list(self.d.setas.keys()),["X-Data","Y-Data"],"setas keys method fails.")
        self.assertEqual(list(self.d.setas.values()),["x","y"],"setas values method fails.")
        self.assertEqual(self.d.setas.get("x"),"X-Data","Simple get with key on setas failed.")
        self.assertEqual(self.d.setas.get("e","Help"),"Help","get on setas with non-existant key failed.")
        self.assertEqual(self.d.setas.get("X-Data"),"x","get on setas by column named failed.")
        self.assertEqual(self.d.setas.pop("x"),"X-Data","pop on setas failed.")
        self.assertEqual(self.d.setas,".y","Residual setas after pop wrong.")
        self.d.setas.clear()
        self.assertEqual(self.d.setas,"","setas clear failed.")
        self.d.setas.update({"x":0,"y":1})
        self.assertEqual(self.d.setas.popitem(),("x","X-Data"),"Popitem method failed.")
        self.assertEqual(self.d.setas,".y","residual after popitem wrong on setas.")
        self.assertEqual(self.d.setas.setdefault("x","X-Data"),"X-Data","setas setdefault failed.")
        self.assertEqual(self.d.setas,"xy","Result after set default wrong.")

    def test_setas_metadata(self):
        d2=self.d2.clone
        d2.setas+={"x":0,"y":1}
        self.assertEqual(d2.setas.to_dict(),{'x': 'Temperature', 'y': 'Resistance'},"__iadd__ failure {}".format(repr(d2.setas.to_dict())))
        d3=self.d2.clone
        s=d3.setas.clone
        self.assertEqual(d3.setas+s,s,"Addition operator to itself is not equal to iself in setas.")
        self.assertEqual(d3.setas-s,"...","Subtraction operator of setas on itself not equal to empty setas.")
        d3.setas="..e"
        d2.setas.update(d3.setas)
        self.assertEqual(d2.setas.to_dict(),{'x': 'Temperature', 'y': 'Resistance', 'e': 'Column 2'},"__iadd__ failure {}".format(repr(d2.setas.to_dict())))
        auto_setas={ 'axes': 2,
                     'has_axes': True,
                     'has_ucol': False,
                     'has_uvw': False,
                     'has_vcol': False,
                     'has_wcol': False,
                     'has_xcol': True,
                     'has_xerr': False,
                     'has_ycol': True,
                     'has_yerr': True,
                     'has_zcol': False,
                     'has_zerr': False,
                     'ucol': None,
                     'vcol': None,
                     'wcol': None,
                     'xcol': 0,
                     'xerr': None,
                     'ycol': 1,
                     'yerr': 2,
                     'zcol': None,
                     'zerr': None}
        self.assertEqual(self.d2.setas._get_cols(),auto_setas,"Automatic guessing of setas failed!")
        d2.setas.clear()
        self.assertEqual(list(d2.setas),["."]*3,"Failed to clear() setas")
        d2.setas[[0,1,2]]="x","y","z"
        self.assertEqual("".join(d2.setas),"xyz","Failed to set setas with a list")
        d2.setas[[0,1,2]]="x"
        self.assertEqual("".join(d2.setas),"xxx","Failed to set setas with an element from a list")
        d=self.d.clone
        d.setas="xyz"
        self.assertEqual(repr(d.setas),"['x', 'y']","setas __repr__ failure {}".format(repr(d.setas)))
        self.assertEqual(d.find_col(slice(5)),[0,1],"findcol with a slice failed {}".format(d.find_col(slice(5))))
        d=self.d2.clone
        d.setas="xyz"
        self.assertTrue(d["Column 2",5]==d[5,"Column 2"],"Indexing with mixed integer and string failed.")
        self.assertEqual(d.metadata.type(["User","Timestamp"]),['String', 'String'],"Metadata.type with slice failed")
        d.data["Column 2",:]=np.zeros(len(d)) #TODO make this work with d["Column 2",:] as well
        self.assertTrue(d.z.max()==0.0 and d.z.min()==0.0,"Failed to set Dataarray using string indexing")
        self.assertTrue(d.setas.x==0 and d.setas.y==[1] and d.setas.z==[2])
        d.setas(x=1, y='Column 2')
        self.assertTrue(d.setas.x==1 and d.setas.y==[2])
        
if __name__=="__main__": # Run some tests manually to allow debugging
    test=SetasTest("test_setas_metadata")
    test.setUp()
    test.test_setas_metadata()
    #unittest.main()