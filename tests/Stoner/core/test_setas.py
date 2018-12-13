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
from Stoner.tools import isiterable
from Stoner.Core import typeHintedDict

class SetasTest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(path.dirname(__file__),"..","CoreTest.dat"),setas="xy")
        self.d2=Data(path.join(__home__,"..","sample-data","TDI_Format_RT.txt"))
        self.d3=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        self.d4=Data(path.join(__home__,"..","sample-data","Cu_resistivity_vs_T.txt"))

    def test_setas_basics(self):
        self.d4.setas="2.y.x3."
        self.assertEqual(self.d4.setas.x,4,"Failed to set set as from string with numbers.")
        tmp=list(self.d4.setas)
        self.d4.setas="..y.x..."
        self.assertEqual(list(self.d4.setas),tmp,"Explicit exapnsion of setas not the same as with numbers.")
        self.d4.setas(x="T (K)")
        self.d4.setas(y="rho",reset=False)
        self.assertEqual(list(self.d4.setas),tmp,"setas from calls with and without reset failed")
        s=self.d4.setas.clone
        self.d4.setas=[]
        self.assertEqual(list(self.d4.setas),["." for i in range(8)],"Failed to clear setas")
        self.d4.setas(s)
        self.assertEqual(list(self.d4.setas),tmp,"setas from call with setas failed")
        self.d4.setas(reset="Yes")
        self.assertEqual(list(self.d4.setas),["." for i in range(8)],"Failed to clear setas")
        self.assertEqual(self.d4.setas._size,8,"Size attribute failed.")
        self.assertEqual(self.d4[0,:].setas._size,8,"Size attribute for array row failed.")
        self.assertEqual(self.d4[:,0].setas._size,1,"Size attribute for array column  failed.")
        self.d4.column_headers[-3:]=["Column","Column","Column"]
        self.assertEqual(self.d4.setas._unique_headers,['Voltage', 'Current', '$\\rho$ ($\\Omega$m)', 'Resistance', 'T (K)', 'Column', 6, 7],
                                                        "Unique Headers failed in setas.")
        self.d4.setas("2.y.x3.")
        s=self.d4.setas.clone
        self.d4.setas.clear()
        self.d4.setas(s)
        self.assertTrue(self.d4.setas=="..y.x...","setas set by call with setas argument failed")
        self.d4.setas(self.d4.setas)
        self.assertTrue(self.d4.setas=="..y.x...","setas set by call with self argument failed")
        self.d4.setas()
        self.assertTrue(self.d4.setas=="..y.x...","setas set by call with no argument failed")

        self.d4.setas-=["x","y"]
        self.assertTrue(self.d4.setas=="8.","setas __sub__ with iterable failed")

        self.d4.setas("2.y.x3.")
        self.assertTrue(self.d4.setas=="..y.x...","Equality test by string failed.")
        self.assertTrue(self.d4.setas=="2.y.x3.","Equality test by numbered string failed.")
        self.assertEqual(self.d4.setas.to_string(),"..y.x...","To_string() failed.")
        self.assertEqual(self.d4.setas.to_string(encode=True),"..y.x3.","To_string() failed.")
        self.d4.setas.clear()
        self.d4.setas+="2.y.x3."
        self.assertTrue(self.d4.setas=="2.y.x3.","Equality test by numbered string failed.")
        self.d4.setas-="2.y"
        self.assertTrue(self.d4.setas=="4.x3.","Deletion Operator failed.")
        self.d4.setas="2.y.x3."
        self.assertEqual(self.d4.setas["rho"],"y","Getitem by type failed.")
        self.assertEqual(self.d4.setas["x"],"T (K)","Getitem by name failed.")
        self.assertEqual(self.d4.setas.x,4,"setas x attribute failed.")
        self.assertEqual(self.d4.setas.y,[2],"setas y attribute failed.")
        self.assertEqual(self.d4.setas.z,[],"setas z attribute failed.")
        self.assertTrue(all(self.d4.setas.set==np.array([False,False,True,False,True,False,False,False])),"setas.set attribute not working.")
        self.assertTrue(all(self.d4.setas.not_set==np.array([True,True,False,True,False,True,True,True])),"setas.set attribute not working.")
        self.assertTrue("x" in self.d4.setas,"set.__contains__ failed")
        del self.d4.setas["x"]
        self.assertFalse("x" in self.d4.setas,"setas del item by column type failed.")
        del self.d4.setas["rho"]
        self.assertFalse("y" in self.d4.setas,"setas del by column named failed")
        self.d4.setas({"T (K)":"x","y":"rho"})
        self.assertTrue(self.d4.setas=="..y.x...","Setting setas by call with dictionary failed")


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
        self.d4.setas="2.y.x3."
        self.assertEqual(self.d4.setas["#x"],4,"Indexing setas by #type failed.")
        self.assertEqual(self.d4.setas[1::2],['.', '.', '.', '.'],"Indexing setas with a slice failed.")
        self.assertEqual(self.d4.setas[[1,3,5,7]],['.', '.', '.', '.'],"Indexing setas with a slice failed.")
        self.d4.setas.clear()
        self.d4.setas+={"x":"T (K)","rho":"y"}
        self.assertEqual(self.d4.setas,"2.y.x3.","Adding dictionary with type:column and column:type to setas failed.")
        self.assertTrue(self.d4.setas.pop("x")=="T (K)" and self.d4.setas=="2.y5.","Pop from setas failed.")
        self.assertEqual(self.d4.setas.pop("z",False),False,"Pop with non existent key and default value failed")
        self.d4.setas.update({"x":"T (K)","rho":"y"})
        self.assertEqual(self.d4.setas,"2.y.x3.","setas.update failed.")


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
        auto_setas={'axes': 2, 'xcol': 0, 'ycol': [1], 'zcol': [], 'ucol': [], 'vcol': [], 'wcol': [], 'xerr': None, 'yerr': [2], 'zerr': [], 'has_xcol': True, 'has_xerr': False, 'has_ycol': True, 'has_yerr': True, 'has_zcol': False, 'has_zerr': False, 'has_ucol': False, 'has_vcol': False, 'has_wcol': False, 'has_axes': True, 'has_uvw': False}
        d2.setas=""
        self.assertEqual(sorted(d2.setas._get_cols().items()),sorted(auto_setas.items()),"Automatic guessing of setas failed!")
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
        self.d4.setas="xyxy4."
        m1=self.d4.setas._get_cols()
        m2=self.d4.setas._get_cols(startx=2)
        self.assertEqual("{xcol} {ycol}".format(**m1),"0 [1]","setas._get_cols without startx failed.\n{}".format(m1))
        self.assertEqual("{xcol} {ycol}".format(**m2),"2 [3]","setas._get_cols without startx failed.\n{}".format(m1))


if __name__=="__main__": # Run some tests manually to allow debugging
    test=SetasTest("test_setas_metadata")
    test.setUp()
    #test.test_setas_basics()
    unittest.main()