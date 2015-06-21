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
import re

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data

class Datatest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(path.dirname(__file__),"CoreTest.dat"),setas="xy")

    def test_column(self):
        for i,c in enumerate(self.d.column_headers):
            # Column function checks
            self.assertTrue(np.all(self.d.data[:,i]==self.d.column(i)),"Failed to Access column {} by index".format(i))
            self.assertTrue(np.all(self.d.data[:,i]==self.d.column(c)),"Failed to Access column {} by name".format(i))
            self.assertTrue(np.all(self.d.data[:,i]==self.d.column(c[0:3])),"Failed to Access column {} by partial name".format(i))

        # Check that access by list of strings returns multpiple columns
        self.assertTrue(np.all(self.d.column(self.d.column_headers)==self.d.data),"Failed to access all columns by list of string indices")
        self.assertTrue(np.all(self.d.column([0,self.d.column_headers[1]])==self.d.data),"Failed to access all columns by mixed list of indices")

        # Check regular expression column access
        self.assertTrue(np.all(self.d.column(re.compile(r"^X\-"))[:,0]==self.d.column(0)),"Failed to access column by regular expression")
        # Check attribute column access
        self.assertTrue(np.all(self.d.X==self.d.column(0)),"Failed to access column by attribute name")

    def test_len(self):
        # Check that length of the column is the same as length of the data
        self.assertEqual(len(self.d.column(0)),len(self.d),"Column 0 length not equal to DataFile length")
        self.assertEqual(len(self.d),self.d.data.shape[0],"DataFile length not equal to data.shape[0]")
        # Check that self.column_headers returns the right length
        self.assertEqual(len(self.d.column_headers),self.d.data.shape[1],"Length of column_headers not equal to data.shape[1]")

    def test_attributes(self):
        """Test various atribute accesses,"""
        self.assertEqual(self.d.shape,(100,2),"shape attribute not correct.")
        self.assertTrue(np.all(self.d.T==self.d.data.T),"Transpose attribute not right")
        self.assertTrue(np.all(self.d.x==self.d.column(0)),"x attribute quick access not right.")
        self.assertTrue(np.all(self.d.y==self.d.column(1)),"y attribute not right.")
        self.assertTrue(np.all(self.d.q==np.arctan2(self.d.data[:,0],self.d.data[:,1])),"Calculated theta attribute not right.")

    def test_setas(self):
        #Check readback of setas
        self.assertEqual(self.d.setas[:],["x","y"],"setas attribute not set in constructor")
        self.assertTrue(np.all(self.d.x==self.d.column(0)),"Attribute setas column axis fails")
        self.d.setas(x="Y-Data")
        self.assertEqual(self.d.setas[:],["x","x"],"Failed to set setas by type=column keyword assignment")
        self.d.setas(Y="y")
        self.assertEqual(self.d.setas[:],["x","y"],"Failed to set setas by column=type keyword assignment")


    def test_iterators(self):
        for i,c in enumerate(self.d.columns()):
            self.assertTrue(np.all(self.d.column(i)==c),"Iterating over DataFile.columns not the same as direct indexing column")
        for j,r in enumerate(self.d.rows()):
            self.assertTrue(np.all(self.d[j]==r),"Iteratinf over DataFile.rows not the same as indexed access")
        for k,r in enumerate(self.d):
            pass
        self.assertEqual(j,k,"Iterating over DataFile not the same as DataFile.rows")
        self.assertEqual(self.d.data.shape,(j+1,i+1),"Iterating over rows and columns not the same as data.shape")

    def test_metadata(self):
        self.d["Test"]="This is a test"
        self.d["Int"]=1
        self.d["Float"]=1.0
        self.assertEqual(self.d["Int"],1)
        self.assertEqual(self.d["Float"],1.0)
        self.assertEqual(self.d["Test"],self.d.metadata["Test"])
        self.assertEqual(self.d.metadata._typehints["Int"],"I32")
        self.assertEqual(len(self.d.dir()),4,"Failed meta data directory listing ({})".format(len(self.d.dir())))


    def test_dir(self):
        self.assertTrue(self.d.dir("S")==["Stoner.class"],"Dir method failed")

    def test_filter(self):
        self.d._push_mask()
        ix=np.argmax(self.d.x)
        self.d.filter(lambda r:r[0]<50)
        self.assertTrue(np.max(self.d.x)<50,"Failure of filter method to set mask")
        self.assertTrue(np.ma.is_masked(self.d.x[ix]),"Failed to mask maximum value")
        self.d._pop_mask()

    def test_operators(self):
        #Test Column Indexer
        self.assertTrue(np.all(self.d//0==self.d.column(0)),"Failed the // operator with integer")
        self.assertTrue(np.all(self.d//"X-Data"==self.d.column(0)),"Failed the // operator with string")
        self.assertTrue(np.all((self.d//re.compile(r"^X\-"))[:,0]==self.d.column(0)),"Failed the // operator with regexp")
        t=[self.d%1,self.d%"Y-Data",(self.d%re.compile(r"Y\-"))]
        for ix,tst in enumerate(["integer","string","regexp"]):
            self.assertTrue(np.all(t[ix].data==np.atleast_2d(self.d.column(0)).T),"Failed % operator with {} index".format(tst))
        d=self.d&self.d.x
        self.assertTrue(d.shape[1]==3,"& operator failed.")
        d&=d.x
        self.assertTrue(d.shape[1]==4,"Inplace & operator failed")
        d=self.d+np.array([0,0])
        self.assertTrue(len(d)==len(self.d)+1,"+ operator failed.")
        d+=np.array([0,0])
        self.assertTrue(len(d)==len(self.d)+2,"Inplace + operator failed.")
        d=d-(-1)
        self.assertTrue(len(d)==len(self.d)+1,"Delete with integer index failed. {} vs {}".format(len(d),len(self.d)+1))
        d-=-1
        self.assertTrue(len(d)==len(self.d),"Inplace delete with integer index failed. {} vs {}".format(len(d),len(self.d)))
        d-=slice(0,-1,2)
        self.assertTrue(len(d)==len(self.d)/2,"Inplace delete with slice index failed. {} vs {}".format(len(d),len(self.d)/2))

    def test_methods(self):
        d=self.d.clone
        d&=np.where(d.x<50,1.0,0.0)
        d.rename(2,"Z-Data")
        d.setas="xyz"
        self.assertTrue(np.all(d.unique(2)==np.array([0,1])),"Unique values failed: {}".format(d.unique(2)))
        d=self.d.clone
        d.insert_rows(10,np.zeros((2,2)))
        self.assertEqual(len(d),102,"Failed to inert extra rows")
        self.assertTrue(d[9,0]==10 and d[10,0]==0 and d[12,0]==11, "Failed to insert rows properly.")



