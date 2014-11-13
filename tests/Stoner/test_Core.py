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
from Stoner.Core import DataFile

class CoreDataFiletest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")
    
    def setUp(self):
        self.d=DataFile(path.join(self.datadir,"TDI_Format_RT.txt"))
        
    def test_column(self):
        for i in range(len(self.d.column_headers)):
            #Check each indivudla column indexed by string and index match
            self.assertTrue(np.all(self.d.column(i)==self.d.column(self.d.column_headers[i])),"Failed to Access column {} by string".format(i))
        # Check that access by list of strings returns multpiple columns
        self.assertTrue(np.all(self.d.column(self.d.column_headers)==self.d.data),"Failed to access all columns by list of string indices")
        # Check regular expression column access
        self.assertTrue(self.d.column(re.compile(r"[T-Z].*$"))==self.d.column("Temperature"),"Failed to access column by regular expression")        
        # Check attribute column access
        self.asserTrue(self.d.Temp==self.d.column(0),"Failed to access column by attribute name")

    def test_len(self):
        # Check that length of the column is the same as length of the data
        self.assertEqual(len(self.d.column(0)),len(self.d),"Column 0 length not equal to DataFile length")
        self.assertEqual(len(self.d),self.d.data.shape[0],"DataFile length not equal to data.shape[0]")
        # Check that self.column_headers returns the right length
        self.assertEqual(len(self.d.column_headers),self.d.data.shape[1],"Length of column_headers not equal to data.shape[1]")    
        
    def test_iterators(self):
        i=0
        for c in self.d.columns():
            self.assertTrue(np.all(self.d.column(i)==c))
            i+=1
        j=0
        for r in self.d.rows():
            self.assertTrue(np.all(self.d[j]==r))
            j+=1
        self.assertEqual(self.d.data.shape,(j,i))
        
    def test_metadata(self):
        self.d["Test"]="This is a test"
        self.d["Int"]=1
        self.d["Float"]=1.0
        self.assertEqual(self.d["Int"],1)
        self.assertEqual(self.d["Float"],1.0)
        self.assertEqual(self.d["Test"],self.d.metadata["Test"])
        self.assertEqual(self.d.metadata._typehints["Int"],"I32")
        
    def test_dir(self):
        self.assertTrue(self.d.dir("U")==["User"],"Dir method failed")
        
    def test_filter(self):
        self.d._push_mask()
        self.d.filter(lambda r:r[0]<100)
        self.assertTrue(np.max(self.d.Temp)<100,"Failure of filter method to set mask")
        self.assertTrue(np.ma.is_masked(max(self.d.Temp)),"Failed to mask maximum value")
        self.d._pop_mask()
        
    def test_operators(self):
        self.d2=DataFile()
        self.d2.column_headers=["C1","C2"]
        
        

