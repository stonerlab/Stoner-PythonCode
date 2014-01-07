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
        
        
        

