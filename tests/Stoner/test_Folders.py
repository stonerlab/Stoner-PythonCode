# -*- coding: utf-8 -*-
"""
test_Folders.py

Created on Mon Jul 18 14:13:39 2016

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
import Stoner.HDF5 as SH
import Stoner.Zip as SZ

from Stoner import Data

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)


class Folders_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.fldr=SF.DataFolder(self.datadir)

    def test_Folders(self):
        self.setUp()
        fldr=self.fldr
        fl=len(fldr)
        datfiles=fnmatch.filter(os.listdir(self.datadir),"*.dat")
        length = len([i for i in os.listdir(self.datadir) if path.isfile(os.path.join(self.datadir,i))])-1 # don't coiunt TDMS index
        self.assertEqual(length,fl,"Failed to initialise DataFolder from sample data")
        self.assertEqual(fldr.index(fldr[-1].filename),fl-1,"Failed to index back on filename")
        self.assertEqual(fldr.count(fldr[-1].filename),1,"Failed to count filename with string")
        self.assertEqual(fldr.count("*.dat"),len(datfiles),"Count with a glob pattern failed")
        self.assertEqual(len(fldr[::2]),ceil(len(fldr)/2.0),"Failed to get the correct number of elements in a folder slice")
        
    def test_Operators(self):
        self.setUp()
        fldr=self.fldr
        fl=len(fldr)
        d=Data(np.ones((100,5)))
        fldr+=d
        self.assertEqual(fl+1,len(fldr),"Failed += operator on DataFolder")
        fldr2=fldr+fldr
        self.assertEqual((fl+1)*2,len(fldr2),"Failed + operator with DataFolder on DataFolder")
        fldr-="Untitled"
        self.assertEqual(len(fldr),fl,"Failed to remove Untitled-0 from DataFolder by name.")
        fldr-="New-XRay-Data.dql"
        self.assertEqual(fl-1,len(fldr),"Failed to remove NEw Xray data by name.")
        fldr+="New-XRay-Data.dql"
        self.assertEqual(len(fldr),fl,"Failed += oeprator with string on DataFolder")
        fldr/="Loaded as"
        self.assertEqual(len(fldr["QDFile"]),4,"Failoed to group folder by Loaded As metadata with /= opeator.")
        fldr.flatten()
        
    def test_Properties(self):
        self.setUp()
        fldr=self.fldr
        fldr/="Loaded as"
        fldr["QDFile"].group("Byapp")
        self.assertEqual(fldr.mindepth,1,"mindepth attribute of folder failed.")
        self.assertEqual(fldr.depth,2,"depth attribute failed.")
        fldr.flatten()
        fldr+=Data()
        self.assertEqual(len(list(fldr.loaded)),1,"loaded attribute failed {}".format(len(list(fldr.loaded))))
        self.assertEqual(len(list(fldr.not_empty)),len(fldr)-1,"not_empty attribute failed.")
        fldr-="Untitled"
        
    def test_methods(self):
        sliced=np.array(['MDAASCIIFile',
                         'BNLFile',
                         'DataFile',
                         'DataFile',
                         'DataFile',
                         'MokeFile',
                         'EasyPlotFile',
                         'DataFile',
                         'DataFile',
                         'DataFile'])
        self.fldr=SF.DataFolder(self.datadir, pattern='*.txt').sort()
        self.assertTrue(np.all(self.fldr.slice_metadata("Loaded as")==sliced),"Slicing metadata failed to work.")

                
    def test_clone(self):
         self.fldr=SF.DataFolder(self.datadir, pattern='*.txt')
         self.fldr.abc = 123 #add an attribute
         t = self.fldr.__clone__()
         self.assertTrue(t.pattern==['*.txt'], 'pattern didnt copy over')
         self.assertTrue(hasattr(t, "abc") and t.abc==123, 'user attribute didnt copy over')
         self.assertTrue(isinstance(t['recursivefoldertest'],SF.DataFolder), 'groups didnt copy over')
        


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Folders_test("test_Folders")
    test.setUp()
    test.test_Folders()
    test.test_Operators()
    test.test_Properties()
    test.test_clone()
    test.test_methods()

