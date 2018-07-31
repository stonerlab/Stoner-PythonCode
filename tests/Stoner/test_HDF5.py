# -*- coding: utf-8 -*-
"""
Test_Util.py

Created on Mon Jul 18 14:13:39 2016

@author: phygbu"""


import unittest
import sys
import os.path as path
import tempfile
import os
import numpy as np
import re
from Stoner.compat import *
import Stoner
from Stoner.tools import isComparable
import Stoner.HDF5 as SH
Data=Stoner.Data

pth=path.dirname(__file__)
testdata=path.realpath(path.join(pth,"test-data"))

root=path.realpath(path.join(Stoner.__home__,".."))
sample_data=path.realpath(path.join(root,"sample-data","NLIV"))
tmpdir=tempfile.mkdtemp()


class HDF5_test(unittest.TestCase):

    """Path to sample Data File"""

    def setUp(self):
        self.fldr=Stoner.DataFolder(sample_data,pattern="*.txt")

    def test_HDF5folder(self):
        #Test constructor from DataFolder
        self.HDF5fldr=SH.HDF5Folder(self.fldr)
        self.assertEqual(self.fldr.shape,self.HDF5fldr.shape,"HDF5Folder created from DataFolder didn't keep the same shape")
        self.assertEqual(self.fldr[0],self.HDF5fldr[0],"First element of HDF5Folder created from DataFolder changed!")
        HDF5name=path.join(tmpdir,"test-HDF5folder.HDF5")
        self.HDF5fldr.save(HDF5name)
        self.assertEqual(self.fldr.shape,self.HDF5fldr.shape,"HDF5Folder Changed shape when saving!")
        self.HDF5fldr_2=SH.HDF5Folder(HDF5name)
        self.HDF5fldr_2.compress()
        self.HDF5fldr.sort("i")
        self.HDF5fldr_2.sort("i")
        self.assertEqual(self.HDF5fldr_2.shape,self.HDF5fldr.shape,"HDF5Folder loaded from disc not same shape as HDF5Folder in memory!")
        self.h1=self.HDF5fldr[0]
        self.h2=self.HDF5fldr_2[0]
        self.h2.metadata["Stoner.class"]="Data" # Correct the loader class
        self.h2.metadata["Loaded from"]=self.h1.metadata["Loaded from"] #Corrects a path separator bug on Windows
        self.assertEqual(self.h1,self.h2,"File from loaded HDF5Folder not the same as in memeory HDF5Folder.")


if __name__=="__main__": # Run some tests manually to allow debugging
    test=HDF5_test("test_HDF5folder")
    test.setUp()
    test.test_HDF5folder()
    #unittest.main()
