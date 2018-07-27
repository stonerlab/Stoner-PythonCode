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
import Stoner.Zip as SZ
Data=Stoner.Data

pth=path.dirname(__file__)
testdata=path.realpath(path.join(pth,"test-data"))

root=path.realpath(path.join(Stoner.__home__,".."))
sample_data=path.realpath(path.join(root,"sample-data","NLIV"))
tmpdir=tempfile.mkdtemp()


class Zip_test(unittest.TestCase):

    """Path to sample Data File"""

    def setUp(self):
        self.fldr=Stoner.DataFolder(sample_data,pattern="*.txt")

    def test_zipfolder(self):
        #Test constructor from DataFolder
        self.zipfldr=SZ.ZipFolder(self.fldr)
        self.assertEqual(self.fldr.shape,self.zipfldr.shape,"ZipFolder created from DataFolder didn't keep the same shape")
        self.assertEqual(self.fldr[0],self.zipfldr[0],"First element of ZipFolder created from DataFolder changed!")
        zipname=path.join(tmpdir,"test-zipfolder.zip")
        self.zipfldr.save(zipname)
        self.assertEqual(self.fldr.shape,self.zipfldr.shape,"ZipFolder Changed shape when saving!")
        self.zipfldr_2=SZ.ZipFolder(zipname)
        #self.assertEqual(self.zipfldr_2.shape,self.zipfldr.shape,"ZipFolder loaded from disc not same shape as ZipFolder in memory!")


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Zip_test("test_zipfolder")
    test.setUp()
    test.test_zipfolder()
    #unittest.main()
