# -*- coding: utf-8 -*-
"""
Test_Util.py

Created on Mon Jul 18 14:13:39 2016

@author: phygbu"""


import unittest
import sys
import os.path as path
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

class ZipFolder(SZ.ZipFolderMixin,Stoner.Folders.baseFolder):
    pass

class Zip_test(unittest.TestCase):

    """Path to sample Data File"""

    def setUp(self):
        self.fldr=ZipFolder(path.join(testdata,"ZipFolder.zip"),flat=True)
        
    def test_zipfolder(self):
        pass


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Zip_test("test_zipfolder")
    test.setUp()
    unittest.main()