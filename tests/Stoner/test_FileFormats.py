# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import unittest
import sys
import os.path as path
import os
import numpy as np
import re

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data
from Stoner.Core import DataFile
from Stoner.HDF5 import HDF5File
from Stoner.Zip import ZipFile


class FileFormats_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_loaders(self):
        d=None
        for f in os.listdir(self.datadir):
            if f.strip().lower() in ["ad_data_filemnames_list"]: # Known bad files to load
                print("Skipping {}".format(f))
                continue
            else:
                try:
                    del d
                    d=Data(path.join(self.datadir,f))
                except Exception as e:
                    self.assertTrue(False,"Failed in loading <{}>\n{}".format(path.join(self.datadir,f),e.message))
                self.assertTrue(isinstance(d,DataFile),"Failed to load {} correctly.".format(path.join(self.datadir,f)))

if __name__=="__main__": # Run some tests manually to allow debugging
    test=FileFormats_test("test_loaders")
    test.setUp()
    test.test_loaders()
