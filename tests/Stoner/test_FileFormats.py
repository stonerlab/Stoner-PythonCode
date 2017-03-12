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
from Stoner.compat import *

from Stoner import Data
from Stoner.Core  import DataFile
import Stoner.HDF5 as SH
import Stoner.Zip as SZ

import warnings

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)

class FileFormats_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_loaders(self):
        d=None
        if python_v3:
            skip_files=[] # HDF5 loader not working Python 3.5
            #return None # skip this completely at this time
        else:
            skip_files=[]
        for i,f in enumerate(os.listdir(self.datadir)):
            if f.strip().lower() in skip_files: # Known bad files to load
                print("Skipping {}".format(f))
                continue
            else:
                print("Testing {}".format(f))
                try:
                    del d
                    fname=path.join(self.datadir,f)
                    d=Data(fname,debug=False)
                except Exception as e:
                    self.assertTrue(False,"Failed in loading <{}>\n{}".format(path.join(self.datadir,f),str(e)))
                self.assertTrue(isinstance(d,DataFile),"Failed to load {} correctly.".format(path.join(self.datadir,f)))

if __name__=="__main__": # Run some tests manually to allow debugging
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        test=FileFormats_test("test_loaders")
        test.setUp()
        test.test_loaders()
