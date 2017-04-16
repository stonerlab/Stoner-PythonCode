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
import tempfile
from Stoner.compat import *

from Stoner import Data
from Stoner.Core  import DataFile
import Stoner.HDF5 as SH
import Stoner.Zip as SZ

import warnings
from traceback import format_exc

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
            
        tmpdir=tempfile.mkdtemp()
        print("Exporting to {}".format(tmpdir))
        print("Data files {}".format(self.datadir))
        incfiles=[x for x in os.listdir(self.datadir) if os.path.isfile(os.path.join(self.datadir,x)) and not x.endswith("tdms_index")]

        for i,f in enumerate(incfiles):
            if f.strip().lower() in skip_files: # Known bad files to load
                print("Skipping {}".format(f))
                continue
            else:
                print("Testing {}".format(f))
                try:
                    del d
                    fname=path.join(self.datadir,f)
                    d=Data(fname,debug=False)
                    self.assertTrue(isinstance(d,DataFile),"Failed to load {} correctly.".format(fname))
                    if "save" in d.subclasses[d["Loaded as"]].__dict__:
                        print("Checking save routine for {}".format(d["Loaded as"]))
                        pth=os.path.join(tmpdir,f)
                        name,ext=os.path.splitext(pth)
                        pth2="{}-2.{}".format(name,ext)
                        d.save(pth,as_loaded=True)
                        self.assertTrue(os.path.exists(pth) or os.path.exists(d.filename),"Failed to save as {}".format(pth))
                        os.remove(d.filename)
                        d.save(pth2,as_loaded=d["Loaded as"])
                        self.assertTrue(os.path.exists(pth2) or os.path.exists(d.filename),"Failed to save as {}".format(pth))
                        os.remove(d.filename)
                except Exception as e:
                    self.assertTrue(False,"Failed in loading <{}>\n{}".format(path.join(self.datadir,f),format_exc()))
        os.rmdir(tmpdir)

if __name__=="__main__": # Run some tests manually to allow debugging
    test=FileFormats_test("test_loaders")
    test.setUp()
    test.test_loaders()