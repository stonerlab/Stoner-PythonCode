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
from Stoner.compat import *
import Stoner.Folders as SF
import Stoner.HDF5 as SH

from Stoner import Data

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)


class Folders_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.fldr=SF.objectFolder(directory=self.datadir)

    def test_Folders(self):
        fldr=self.fldr
        fl=len(fldr)
        print("fldr={}".format(fldr))
        print("fl={}".format(fl))
        datfiles=fnmatch.filter(os.listdir(self.datadir),"*.dat")
        print("datfiles={}".format(datfiles))
        self.assertEqual(len(os.listdir(self.datadir)),fl,"Failed to initialise DataFolder from sample data")
        print("Passed 1")
        self.assertEqual(fldr.index(fldr[-1].filename),fl-1,"Failed to index back on filename")
        print("Passed 2")
        self.assertEqual(fldr.count(fldr[-1].filename),1,"Failed to count filename with string")
        print("passed 3")
        self.assertEqual(fldr.count("*.dat"),len(datfiles),"Count with a glob pattern failed")
        print("Passed 4")
        self.assertEqual(len(fldr[::2]),round(len(fldr)/2.0),"Failed to get the correct number of elements in a folder slice")
        print("Passed 5")
        
    def test_Operators(self):
        fldr=self.fldr
        fl=len(fldr)
        d=Data(np.ones((100,5)))
        fldr+=d
        print("Starting 2")
        self.assertEqual(fl+1,len(fldr),"Failed += operator on DataFolder")
        print("Passed 1")
        fldr2=fldr+fldr
        print("Starting....")
        self.assertEqual((fl+1)*2,len(fldr2),"Failed + operator with DataFolder on DataFolder")
        print("Passed 2")
        fldr-="Untitled-0"
        self.assertEqual(len(fldr),fl,"Failed to remove Untitled-0 from DataFolder by name.")
        print("Passed 3")
        fldr-="New-XRay-Data.dql"
        self.assertEqual(fl-1,len(fldr),"Failed to remove NEw Xray data by name.")
        print("Passed 4")
        fldr+="New-XRay-Data.dql"
        self.assertEqual(len(fldr),fl,"Failed += oeprator with string on DataFolder")
        print("Passed 5")


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Folders_test("test_Folders")
    test.setUp()
    test.test_Folders()
    test.test_Operators()
    test.fldr/="Loaded as"
    print(test.fldr)

