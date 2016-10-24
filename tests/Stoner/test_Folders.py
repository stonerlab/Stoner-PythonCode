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
from Stoner.compat import *
import Stoner.Folders as SF

from Stoner import Data

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)


class Folders_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_Folders(self):
        fldr=SF.objectFolder(directory=self.datadir,)
        print(fldr)


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Folders_test("test_Folders")
    test.setUp()
    test.test_Folders()