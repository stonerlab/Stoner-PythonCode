#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:56:38 2018

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

from Stoner import Data,set_option
import Stoner.HDF5, Stoner.Zip
from Stoner.Util import hysteresis_correct
from pandas import DataFrame,Series

import matplotlib.pyplot as plt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)

class folders_metadata_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_metadata(self):
        os.chdir(self.datadir)
        fldr6=SF.DataFolder(".",pattern="QD*.dat",pruned=True)
        fldr6.sort()
        self.assertEqual(repr(fldr6.metadata),"The DataFolder . has 9 common keys of metadata in 4 Data objects",
                         "Representation method of metadata wrong.")
        self.assertEqual(len(fldr6.metadata),9,"Length of common metadata not right.")
        self.assertEqual(list(fldr6.metadata.keys()),['Byapp',
                                                       'Datatype,Comment',
                                                       'Datatype,Time',
                                                       'Fileopentime',
                                                       'Loaded as',
                                                       'Loaded from',
                                                       'Startupaxis-X',
                                                       'Startupaxis-Y1',
                                                       'Stoner.class'],"metadata.keys() not right.")
        self.assertEqual(len(list(fldr6.metadata.all_keys())),49,"metadata.all_keys() the wrong length.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from")[0],dict),"metadata.slice not returtning a dictionary.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from",values_only=True),list),"metadata.slice not returtning a list with values_only=True.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from",output="Data"),Data),"metadata.slice not returtning Data with outpt='data'.")
        for fmt,typ in zip(["dict","list","array","data","frame","smart"],
                           [(list,dict,int),(list,tuple,int),(np.ndarray,np.ndarray,np.int64),(Data,np.ndarray,np.int64),(DataFrame,Series,np.int64),(list,dict,int)]):
            ret=fldr6.metadata.slice("Datatype,Comment","Datatype,Time",output=fmt)
            for ix,t in enumerate(typ):
                self.assertTrue(isinstance(ret,t),"Return from slice metadata for output={} and dimension {} had type {} and not {}".format(fmt,ix,type(ret),t))
                try:
                    ret=ret[0]
                except (KeyError):
                    ret=ret[list(ret.keys())[0]]
                except (IndexError,TypeError):
                    pass
        for k,typ in zip(['Info.Sample_Holder',('Info.Sample_Holder',"Datatype,Comment")],[(np.ndarray,np.ndarray,np.str),(np.ndarray,np.ndarray)]):
            ret=fldr6.metadata[k]
            for ix,t in enumerate(typ):
                self.assertTrue(isinstance(ret,t),"Indexing metadata for key={} and dimension {} had type {} and not {}".format(k,ix,type(ret),t))
                try:
                    ret=ret[0]
                except (KeyError):
                    ret=ret[list(ret.keys())[0]]
                except AttributeError:
                    ret=ret.data[0]
                except (IndexError,TypeError):
                    pass
        del fldr6.metadata["Datatype,Comment"]
        try:
            ret=fldr6.metadata["Datatype,Comment"]
        except KeyError:
            pass
        else:
            self.assertTrue(False,"Failed to delete from metadata : {}".format(ret))


if __name__=="__main__": # Run some tests manually to allow debugging
    test=folders_metadata_test("test_metadata")
    test.setUp()
    #test.test_metadata()
    unittest.main()
