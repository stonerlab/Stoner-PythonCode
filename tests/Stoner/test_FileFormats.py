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

from Stoner import Data,__home__
from Stoner.compat import hyperspy_ok
from Stoner.Core  import DataFile
import Stoner.HDF5 as SH
import Stoner.Zip as SZ

from Stoner.formats.attocube import AttocubeScan

import warnings
from traceback import format_exc

pth=path.join(__home__,"..")
datadir=path.join(pth,"sample-data")

sys.path.insert(0,pth)

class FileFormats_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_loaders(self):
        d=None
        skip_files=[] # HDF5 loader not working Python 3.5

        tmpdir=tempfile.mkdtemp()
        print("Exporting to {}".format(tmpdir))
        print("Data files {}".format(self.datadir))
        incfiles=[x for x in os.listdir(self.datadir) if os.path.isfile(os.path.join(self.datadir,x)) and not x.endswith("tdms_index")]

        if not hyperspy_ok:
            print("hyperspy too old, skupping emd file for test")
            incfiles.remove("1449 37.0 kx.emd")

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

    def test_csvfile(self):

        self.csv=Data(path.join(self.datadir,"working","CSVFile_test.dat"),filetype="JustNumbers",column_headers=["Q","I","dI"],setas="xye")
        self.assertEqual(self.csv.shape,(167,3),"Failed to load CSVFile from text")

    def test_attocube_scan(self):
        scandir=path.join(self.datadir,"attocube_scan")
        scan1=AttocubeScan("SC_085",scandir,regrid=False)
        scan2=AttocubeScan(85,scandir,regrid=False)
        self.s1=scan1
        self.s2=scan2
        #self.assertEqual(scan1,scan2,"Loading Attocube Scans by root name and number didn't match")
        tmpdir=tempfile.mkdtemp()

        pth=os.path.join(tmpdir,f"SC_{scan1.scan_no:03d}.hdf5")
        scan1.to_HDF5(pth)

        scan2=AttocubeScan.read_HDF(pth)
        self.s3=scan2

        self.assertTrue(os.path.exists(pth),f"Failed to save scan as {pth}")
        if scan1!=scan2:
            print("A"*80)
            print(scan1.layout,scan2.layout)
            for grp in scan1.groups:
                print(scan1[grp].metadata.all_by_keys^scan3[grp].metadata.all_by_keys)
        self.assertEqual(scan1,scan2,"Roundtripping scan through hdf5 failed")
        os.remove(pth)

        pth=os.path.join(tmpdir,f"SC_{scan1.scan_no:03d}.tiff")
        scan1.to_tiff(pth)
        scan2=AttocubeScan.from_tiff(pth)
        self.assertTrue(os.path.exists(pth),f"Failed to save scan as {pth}")
        self.assertEqual(scan1.layout,scan2.layout,"Round tripping thropugh tiff file failed")
        os.remove(pth)

        scan2=AttocubeScan()
        scan2._marshall(layout=scan1.layout,data=scan1._marshall())
        self.assertEqual(scan1,scan2,"Recreating scan through _marshall failed.")

        os.rmdir(tmpdir)
        scan1["fwd"].level_image(method="parabola",signal="Amp")
        scan1["bwd"].regrid()



if __name__=="__main__": # Run some tests manually to allow debugging
    test=FileFormats_test("test_loaders")
    test.setUp()
    test.test_attocube_scan()
    #unittest.main()
    #test.test_csvfile()