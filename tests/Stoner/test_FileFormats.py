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
from Stoner.compat import Hyperspy_ok
from Stoner.Core  import DataFile
import Stoner.HDF5 as SH
import Stoner.Zip as SZ

import pytest

from Stoner.formats.attocube import AttocubeScan
from Stoner.core.utils import subclasses
import warnings
from traceback import format_exc

pth=path.join(__home__,"..")
datadir=path.join(pth,"sample-data")

sys.path.insert(0,pth)

datadir=path.join(pth,"sample-data")

def list_files():
    skip_files=[] # HDF5 loader not working Python 3.5
    incfiles=[x for x in os.listdir(datadir) if os.path.isfile(os.path.join(datadir,x)) and not x.endswith("tdms_index") and not x.lower() in skip_files]

    if not Hyperspy_ok:
        print("hyperspy too old, skupping emd file for test")
        incfiles.remove("1449 37.0 kx.emd")

    return incfiles



@pytest.mark.parametrize("filename", list_files(),ids=list_files())
def test_one_file(tmpdir, filename):
    try:
        fname=path.join(datadir,filename)
        d=Data(fname,debug=False)
        assert isinstance(d,DataFile),"Failed to load {} correctly.".format(fname)
        if "save" in subclasses()[d["Loaded as"]].__dict__:
            print("Checking save routine for {}".format(d["Loaded as"]))
            pth=os.path.join(tmpdir,filename)
            name,ext=os.path.splitext(pth)
            pth2="{}-2.{}".format(name,ext)
            d.save(pth,as_loaded=True)
            assert os.path.exists(pth) or os.path.exists(d.filename),"Failed to save as {}".format(pth)
            os.remove(d.filename)
            d.save(pth2,as_loaded=d["Loaded as"])
            assert os.path.exists(pth2) or os.path.exists(d.filename),"Failed to save as {}".format(pth)
            os.remove(d.filename)
    except Exception as e:
         assert False,f"Failed in loading <{fname}>\n{format_exc()}"

def test_csvfile():

    csv=Data(path.join(datadir,"working","CSVFile_test.dat"),filetype="JustNumbers",column_headers=["Q","I","dI"],setas="xye")
    assert csv.shape==(167,3),"Failed to load CSVFile from text"

def test_attocube_scan(tmpdir):
    scandir=path.join(datadir,"attocube_scan")
    scan1=AttocubeScan("SC_085",scandir,regrid=False)
    scan2=AttocubeScan(85,scandir,regrid=False)
    assert scan1==scan2,"Loading scans by number and string not equal"

    #self.assertEqual(scan1,scan2,"Loading Attocube Scans by root name and number didn't match")

    pth=os.path.join(tmpdir,f"SC_{scan1.scan_no:03d}.hdf5")
    scan1.to_HDF5(pth)

    scan3=AttocubeScan.read_HDF(pth)

    assert os.path.exists(pth),f"Failed to save scan as {pth}"
    if scan1!=scan3:
        print("A"*80)
        print(scan1.layout,scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys^scan3[grp].metadata.all_by_keys)
    assert scan1==scan3,"Roundtripping scan through hdf5 failed"
    os.remove(pth)

    pth=os.path.join(tmpdir,f"SC_{scan1.scan_no:03d}.tiff")
    scan1.to_tiff(pth)
    scan3=AttocubeScan.from_tiff(pth)
    assert os.path.exists(pth),f"Failed to save scan as {pth}"
    if scan1!=scan3:
        print("B"*80)
        print(scan1.layout,scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys^scan3[grp].metadata.all_by_keys)
    assert scan1.layout==scan3.layout,"Roundtripping scan through tiff failed"
    os.remove(pth)

    scan3=AttocubeScan()
    scan3._marshall(layout=scan1.layout,data=scan1._marshall())
    assert scan1==scan3,"Recreating scan through _marshall failed."

    scan1["fwd"].level_image(method="parabola",signal="Amp")
    scan1["bwd"].regrid()

if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])