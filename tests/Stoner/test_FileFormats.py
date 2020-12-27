# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import sys
import pathlib

from Stoner import Data,__homepath__
from Stoner.Core import DataFile
from Stoner.compat import Hyperspy_ok

import pytest

from Stoner.formats.attocube import AttocubeScan
from Stoner.tools.classes import subclasses
from Stoner.core.exceptions import StonerUnrecognisedFormat
from traceback import format_exc

pth=__homepath__/".."
datadir=pth/"sample-data"

def setup_module():
    sys.path.insert(0,str(pth))


def teardown_module():
    sys.path.remove(str(pth))

def list_files():
    skip_files=set([]) # HDF5 loader not working Python 3.5
    incfiles=list(set(datadir.glob("*"))-skip_files)
    incfiles=[x for x in incfiles if x.suffix!=".tdms_index"]
    incfiles=[x for x in incfiles if not x.is_dir()]

    if not Hyperspy_ok:
        print("hyperspy too old, skupping emd file for test")
        incfiles=[x for x in incfiles if not x.name.strip().lower().endswith(".emd")]


    return sorted(incfiles)



@pytest.mark.parametrize("filename", list_files(),ids=[x.name for x in list_files()])
def test_one_file(tmpdir, filename):
    loaded=Data(filename,debug=False)
    assert isinstance(loaded,DataFile),f"Failed to load {filename.name} correctly."
    if "save" in subclasses()[loaded["Loaded as"]].__dict__:
        pth = pathlib.Path(tmpdir)/filename.name
        parent, name,ext=pth.parent, pth.stem, pth.suffix
        pth2=pathlib.Path(parent)/f"{name}-2{ext}"
        loaded.save(pth,as_loaded=True)
        assert pth.exists() or pathlib.Path(loaded.filename).exists(),f"Failed to save as {pth}"
        pathlib.Path(loaded.filename).unlink()
        loaded.save(pth2,as_loaded=loaded["Loaded as"])
        assert pth2.exists() or pathlib.Path(loaded.filename).exists(),"Failed to save as {}".format(pth)
        pathlib.Path(loaded.filename).unlink()


def test_csvfile():

    csv=Data(datadir/"working"/"CSVFile_test.dat",filetype="JustNumbers",column_headers=["Q","I","dI"],setas="xye")
    assert csv.shape==(167,3),"Failed to load CSVFile from text"

def test_attocube_scan(tmpdir):
    tmpdir=pathlib.Path(tmpdir)
    scandir=datadir/"attocube_scan"
    scan1=AttocubeScan("SC_085",scandir,regrid=False)
    scan2=AttocubeScan(85,scandir,regrid=False)
    assert scan1==scan2,"Loading scans by number and string not equal"

    #self.assertEqual(scan1,scan2,"Loading Attocube Scans by root name and number didn't match")

    pth=tmpdir/f"SC_{scan1.scan_no:03d}.hdf5"
    scan1.to_HDF5(pth)

    scan3=AttocubeScan.read_HDF(pth)

    assert pth.exists(),f"Failed to save scan as {pth}"
    if scan1!=scan3:
        print("A"*80)
        print(scan1.layout,scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys^scan3[grp].metadata.all_by_keys)
    print(scan1.shape)
    assert scan1.layout==scan3.layout,"Roundtripping scan through hdf5 failed"
    pth.unlink()

    pth=tmpdir/f"SC_{scan1.scan_no:03d}.tiff"
    scan1.to_tiff(pth)
    scan3=AttocubeScan.from_tiff(pth)
    assert pth.exists(),f"Failed to save scan as {pth}"
    if scan1!=scan3:
        print("B"*80)
        print(scan1.layout,scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys^scan3[grp].metadata.all_by_keys)
    assert scan1.layout==scan3.layout,"Roundtripping scan through tiff failed"
    pth.unlink()

    scan3=AttocubeScan()
    scan3._marshall(layout=scan1.layout,data=scan1._marshall())
    assert scan1==scan3,"Recreating scan through _marshall failed."

    scan1["fwd"].level_image(method="parabola",signal="Amp")
    scan1["bwd"].regrid()

def test_fail_to_load():
    with pytest.raises(StonerUnrecognisedFormat):
        d=Data(datadir/"TDMS_File.tdms_index")

def test_arb_class_load():
    d=Data(datadir/"TDI_Format_RT.txt", filetype="dummy.ArbClass")


if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])