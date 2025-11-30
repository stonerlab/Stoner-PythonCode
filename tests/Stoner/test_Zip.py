# -*- coding: utf-8 -*-
"""Test the ZIP module for zipfile compressed objects."""


import os.path as path
import tempfile
import zipfile as zf

import pytest

import Stoner

pth = path.dirname(__file__)

root = path.realpath(path.join(Stoner.__home__, ".."))
sample_data = path.realpath(path.join(root, "sample-data", "NLIV"))
tmpdir = tempfile.mkdtemp()


def test_zipFile(tmpdir):
    d = Stoner.Data(Stoner.__datapath__ / "TDI_Format_RT.txt")
    d.save(path.join(tmpdir, "TDI_Format_RT.zip"), filetype="ZippedFile")

    z2 = Stoner.Data(path.join(tmpdir, "TDI_Format_RT.zip"))
    d["Loaded as"] = z2["Loaded as"]
    d["Stoner.class"] = z2["Stoner.class"]
    assert d == z2
    with zf.ZipFile(path.join(tmpdir, "TDI_Format_RT.zip"), "r") as open_zipfile:
        z3 = Stoner.Data().load(open_zipfile, filetype="ZippedFile")
    z3["Stoner.class"] = z2["Stoner.class"]
    z3["Loaded as"] = z2["Loaded as"]
    assert z2 == z3


def test_zipfolder():
    # Test constructor from DataFolder
    sf = Stoner.DataFolder(sample_data, pattern="*.txt")
    szf = Stoner.folders.zip.ZipFolder(sf)
    assert sf.shape == szf.shape, "ZipFolder created from DataFolder didn't keep the same shape"
    assert sf[0] == szf[0], "First element of ZipFolder created from DataFolder changed!"
    zipname = path.join(tmpdir, "test-zipfolder.zip")
    szf.save(zipname)
    assert sf.shape == szf.shape, "ZipFolder Changed shape when saving!"
    szf_2 = Stoner.folders.zip.ZipFolder(zipname).compress()
    assert szf_2.shape == szf.shape, "ZipFolder loaded from disc not same shape as ZipFolder in memory!"
    fname = path.basename(szf[0].filename)
    assert szf[fname] == szf_2[fname], "File from loaded ZipFolder not the same as in memory ZipFolder."


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
