# -*- coding: utf-8 -*-
"""Test the HDF5 file format handling."""


import os.path as path
import tempfile
import Stoner
import pytest
import Stoner.HDF5 as SH

Data = Stoner.Data

pth = path.dirname(__file__)
testdata = path.realpath(path.join(pth, "test-data"))

root = path.realpath(path.join(Stoner.__home__, ".."))
sample_data = path.realpath(path.join(root, "sample-data", "NLIV"))
tmpdir = tempfile.mkdtemp()


def test_HDF5folder():
    # Test constructor from DataFolder
    self_fldr = Stoner.DataFolder(sample_data, pattern="*.txt")
    self_HDF5fldr = SH.HDF5Folder(self_fldr)
    assert self_fldr.shape == self_HDF5fldr.shape, "HDF5Folder created from DataFolder didn't keep the same shape"
    assert self_fldr[0] == self_HDF5fldr[0], "First element of HDF5Folder created from DataFolder changed!"
    HDF5name = path.join(tmpdir, "test-HDF5folder.HDF5")
    self_HDF5fldr.save(HDF5name)
    assert self_fldr.shape == self_HDF5fldr.shape, "HDF5Folder Changed shape when saving!"
    self_HDF5fldr_2 = SH.HDF5Folder(HDF5name)
    self_HDF5fldr_2.compress()
    self_HDF5fldr.sort("i")
    self_HDF5fldr_2.sort("i")
    assert (
        self_HDF5fldr_2.shape == self_HDF5fldr.shape
    ), "HDF5Folder loaded from disc not same shape as HDF5Folder in memory!"
    self_h1 = self_HDF5fldr[0]
    self_h2 = self_HDF5fldr_2[0]
    self_h2.metadata["Stoner.class"] = "Data"  # Correct the loader class
    self_h2.metadata["Loaded from"] = self_h1.metadata["Loaded from"]  # Corrects a path separator bug on Windows
    assert self_h1 == self_h2, "File from loaded HDF5Folder not the same as in memory HDF5Folder."


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
