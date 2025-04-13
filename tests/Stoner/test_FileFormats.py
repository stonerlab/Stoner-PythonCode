# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import sys
import pathlib
import io
import urllib

from Stoner import Data, __homepath__, __datapath__, ImageFile
from Stoner.Core import DataFile
from Stoner.compat import Hyperspy_ok

import pytest

from Stoner.formats.attocube import AttocubeScan
from Stoner.formats.maximus import MaximusStack
from Stoner.tools.file import get_saver
from Stoner.core.exceptions import StonerLoadError
from Stoner.core.exceptions import StonerUnrecognisedFormat

pth = __homepath__ / ".."
datadir = __datapath__


def setup_module():
    sys.path.insert(0, str(pth))


def teardown_module():
    sys.path.remove(str(pth))


def list_files():
    skip_files = set([])  # HDF5 loader not working Python 3.5
    incfiles = list(set(datadir.glob("*")) - skip_files)
    incfiles = [x for x in incfiles if x.suffix != ".tdms_index"]
    incfiles = [x for x in incfiles if not x.is_dir()]

    if not Hyperspy_ok:
        print("hyperspy too old, skupping emd file for test")
        incfiles = [x for x in incfiles if not x.name.strip().lower().endswith(".emd")]

    return sorted(incfiles)


listed_files = list_files()


@pytest.mark.parametrize("filename", listed_files)
def test_one_file(tmpdir, filename):
    loaded = Data(filename, debug=False)
    assert isinstance(loaded, DataFile), f"Failed to load {filename.name} correctly."
    try:
        saver=get_saver(loaded["Loaded as"])
        pth = pathlib.Path(tmpdir) / filename.name
        _, name, ext = pth.parent, pth.stem, pth.suffix
        pth2 = pathlib.Path(tmpdir) / f"{name}-2{ext}"
        loaded.save(pth, as_loaded=True)
        assert pth.exists() or pathlib.Path(loaded.filename).exists(), f"Failed to save as {pth}"
        pathlib.Path(loaded.filename).unlink()
        loaded.save(pth2, as_loaded=loaded["Loaded as"])
        assert pth2.exists() or pathlib.Path(loaded.filename).exists(), "Failed to save as {}".format(pth)
        pathlib.Path(loaded.filename).unlink()
    except StonerLoadError:
        pass

def test_csvfile():
    csv = Data(
        datadir / "working" / "CSVFile_test.dat", filetype="JustNumbers", column_headers=["Q", "I", "dI"], setas="xye"
    )
    assert csv.shape == (167, 3), "Failed to load CSVFile from text"


def test_attocube_scan(tmpdir):
    tmpdir = pathlib.Path(tmpdir)
    scandir = datadir / "attocube_scan"
    scan1 = AttocubeScan("SC_085", scandir, regrid=False)
    scan2 = AttocubeScan(85, scandir, regrid=False)
    assert scan1 == scan2, "Loading scans by number and string not equal"

    # self.assertEqual(scan1,scan2,"Loading Attocube Scans by root name and number didn't match")

    pth = tmpdir / f"SC_{scan1.scan_no:03d}.hdf5"
    scan1.to_hdf5(pth)

    scan3 = AttocubeScan.read_hdf5(pth)

    assert pth.exists(), f"Failed to save scan as {pth}"
    if scan1 != scan3:
        print("A" * 80)
        print(scan1.layout, scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys ^ scan3[grp].metadata.all_by_keys)
    print(scan1.shape)
    assert scan1.layout == scan3.layout, "Roundtripping scan through hdf5 failed"
    pth.unlink()

    pth = tmpdir / f"SC_{scan1.scan_no:03d}.tiff"
    scan1.to_tiff(pth)
    scan3 = AttocubeScan.from_tiff(pth)
    assert pth.exists(), f"Failed to save scan as {pth}"
    if scan1 != scan3:
        print("B" * 80)
        print(scan1.layout, scan3.layout)
        for grp in scan1.groups:
            print(scan1[grp].metadata.all_by_keys ^ scan3[grp].metadata.all_by_keys)
    assert scan1.layout == scan3.layout, "Roundtripping scan through tiff failed"
    pth.unlink()

    scan3 = AttocubeScan()
    scan3._marshall(layout=scan1.layout, data=scan1._marshall())
    assert scan1 == scan3, "Recreating scan through _marshall failed."

    scan1["fwd"].level_image(method="parabola", signal="Amp")
    scan1["bwd"].regrid()


def test_maximus_image():
    pths = list((datadir / "maximus_scan").glob("MPI_210127019*.*"))
    assert len(pths) == 2
    for pth in pths:
        img = ImageFile.load(pth)
        assert img.shape == (1000, 1000)
        assert len(img.metadata) == 196


def test_maximus_stack(tmpdir):
    tmpdir = pathlib.Path(tmpdir)
    scandir = datadir / "maximus_scan" / "MPI_210127021"
    stack = MaximusStack(scandir / "MPI_210127021")
    stack.to_hdf5(tmpdir / "MPI_210127021.hdf5")
    stack2 = MaximusStack.read_hdf5(tmpdir / "MPI_210127021.hdf5")
    assert stack2.shape == stack.shape, "Round trip through MaximusStack"


def test_fail_to_load():
    with pytest.raises(StonerUnrecognisedFormat):
        _ = Data(datadir /"bad_data" / "Origin_Project.opju")


def test_arb_class_load():
    _ = Data(datadir / "TDI_Format_RT.txt", filetype="dummy.ArbClass")


def test_url_load():
    """Test URL scheme openers."""
    t1 = Data("https://github.com/stonerlab/Stoner-PythonCode/raw/master/sample-data/hairboRaman.spc")
    assert t1 == Data(__datapath__ / "hairboRaman.spc")
    t2 = Data("https://github.com/stonerlab/Stoner-PythonCode/raw/master/sample-data/New-XRay-Data.dql")
    assert t2 == Data(__datapath__ / "New-XRay-Data.dql")
    resp = urllib.request.urlopen(
        "https://github.com/stonerlab/Stoner-PythonCode/raw/master/sample-data/New-XRay-Data.dql"
    )
    t3 = Data(resp)
    assert t3 == t2


def test_from_bytes():
    """Test loading a binary file as bytes."""
    with open(__datapath__ / "harribo.spc", "rb") as data:
        d = Data(data.read())
    assert d == Data(__datapath__ / "harribo.spc")


def test_from_StringIO():
    """Test loading a binary file as bytes."""
    with open(__datapath__ / "RASOR.dat", "r") as data:
        buffer = io.StringIO(data.read())
    assert Data(buffer) == Data(__datapath__ / "RASOR.dat")


def test_ImageAutoLoad():
    """Test ImageFile autoloading"""
    img = ImageFile(__datapath__ / "kermit.png")
    assert img.shape == (479, 359)
    img = ImageFile(__datapath__ / "working" / "hydra_0017.edf")
    assert img.shape == (512, 768)
    img = ImageFile(__datapath__ / "working" / "Sample_Image_2017-06-03_035.hdf5")
    assert img.shape == (80, 300)


def test_FabioImageFle():
    loader = ImageFile.load(datadir / "working" / "hydra_0017.edf", filetype="FabioImage")
    assert loader.shape == (512, 768)


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
