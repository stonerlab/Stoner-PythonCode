# -*- coding: utf-8 -*-
"""
test_Folders.py

Created on Mon Jul 18 14:13:39 2016

@author: phygbu
"""


import sys
import os.path as path
import os
import numpy as np
import re
import fnmatch
from numpy import ceil
from copy import copy
from Stoner.compat import Hyperspy_ok
import pytest
from Stoner import DataFolder, __homepath__
from Stoner.folders import PlotFolder

from Stoner import Data
from Stoner.core.base import regexpDict
from Stoner.folders.core import baseFolder
import matplotlib.pyplot as plt

import tempfile

pth = __homepath__ / ".."
sys.path.insert(0, pth)

"""Path to sample Data File"""
datadir = pth / "sample-data"


def test_Folders():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    fl = len(fldr)
    skip = 1 if Hyperspy_ok else 2
    datfiles = fnmatch.filter(os.listdir(datadir), "*.dat")
    length = (
        len([i for i in os.listdir(datadir) if path.isfile(os.path.join(datadir, i))]) - skip
    )  # don't coiunt TDMS index
    assert length == fl, "Failed to initialise DataFolder from sample data {} {} {} {}".format(
        fl, length, skip, Hyperspy_ok
    )
    assert fldr.index(path.basename(fldr[-1].filename)) == fl - 1, "Failed to index back on filename"
    assert fldr.count(path.basename(fldr[-1].filename)) == 1, "Failed to count filename with string"
    assert fldr.count("*.dat") == len(datfiles), "Count with a glob pattern failed"
    assert len(fldr[::2]) == ceil(len(fldr) / 2.0), "Failed to get the correct number of elements in a folder slice"


def test_loader_opts():
    fldr7 = DataFolder(
        path.join(datadir, "NLIV"), pattern=re.compile(r".*at (?P<field>[0-9\-\.]*)\.txt"), read_means=True
    )
    x = fldr7.metadata.slice(["field", "Voltage", "Current"], output="Data")
    assert x.span("field") == (-0.05, 0.04), "Extract from name pattern and slice into metadata failed."
    assert all(x // "Current" < 0) and all(x // "Current" > -1e-20), "Extract means failed."
    assert list(fldr7.not_loaded) == [], "Not loaded attribute failed."
    fldr7.unload(0)
    assert len(list(fldr7.not_loaded)) == 1, "Unload by index failed."
    fldr7.unload()
    assert len(list(fldr7.not_loaded)) == len(fldr7), "Unload all failed."

    def add_col(d):
        d.add_column(np.ones(len(d)) * d["field"], header="field")

    fldr7.each(add_col)
    fldr7.concatenate()
    assert fldr7[0].shape == (909, 4), "Concatenate failed."


def test_groups_methods():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    fldr.group("Loaded as")
    fldr.groups.keep(["QDFile", "OpenGDAFile"])
    assert fldr.shape == (0, {"OpenGDAFile": (1, {}), "QDFile": (4, {})}), "groups.keep method failed on folder"


def test_discard_earlier():
    fldr2 = DataFolder(path.join(pth, "tests/Stoner/folder_data"), pattern="*.dat", discard_earlier=True)
    fldr3 = DataFolder(path.join(pth, "tests/Stoner/folder_data"), pattern="*.dat")
    assert len(fldr2) == 1, "Folder created with disacrd_earlier has wrong length ({})".format(len(fldr2))
    assert len(fldr3) == 5, "Folder created without disacrd_earlier has wrong length ({})".format(len(fldr3))
    fldr3.keep_latest()
    assert list(fldr2.ls) == list(
        fldr3.ls
    ), "Folder.keep_latest didn't do the same as discard_earliest in constructor."


def test_clear_and_attrs():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    fldr2 = fldr.clone
    fldr2.clear()
    assert fldr2.shape == (0, {}), "Failed to clear"
    fldr2.files = fldr.files
    fldr2.groups = fldr.groups
    assert fldr2.shape == fldr.shape, "Failed to write to files and groups attri"
    fldr.each.debug = True
    assert fldr[0].debug, "Setting an attribute on fldr didn't propagate to the contents"
    del fldr.each.debug
    assert not hasattr(fldr[0], "hello"), "Failed to delete attribute from DataFolder"
    with pytest.raises(AttributeError):
        del fldr.debug


def test_Operators():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    fl = len(fldr)
    d = Data(np.ones((100, 5)))
    fldr += d
    assert fl + 1 == len(fldr), "Failed += operator on DataFolder"
    fldr2 = fldr + fldr
    assert (fl + 1) * 2 == len(fldr2), "Failed + operator with DataFolder on DataFolder"
    fldr -= "Untitled"
    assert len(fldr) == fl, "Failed to remove Untitled-0 from DataFolder by name."
    fldr -= "New-XRay-Data.dql"
    assert fl - 1 == len(fldr), "Failed to remove NEw Xray data by name."
    fldr += "New-XRay-Data.dql"
    assert len(fldr) == fl, "Failed += operator with string on DataFolder"
    fldr /= "Loaded as"
    assert len(fldr["QDFile"]) == 4, "Failoed to group folder by Loaded As metadata with /= operator."
    assert isinstance(fldr["QDFile", "Byapp"], Data), "Indexing group and then common metadata failed"
    fldr = DataFolder(datadir, debug=False, recursive=False)
    fldr2 = DataFolder(path.join(datadir, "NLIV"), pattern="*.txt")
    fldr2.group(lambda x: "zero" if x["iterator"] % 2 == 0 else "one")
    fldr3 = fldr + fldr2
    assert fldr3.shape == (52, {"one": (9, {}), "zero": (7, {})}), "Adding two DataFolders with groups failed"
    fldr4 = fldr3 - fldr2
    fldr4.prune()
    assert fldr4.shape == fldr.shape, "Failed to subtract one DataFolder from another :{}".format(fldr4.shape)
    del fldr2["one"]
    assert fldr2.shape == (0, {"zero": (7, {})}), "Delitem with group failed"
    fldr2.key = path.basename(fldr2.key)
    assert repr(fldr2) == (
        "DataFolder(NLIV) with pattern ('*.txt',) has 0 files and 1 groups\n\tDataFolder(zero) with pattern "
        + "['*.txt'] has 7 files and 0 groups"
    ), "Representation methods failed"
    fldr = DataFolder(datadir, debug=False, recursive=False)
    names = list(fldr.ls)[::2]
    fldr -= names
    assert len(fldr) == 26, "Failed to delete from a sequence"
    with pytest.raises(TypeError):
        fldr - 0.34
    with pytest.raises(RuntimeError):
        fldr - Data()
    with pytest.raises(RuntimeError):
        fldr - "Wiggle"


def test_Base_Operators():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    for d in fldr:
        _ = d["Loaded as"]
    fldr = baseFolder(fldr)
    fl = len(fldr)
    d = Data(np.ones((100, 5)))
    fldr += d
    assert fl + 1 == len(fldr), "Failed += operator on DataFolder"
    fldr2 = fldr + fldr
    assert (fl + 1) * 2 == len(fldr2), "Failed + operator with DataFolder on DataFolder"
    fldr -= "Untitled"
    assert len(fldr) == fl, "Failed to remove Untitled-0 from DataFolder by name."
    fldr -= "New-XRay-Data.dql"
    assert fl - 1 == len(fldr), "Failed to remove NEw Xray data by name."
    if Hyperspy_ok:
        del fldr["1449 37.0 kx.emd"]
    fldr /= "Loaded as"
    assert len(fldr["QDFile"]) == 4, "Failoed to group folder by Loaded As metadata with /= operator."
    fldr = DataFolder(datadir, debug=False, recursive=False)
    for d in fldr:
        _ = d["Loaded as"]
    fldr = baseFolder(fldr)
    fldr2 = DataFolder(path.join(datadir, "NLIV"), pattern="*.txt")
    fldr2.group(lambda x: "zero" if x["iterator"] % 2 == 0 else "one")
    fldr3 = fldr + fldr2
    assert fldr3.shape == (52, {"one": (9, {}), "zero": (7, {})}), "Adding two DataFolders with groups failed"
    fldr4 = fldr3 - fldr2
    fldr4.prune()
    assert fldr4.shape == fldr.shape, "Failed to subtract one DataFolder from another :{}".format(fldr4.shape)
    del fldr2["one"]
    assert fldr2.shape == (0, {"zero": (7, {})}), "Delitem with group failed"
    fldr2.key = path.basename(fldr2.key)
    assert repr(fldr2) == (
        "DataFolder(NLIV) with pattern ('*.txt',) has 0 files and 1 groups\n\tDataFolder(zero) with pattern"
        + " ['*.txt'] has 7 files and 0 groups"
    ), "Representation methods failed"
    fldr = DataFolder(datadir, debug=False, recursive=False)
    names = list(fldr.ls)[::2]
    fldr -= names
    assert len(fldr) == 26, "Failed to delete from a sequence"
    with pytest.raises(TypeError):
        fldr - 0.34
    with pytest.raises(RuntimeError):
        fldr - Data()
    with pytest.raises(RuntimeError):
        fldr - "Wiggle"


def test_Properties():
    fldr = DataFolder(datadir, debug=False, recursive=False)
    if not Hyperspy_ok:
        del fldr[".*emd$"]
    assert fldr.mindepth == 0, "Minimum depth of flat group n ot equal to zero."
    fldr /= "Loaded as"
    grps = list(fldr.lsgrp)
    skip = 0 if Hyperspy_ok else 1
    assert len(grps) == 27 - skip, f"Length of lsgrp not as expected: {len(grps)} not {27-skip}"
    fldr.debug = True
    fldr = fldr
    assert fldr["XRDFile"][0].debug, "Setting debug on folder failed!"
    fldr.debug = False
    fldr["QDFile"].group("Byapp")
    assert fldr.trunkdepth == 1, "Trunkdepth failed"
    assert fldr.mindepth == 1, "mindepth attribute of folder failed."
    assert fldr.depth == 2, "depth attribute failed."
    fldr = DataFolder(datadir, debug=False, recursive=False)
    fldr += Data()
    skip = 1 if Hyperspy_ok else 2
    assert len(list(fldr.loaded)) == 1, "loaded attribute failed {}".format(len(list(fldr.loaded)))
    assert len(list(fldr.not_empty)) == len(fldr) - skip, "not_empty attribute failed."
    fldr -= "Untitled"
    assert not fldr.is_empty, "fldr.is_empty failed"
    fldr = DataFolder(datadir, debug=False, recursive=False)
    objects = copy(fldr.objects)
    fldr.objects = dict(objects)
    assert isinstance(fldr.objects, regexpDict), "Folder objects not reset to regexp dictionary"
    fldr.objects = objects
    assert isinstance(fldr.objects, regexpDict), "Setting Folder objects mangled type"
    fldr.type = Data()
    assert issubclass(fldr.type, Data), "Setting type by instance of class failed"


def test_methods():
    sliced = np.array(
        [
            "DataFile",
            "MDAASCIIFile",
            "BNLFile",
            "DataFile",
            "DataFile",
            "DataFile",
            "DataFile",
            "DataFile",
            "MokeFile",
            "EasyPlotFile",
            "DataFile",
            "DataFile",
            "DataFile",
        ],
        dtype="<U12",
    )
    fldr = DataFolder(datadir, pattern="*.txt", recursive=False).sort()

    test_sliced = fldr.slice_metadata("Loaded as")
    assert len(sliced) == len(test_sliced), "Test slice not equal length - sample-data changed? {}".format(
        test_sliced
    )
    assert np.all(test_sliced == sliced), "Slicing metadata failed to work."
    fldr.insert(5, Data())
    assert list(fldr.ls)[5] == "Untitled", "Insert failed"
    fldr = fldr
    _ = fldr[-1]
    assert list(reversed(fldr))[0].filename == fldr[-1].filename


def test_clone():
    fldr = DataFolder(datadir, pattern="*.txt")
    fldr.abc = 123  # add an attribute
    t = fldr.__clone__()
    assert t.pattern == fldr.pattern, "pattern didn't copy over"
    assert hasattr(t, "abc") and t.abc == 123, "user attribute didn't copy over"
    assert isinstance(t["recursivefoldertest"], DataFolder), "groups didn't copy over"


def test_grouping():
    fldr4 = DataFolder()
    x = np.linspace(-np.pi, np.pi, 181)
    for phase in np.linspace(0, 1.0, 5):
        for amplitude in np.linspace(1, 2, 6):
            for frequency in np.linspace(1, 2, 5):
                y = amplitude * np.sin(frequency * x + phase * np.pi)
                d = Data(x, y, setas="xy", column_headers=["X", "Y"])
                d["frequency"] = frequency
                d["amplitude"] = amplitude
                d["phase"] = phase
                d["params"] = [phase, frequency, amplitude]
                d.filename = "test/{amplitude}/{phase}/{frequency}.dat".format(**d)
                fldr4 += d
    fldr4.unflatten()
    assert fldr4.mindepth == 3, "Unflattened DataFolder had wrong mindepth."
    assert fldr4.shape == (~~fldr4).shape, "Datafodler changed shape on flatten/unflatten"
    fldr5 = fldr4.select(amplitude=1.4, recurse=True)
    fldr5.prune()
    pruned = (
        0,
        {
            "test": (
                0,
                {"1.4": (0, {"0.0": (5, {}), "0.25": (5, {}), "0.5": (5, {}), "0.75": (5, {}), "1.0": (5, {})})},
            )
        },
    )
    selected = (0, {"test": (0, {"1.4": (0, {"0.25": (1, {}), "0.5": (1, {}), "0.75": (1, {}), "1.0": (1, {})})})})
    assert fldr5.shape == pruned, "Folder pruning gave an unexpected shape."
    assert fldr5[("test", "1.4", "0.5", 0, "phase")] == 0.5, "Multilevel indexing of tree failed."
    shape = (~(~fldr4).select(amplitude=1.4).select(frequency=1).select(phase__gt=0.2)).shape
    fldr4 = fldr4
    assert shape == selected, "Multi selects and inverts failed."
    g = (~fldr4) / 10
    assert g.shape == (
        0,
        {
            "Group 0": (15, {}),
            "Group 1": (15, {}),
            "Group 2": (15, {}),
            "Group 3": (15, {}),
            "Group 4": (15, {}),
            "Group 5": (15, {}),
            "Group 6": (15, {}),
            "Group 7": (15, {}),
            "Group 8": (15, {}),
            "Group 9": (15, {}),
        },
    ), "Dive by int failed."
    g["Group 6"] -= 5
    assert g.shape == (
        0,
        {
            "Group 0": (15, {}),
            "Group 1": (15, {}),
            "Group 2": (15, {}),
            "Group 3": (15, {}),
            "Group 4": (15, {}),
            "Group 5": (15, {}),
            "Group 6": (14, {}),
            "Group 7": (15, {}),
            "Group 8": (15, {}),
            "Group 9": (15, {}),
        },
    ), "Sub by int failed."
    remove = g["Group 3"][4]
    g["Group 3"] -= remove
    assert g.shape == (
        0,
        {
            "Group 0": (15, {}),
            "Group 1": (15, {}),
            "Group 2": (15, {}),
            "Group 3": (14, {}),
            "Group 4": (15, {}),
            "Group 5": (15, {}),
            "Group 6": (14, {}),
            "Group 7": (15, {}),
            "Group 8": (15, {}),
            "Group 9": (15, {}),
        },
    ), "Sub by object failed."
    d = fldr4["test", 1.0, 1.0].gather(0, 1)
    assert d.shape == (181, 6), "Gather seems have failed."
    assert np.all(fldr4["test", 1.0, 1.0].slice_metadata("phase") == np.ones(5)), "Slice metadata failure."
    d = (~fldr4).extract("phase", "frequency", "amplitude", "params")
    assert d.shape == (150, 6), "Extract failed to produce data of correct shape."
    assert d.column_headers == [
        "phase",
        "frequency",
        "amplitude",
        "params",
        "params",
        "params",
    ], "Extract failed to get correct column headers."
    p = fldr4["test", 1.0, 1.0]
    p = PlotFolder(p)
    p.plot()
    assert len(plt.get_fignums()) == 1, "Failed to generate a single plot for PlotFolder."
    plt.close("all")


def test_saving():
    fldr4 = DataFolder()
    x = np.linspace(-np.pi, np.pi, 181)
    for phase in np.linspace(0, 1.0, 5):
        for amplitude in np.linspace(1, 2, 6):
            for frequency in np.linspace(1, 2, 5):
                y = amplitude * np.sin(frequency * x + phase * np.pi)
                d = Data(x, y, setas="xy", column_headers=["X", "Y"])
                d["frequency"] = frequency
                d["amplitude"] = amplitude
                d["phase"] = phase
                d["params"] = [phase, frequency, amplitude]
                d.filename = "test/{amplitude}/{phase}/{frequency}.dat".format(**d)
                fldr4 += d
    fldr4.unflatten()
    newdir = tempfile.mkdtemp()
    fldr4.save(newdir)
    fldr5 = DataFolder(newdir)
    assert fldr4.shape == fldr5.shape, "Saved DataFolder and loaded DataFolder have different shapes"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
