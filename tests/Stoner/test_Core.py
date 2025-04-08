"""test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import sys
import os.path as path
import numpy as np
import re
import pandas as pd
import pytest
from numpy import all, sqrt, nan
from collections.abc import MutableMapping

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
old_path = []
from Stoner import Data, __home__
from Stoner.Core import metadataObject
import Stoner.compat


def mask_func(r):
    return np.abs(r.q) < 0.25 * np.pi


datadir = path.join(pth, "sample-data")


def setup_module():
    global old_path
    old_path = sys.path
    sys.path.insert(0, pth)


def teardown_module():
    global old_path
    sys.path = old_path


def setup_function(function):
    global selfd, selfd1, selfd2, selfd3, selfd4

    selfd = Data(path.join(path.dirname(__file__), "CoreTest.dat"), setas="xy")
    selfd2 = Data(path.join(__home__, "..", "sample-data", "TDI_Format_RT.txt"))
    selfd3 = Data(path.join(__home__, "..", "sample-data", "New-XRay-Data.dql"))
    selfd4 = Data(path.join(__home__, "..", "sample-data", "Cu_resistivity_vs_T.txt"))


def test_repr():
    global selfd, selfd1, selfd2, selfd3, selfd4
    header = """==============  ======  ======
TDI Format 1.5  X-Data  Y-Data
    index       0 (x)   1 (y)
==============  ======  ======"""
    repr_header = """=============================  ========  ========
TDI Format 1.5                   X-Data    Y-Data
index                             0 (x)     1 (y)
=============================  ========  ========"""
    assert selfd.header, header == "Header output changed."
    assert "\n".join(repr(selfd).split("\n")[:4]) == repr_header, "Representation gave unexpected header."
    assert len(repr(selfd2).split("\n")) == 261, "Long file representation failed"
    assert len(selfd2._repr_html_().split("\n")) == 263, "Long file representation failed"


def test_base_class():
    global selfd, selfd1, selfd2, selfd3, selfd4
    m = metadataObject()
    m.update(selfd2.metadata)
    assert isinstance(m, MutableMapping), "metadataObject is not a MutableMapping"
    assert len(m) == len(selfd2.metadata), "metadataObject length failure."
    assert len(m[".*"]) == len(m), "Failure to index by regexp."
    for k in selfd2.metadata:
        assert m[k] == selfd2.metadata[k], f"Failure to have equal keys for {k}"
        m[k] = k
        assert m[k] == k, f"Failure to set keys for {k}"
        del m[k]
    assert len(m) == 0, "Failed to delete from metadataObject."


def test_constructor():
    """Constructor Tests"""
    global selfd, selfd1, selfd2, selfd3, selfd4
    d = Data()
    assert d.shape == (1, 0), "Bare constructor failed"
    d = Data(selfd)
    assert np.all(d.data == selfd.data), "Constructor from DataFile failed"
    d = Data([np.ones(100), np.zeros(100)])
    assert d.shape == (100, 2), "Constructor from iterable list of nd array failed"
    d = Data([np.ones(100), np.zeros(100)], ["X", "Y"])
    assert d.column_headers == ["X", "Y"], f"Failed to set column headers in constructor: {d.column_headers}"
    c = np.zeros(100)
    d = Data({"X-Data": c, "Y-Data": c, "Z-Data": c})
    assert d.shape == (100, 3), "Construction from dictionary of columns failed."
    d = selfd.clone
    df = d.to_pandas()
    e = Data(df)
    assert d == e, "Roundtripping through Pandas DataFrame failed."
    d = selfd.clone
    e = Data(d.dict_records)
    e.metadata = d.metadata
    assert d == e, "Copy from dict records failed."
    d = Data(pd.DataFrame(np.zeros((100, 3))))
    assert d.column_headers == ["Column 0:0", "Column 1:1", "Column 2:2"]
    d = Data(np.ones(100), np.zeros(100), np.zeros(100), selfd.filename)
    assert d == selfd, "Multiple argument constructor dind't find it's filename"
    with pytest.raises(TypeError):
        e = d(setas=34)
    with pytest.raises(TypeError):
        e = d(column_headers=34)

    with pytest.raises(AttributeError):
        d = Data(bad_attr=False)
    with pytest.raises(TypeError):
        d = Data(column_headers=43)
    with pytest.raises(TypeError):
        d = Data(object)


def test_column():
    global selfd, selfd1, selfd2, selfd3, selfd4
    for i, c in enumerate(selfd.column_headers):
        # Column function checks
        assert all(selfd.data[:, i] == selfd.column(i)), f"Failed to Access column {i} by index"
        assert all(selfd.data[:, i] == selfd.column(c)), f"Failed to Access column {i} by name"
        assert all(selfd.data[:, i] == selfd.column(c[0:3])), f"Failed to Access column {i} by partial name"

    # Check that access by list of strings returns multpiple columns
    assert all(
        selfd.column(selfd.column_headers) == selfd.data
    ), "Failed to access all columns by list of string indices"
    assert all(
        selfd.column([0, selfd.column_headers[1]]) == selfd.data
    ), "Failed to access all columns by mixed list of indices"

    # Check regular expression column access
    assert all(
        selfd.column(re.compile(r"^X\-"))[:, 0] == selfd.column(0)
    ), "Failed to access column by regular expression"
    # Check attribute column access
    assert all(selfd.X == selfd.column(0)), "Failed to access column by attribute name"


def test_deltions():
    global selfd, selfd1, selfd2, selfd3, selfd4
    ch = ["{}-Data".format(chr(x)) for x in range(65, 91)]
    data = np.zeros((100, 26))
    metadata = dict([("Key 1", True), ("Key 2", 12), ("Key 3", "Hello world")])
    selfdd = Data(metadata)
    selfdd.data = data
    selfdd.column_headers = ch
    selfdd.setas = "3.x3.y3.z"
    selfrepr_string = (
        "==========================  ========  =======  ========  =======  ========  ========\n"
        + "TDI Format 1.5                D-Data   ....      H-Data   ....      Y-Data    Z-Data\n"
        + "index                          3 (x)              7 (y)                 24        25\n"
        + "==========================  ========  =======  ========  =======  ========  ========\n"
        + "Key 1{Boolean}= True               0                  0                  0         0\n"
        + "Key 2{I32}= 12                     0  ...             0  ...             0         0\n"
        + "Key 3{String}= Hello world         0  ...             0  ...             0         0\n"
        + "Stoner.class{String}= Data         0  ...             0  ...             0         0\n"
        + "...                                0  ...             0  ...             0         0"
    )
    assert (
        "\n".join(repr(selfdd).split("\n")[:9]) == selfrepr_string
    ), "Representation with interesting columns failed."
    del selfdd["Key 1"]
    assert len(selfdd.metadata) == 3, "Deletion of metadata failed."
    del selfdd[20:30]
    assert selfdd.shape == (90, 26), "Deleting rows directly failed."
    selfdd.del_column("Q")
    assert selfdd.shape == (90, 25), "Deleting rows directly failed."
    selfdd %= 3
    assert selfdd.shape == (90, 24), "Deleting rows directly failed."
    selfdd.setas = "x..y..z"
    selfdd.del_column(selfdd.setas.not_set)
    assert selfdd.shape == (90, 3), "Deleting rows directly failed."
    selfdd.mask[50, 1] = True
    selfdd.del_column()
    assert selfdd.shape == (90, 2), "Deletion of masked rows failed."
    selfd5 = Data(np.ones((10, 10)))
    selfd5.column_headers = ["A"] * 5 + ["B"] * 5
    selfd5.del_column("A", duplicates=True)
    assert selfd5.shape == (10, 6), "Failed to delete columns with duplicates True and col specified."
    selfd5 = Data(np.ones((10, 10)))
    selfd5.column_headers = list("ABCDEFGHIJ")
    selfd5.setas = "..x..y"
    selfd5.del_column(True)
    assert selfd5.column_headers == ["C", "F"], "Failed to delete columns with col=True"


def test_attributes():
    """Test various attribute accesses,"""
    global selfd, selfd1, selfd2, selfd3, selfd4
    header = (
        "==============  ======  ======\n"
        + "TDI Format 1.5  X-Data  Y-Data\n"
        + "    index       0 (x)   1 (y)\n"
        + "==============  ======  ======"
    )
    assert selfd.shape == (100, 2), "shape attribute not correct."
    assert selfd.dims == 2, "dims attribute not correct."
    assert selfd.header == header, "Shorter header not as expected"
    assert all(selfd.T == selfd.data.T), "Transpose attribute not right"
    assert all(selfd.x == selfd.column(0)), "x attribute quick access not right."
    assert all(selfd.y == selfd.column(1)), "y attribute not right."
    assert all(selfd.q == np.arctan2(selfd.data[:, 0], selfd.data[:, 1])), "Calculated theta attribute not right."
    assert all(sqrt(selfd[:, 0] ** 2 + selfd[:, 1] ** 2) == selfd.r), "Calculated r attribute not right."
    assert selfd2.records._[5]["Column 2"] == selfd2.data[5, 2], "Records and as array attributes problem"
    d = Data(np.ones((10, 2)), ["A", "A"])
    assert d.records.dtype == np.dtype(
        [("A_1", "<f8"), ("A", "<f8")]
    ), ".records with identical column headers handling failed"
    e = selfd.clone
    e.data = np.array([1, 1, 1])
    e.T = selfd.T
    e.column_headers = selfd.column_headers
    assert e == selfd, "Writingto transposed data failed"
    assert selfd.xcol == 0
    assert selfd.ycol == [1]
    with pytest.raises(AttributeError):
        selfd.zcol is None
    selfd.setas = "xz"
    assert selfd.zcol == [1]
    selfd.x = np.ones((100, 1))
    assert all(selfd.x == np.ones(100))
    selfd.x = np.ones((1, 100))
    assert all(selfd.x == np.ones(100))
    with pytest.raises(RuntimeError):
        selfd.x = np.ones((2, 2))


def test_mask():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selflittle = Data()
    p = np.linspace(0, np.pi, 91)
    q = np.linspace(0, 2 * np.pi, 91)
    r = np.cos(p)
    x = r * np.sin(q)
    y = r * np.cos(q)
    selflittle.data = np.column_stack((x, y, r))
    selflittle.setas = "xyz"
    selflittle.mask = mask_func
    selflittle.mask[:, 2] = True
    assert len(repr(selflittle).split("\n")) == len(selflittle) + 5, "Non shorterned representation failed."
    selflittle.mask = lambda r: np.abs(r) > 0.5
    selflittle.del_rows()
    assert selflittle.shape == (30, 3)


def test_iterators():
    global selfd, selfd1, selfd2, selfd3, selfd4
    for i, c in enumerate(selfd.columns()):
        assert all(selfd.column(i) == c), "Iterating over DataFile.columns not the same as direct indexing column"
    for j, r in enumerate(selfd.rows()):
        assert all(selfd[j] == r), "Iteratinf over DataFile.rows not the same as indexed access"
    for k, r in enumerate(selfd):
        pass
    assert j == k, "Iterating over DataFile not the same as DataFile.rows"
    assert selfd.data.shape == (j + 1, i + 1), "Iterating over rows and columns not the same as data.shape"


def test_dir():
    global selfd, selfd1, selfd2, selfd3, selfd4
    assert selfd.dir("S") == ["Stoner.class"], f"Dir method failed: dir was {selfd.dir()}"
    bad_keys = set(
        [
            "__class_getitem__",
            "__metaclass__",
            "iteritems",
            "iterkeys",
            "itervalues",
            "__ge__",
            "__gt__",
            "__init_subclass__",
            "__le__",
            "__lt__",
            "__reversed__",
            "__slots__",
            "_abc_negative_cache",
            "_abc_registry",
            "_abc_negative_cache_version",
            "_abc_cache",
            "_abc_impl",
            "__annotations__",
            "__getstate__",
            "__static_attributes__",
            "__firstlineno__",
        ]
    )
    attrs = set(dir(selfd)) - bad_keys
    assert len(attrs) == 247, "DataFile.__dir__ failed."
    selfd.setas.clear()
    attrs = set(dir(selfd)) - bad_keys
    assert len(attrs) == 245, "DataFile.__dir__ failed."


def test_filter():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selfd._push_mask()
    ix = np.argmax(selfd.x)
    selfd.filter(lambda r: r.x <= 50)
    assert np.max(selfd.x) == 50, "Failure of filter method to set mask"
    assert np.isnan(selfd.x[ix]), "Failed to mask maximum value"
    selfd._pop_mask()
    assert selfd2.select(Temp__not__gt=150).shape == (839, 3), "Seect method failure."
    assert selfd.select(lambda r: r.x < 30) == selfd.select(X__lt=30), "Select  method as callable failed."
    assert selfd.select(__=lambda r: r.x < 30) == selfd.select(X__lt=30), "Select  method as callable failed."


def test_properties():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selflittle = Data()
    p = np.linspace(0, np.pi, 91)
    q = np.linspace(0, 2 * np.pi, 91)
    r = np.cos(p)
    x = r * np.sin(q)
    y = r * np.cos(q)
    selflittle.data = np.column_stack((x, y, r))
    selflittle.setas = "xyz"
    q_ang = np.round(selflittle.q / np.pi, decimals=2)
    p_ang = np.round(selflittle.p / np.pi, decimals=2)
    assert np.max(q_ang) == 1.0, "Data.q failure"
    assert np.max(p_ang) == 0.5, "Data.p failure"
    assert np.min(p_ang) == -0.5, "Data.p failure"


def test_methods():
    global selfd, selfd1, selfd2, selfd3, selfd4
    d = selfd.clone
    d &= np.where(d.x < 50, 1.0, 0.0)
    d.rename(2, "Z-Data")
    d.setas = "xyz"
    assert all(d.unique(2) == np.array([0, 1])), f"Unique values failed: {d.unique(2)}"
    d = selfd.clone
    d.insert_rows(10, np.zeros((2, 2)))
    assert len(d) == 102, "Failed to inert extra rows"
    assert d[9, 0] == 10 and d[10, 0] == 0 and d[12, 0] == 11, "Failed to insert rows properly."
    d = selfd.clone
    d.add_column(np.ones(len(d)), replace=False, header="added")
    assert d.shape[1] == selfd.shape[1] + 1, "Adding a column with replace=False did not add a column."
    assert np.all(d.data[:, -1] == np.ones(len(d))), "Didn't add the new column to the end of the data."
    assert len(d.column_headers) == len(selfd.column_headers) + 1, "Column headers isn't bigger by one"
    assert d.column_headers == selfd.column_headers + [
        "added",
    ], "Column header not added correctly"
    d = selfd.clone
    d.add_column(selfd.x)
    assert np.all(d.x == d[:, 2]), "Adding a column as a DataArray with column headers didn't work"
    assert (
        d.x.column_headers[0] == d.column_headers[2]
    ), "Adding a column as a DataArray with column headers didn't work"
    e = d.clone
    d.swap_column([(0, 1), (0, 2)])
    assert d.column_headers == [e.column_headers[x] for x in [2, 0, 1]], f"Swap column test failed: {d.column_headers}"
    e = selfd(setas="yx")
    assert e.shape == selfd.shape and e.setas[0] == "y", "Failed on a DataFile.__call__ test"
    spl = len(repr(selfd).split("\n"))
    assert spl, 105 == f"Failed to do repr function got {spl} lines"
    e = selfd.clone
    e = e.add_column(e.x, header=e.column_headers[0])
    e.del_column(duplicates=True)
    assert e.shape == (100, 2), "Deleting duplicate columns failed"
    e = selfd2.clone
    e.reorder_columns([2, 0, 1])
    assert e.column_headers == [selfd2.column_headers[x] for x in [2, 0, 1]], "Failed to reorder columns: {}".format(
        e.column_headers
    )
    d = selfd.clone
    d.del_rows(0, 10.0)
    assert d.shape == (99, 2), f"Del Rows with value and column failed - actual shape {d.shape}"
    d = selfd.clone
    d.del_rows(0, (10.0, 20.0))
    assert d.shape == (89, 2), f"Del Rows with tuple and column failed - actual shape {d.shape}"
    d = selfd.clone
    d.mask[::2, 0] = True
    d.del_rows()
    assert d.shape == (50, 2), f"Del Rows with mask set - actual shape {d.shape}"
    d = selfd.clone
    d[::2, 1] = nan
    d.del_nan(1)
    assert d.shape == (50, 2), f"del_nan with explicit column set failed shape was {d.shape}"
    d = selfd.clone
    d[::2, 1] = nan
    d.del_nan(0)
    assert d.shape == (100, 2), f"del_nan with explicit column set and not nans failed shape was {d.shape}"
    d = selfd.clone
    d[::2, 1] = nan
    d.setas = ".y"
    d.del_nan()
    assert d.shape == (50, 2), f"del_nan with columns from setas failed shape was {d.shape}"
    d = selfd.clone
    d2 = selfd.clone
    d2.data = d2.data[::-1, :]
    assert d.sort(reverse=True) == d2, "Sorting reverse not the same as manually reversed data."
    d = selfd.clone
    d.mask[::2] = True
    assert d.count() == 50
    assert d.count(9895) == 1
    assert d.count(100, col="X") == 1
    assert selfd.search("X", [98.0, 100]).shape == (2, 2)
    assert selfd.search("X", 50.1, accuracy=0.2, columns=0) == 50.0
    with pytest.raises(RuntimeError):
        selfd.search("X", "Y")
    assert selfd.section(x=(48, 52), accuracy=0.2).shape == (5, 2)
    assert selfd.section(y=(1000, 2000))[0, 0] == 33.0
    assert selfd.select({"X-Data__lt": 50}).shape == (49, 2)
    assert selfd.select({"X-Data": 50})[0, 0] == 50.0
    dd = selfd.clone
    dd.add_column(dd.x % 2, "Channel")
    fldr = dd.split("Channel")
    assert fldr.shape == (2, {})
    assert dd.split("Channel", lambda d: d.x % 3).shape == (0, {0.0: (3, {}), 1.0: (3, {})})
    dd = selfd.clone
    assert dd.split(lambda d: d.x % 3).shape == (3, {})
    assert len(dd.split(lambda d: 1)) == 1
    shp1 = dd.split(lambda d: np.sqrt(d.x) == np.round(np.sqrt(d.x))).shape
    shp2 = dd.split(lambda d: np.sqrt(d.x) == int(np.sqrt(d.x))).shape
    assert shp1 == shp2


def test_metadata_save():
    global selfd, selfd1, selfd2, selfd3, selfd4
    local = path.dirname(__file__)
    t = np.arange(12).reshape(3, 4)  # set up a test data file with mixed metadata
    t = Data(t)
    t.column_headers = ["1", "2", "3", "4"]
    metitems = [
        True,
        1,
        0.2,
        {"a": 1, "b": "abc"},
        (1, 2),
        np.arange(3),
        [1, 2, 3],
        "abc",  # all types accepted
        r"\\abc\cde",
        1e-20,  # extra tests
        [1, (1, 2), "abc"],  # list with different types
        [[[1]]],  # nested list
        None,  # None value
    ]
    metnames = ["t" + str(i) for i in range(len(metitems))]
    for k, v in zip(metnames, metitems):
        t[k] = v
    t.save(path.join(local, "mixedmetatest.dat"))
    tl = Data(
        path.join(local, "mixedmetatest.txt")
    )  # will change extension to txt if not txt or tdi, is this what we want?
    t2 = selfd4.clone  # check that python tdi save is the same as labview tdi save
    t2.save(path.join(local, "mixedmetatest2.txt"))
    t2l = Data(path.join(local, "mixedmetatest2.txt"))
    for orig, load in [(t, tl), (t2, t2l)]:
        for k in ["Loaded as", "TDI Format"]:
            orig[k] = load[k]
        assert np.allclose(orig.data, load.data)
        assert orig.column_headers == load.column_headers
        _ = load.metadata ^ orig.metadata
        assert load.metadata == orig.metadata, "Metadata not the same on round tripping to disc"
    # os.remove(path.join(local, "mixedmetatest.txt")) #clear up
    # os.remove(path.join(local, "mixedmetatest2.txt"))


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
