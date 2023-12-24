# -*- coding: utf-8 -*-
"""Test Stoner.Data as a sequence and mapping structure."""

import pytest
import sys
from os import path
import numpy as np
from numpy import all

from Stoner import Data, __home__

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
sys.path.insert(0, pth)
datadir = path.join(pth, "sample-data")
selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")


def test_indexing():
    # Check all the indexing possibilities
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    data = np.array(selfd.data)
    colname = selfd.column_headers[0]
    assert all(selfd.column(colname) == selfd[:, 0]), "Failed direct indexing versus column method"
    assert all(selfd[:, 0] == data[:, 0]), "Failed direct idnexing versusus direct array index"
    assert all(selfd[:, [0, 1]] == data), "Failed direct list indexing"
    assert all(selfd[::2, :] == data[::2]), "Failed slice indexing rows"
    assert all(selfd[colname] == data[:, 0]), "Failed direct indexing by column name"
    assert all(selfd[:, colname] == data[:, 0]), "Failed fallback indexing by column name"
    assert selfd[25, 1] == 645.0, "Failed direct single cell index"
    assert selfd[25, "Y-Data"] == 645.0, "Failoed single cell index direct"
    assert selfd["Y-Data", 25] == 645.0, "Failoed single cell fallback index order"
    selfd["X-Dat"] = [11, 12, 13, 14, 15]
    assert selfd["X-Dat", 2] == 13, "Failed indexing of metadata lists with tuple"
    assert selfd["X-Dat"][2] == 13, "Failed indexing of metadata lists with double indices"
    d = Data(np.ones((10, 10)))
    d[0, 0] = 5  # Index by tuple into data
    d["Column_1", 0] = 6  # Index by column name, row into data
    d[0, "Column_2"] = 7  # Index by row, column name into data
    d["Column_3"] = [1, 2, 3, 4]  # Create a metadata
    d["Column_3", 2] = 2  # Index existing metadata via tuple
    d.metadata[0, 5] = 10
    d[0, 5] = 12  # Even if tuple, index metadata if already existing.
    assert np.all(
        d[0] == np.array([5, 6, 7, 1, 1, 1, 1, 1, 1, 1])
    ), f"setitem on Data to index into Data.data failed.\n{d[0]}"
    assert d.metadata["Column_3"] == [1, 2, 2, 4], "Tuple indexing into metadata Failed."
    assert d.metadata[0, 5] == 12, "Indexing of pre-existing metadta keys rather than Data./data failed."
    assert d.metadata[1] == [1, 2, 2, 4], "Indexing metadata by integer failed."
    with pytest.raises(KeyError):
        selfd["Bad key"]
    selfd["tmp"] = np.linspace(1, 9, 9).reshape((3, 3))
    assert all(selfd["tmp", 1] == np.array([4, 5, 6]))
    selfd["tmp", 1, 1] = -5
    assert all(selfd["tmp", 1] == np.array([4, -5, 6]))
    assert all(selfd["X-Data"] == selfd.x)


def test_metadata():
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    selfd3 = Data(path.join(__home__, "..", "sample-data", "New-XRay-Data.dql"))
    selfd["Test"] = "This is a test"
    selfd["Int"] = 1
    selfd["Float"] = 1.0
    keys, values = zip(*selfd.items())
    assert tuple(selfd.keys()) == keys, "Keys from items() not equal to keys from keys()"
    assert tuple(selfd.values()) == values, "Values from items() not equal to values from values()"
    assert selfd["Int"] == 1
    assert selfd["Float"] == 1.0
    assert selfd["Test"] == selfd.metadata["Test"]
    assert selfd.metadata._typehints["Int"] == "I32"
    assert len(selfd.dir()) == 6, f"Failed meta data directory listing ({selfd.dir()})"
    assert len(selfd3["Temperature"]) == 7, "Regular expression metadata search failed"


def test_len():
    # Check that length of the column is the same as length of the data
    selfd = Data(path.join(path.dirname(__file__), "..", "CoreTest.dat"), setas="xy")
    assert len(Data()) == 0, "Empty DataFile not length zero"
    assert len(selfd.column(0)) == len(selfd), "Column 0 length not equal to DataFile length"
    assert len(selfd) == selfd.data.shape[0], "DataFile length not equal to data.shape[0]"
    # Check that self.column_headers returns the right length
    assert len(selfd.column_headers) == selfd.data.shape[1], "Length of column_headers not equal to data.shape[1]"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
