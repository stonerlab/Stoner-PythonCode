# -*- coding: utf-8 -*-
"""Example of Folder operations."""
import re
from pathlib import Path

from Stoner import __datapath__, DataFolder, Data
from Stoner.compat import Hyperspy_ok

fldr = DataFolder(__datapath__)
del fldr["bad_data"]
if not Hyperspy_ok:
    del fldr[".*emd$"]
# .all() will recursely iterate all files and groups, depth first.
for ix, (fname, data) in enumerate(fldr.all()):
    print(ix, fname)
# filterout with a recurse filer using a regular expression and in-place
fldr.filterout(re.compile(r".*\.asc$"), recurse=True, prune=False)
print(fldr.shape)
# Now filter with a glob, recursely but making a copy
fldr2 = fldr.filter("*.dat", recurse=True, copy=True)
print(fldr2.shape)
# Count the contents in the root folder in various di
print(
    fldr2.count("**/QD*.dat"),
    fldr2.count(re.compile(r"QD")),
    fldr2.count(fldr2[1]),
)
# baseFolders implement a mapping interface to their contents
one_qd = fldr2.get("QD", Data())
two_qd = fldr2.pop("QD", Data())
# These should be the same
print(one_qd == two_qd)
# Search for a particular name
print(
    fldr2.index("**/QD*.dat"),
    fldr2.index(re.compile(r"QD")),
    fldr2.index(fldr2[5].filename),
)
# Because we shpuld plot something
fldr2[5].plot()


def print_filename(data):
    """Print the filename for the data object."""
    print(data.filename, data.shape)


(print_filename @ fldr2)()

# Remove groups without DataFiles
del fldr["attocube_scan"]
del fldr["maximus_scan"]

fldr3 = fldr.filter(
    lambda d: Path(d.filename).stem.startswith("QD"), copy=True, recurse=True
)
fldr4 = fldr3.clone
fldr4.update(fldr)
fldr4.sort(lambda d: Path(d.filename).stem)
print(fldr4.layout)
