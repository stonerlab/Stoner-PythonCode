# -*- coding: utf-8 -*-
"""Example of Folder operations."""
import re

from Stoner import __datapath__, DataFolder, Data

fldr = DataFolder(__datapath__)
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
# baseFolders implement a mapping interface to thier contents
oneQD = fldr2.get("QD", Data())
twoQD = fldr2.pop("QD", Data())
# These should be the same
print(oneQD.metadata == twoQD.metadata)
# Search for a particular name
print(
    fldr2.index("**/QD*.dat"),
    fldr2.index(re.compile(r"QD")),
    fldr2.index(fldr2[5].filename),
)
# Because we shpuld plot something
fldr2[5].plot()
