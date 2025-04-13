"""Splits Brookhaven files up into seperate scan files and dumps them in a folder 'BNLSplitFiles'.

R.C.Temple

Will overwrite any files already split but that should be ok for an update.
"""

# pylint: disable=invalid-name
import os
import numpy as np

import Stoner

# Get file open ###############
while True:
    try:
        directory = input(
            "Enter the directory path where your data is stored:\n"
        )
        os.chdir(directory)
        filename = input(
            "Enter the filename (including extension) for your file\r\n"
        )
        mainFP = open(filename, "r", encoding="utf-8")
        break
    except IOError:
        print("Oops I couldn't find that file.")
if "BNLSplitFiles" not in os.listdir(directory):
    os.mkdir("BNLSplitFiles")
os.chdir("BNLSplitFiles")

# Main algorithm ###########

# writeName=re.split(r'[.]',filename)
writeFP = open(
    "title.txt", "w", encoding="utf-8"
)  # title sequence goes in this file
counter = 1  # this will label the files
for line in mainFP:
    if line[0:2] == "#S":
        if int(line.split()[1]) != counter:
            raise ValueError  # check for inconsistencies with filenames and scan numbers
        writeFP.close()
        writeFP = open(str(counter) + ".bnl", "w", encoding="utf-8")
        counter += 1
    if line[0:2] != "#C":
        writeFP.write(line)
        # ignore #C statements which are usually abort and rarely useful, they come
        # after data and before the next #S"""
writeFP.close()
mainFP.close()

# test files
filelist = os.listdir(os.getcwd())
filelist.pop(0)
print("Testing files with Stoner:")
for filename in filelist:
    if filename.split(".")[-1] == "bnl":
        d = Stoner.Data(
            filename, filetype="BNLFile"
        )  # will throw suitable errors if there are problems
        if len(np.shape(d.data)) == 1:
            print("Removing file {} due to lack of data".format(filename))
            d = 0
            os.remove(
                filename
            )  # delete files with only 1 dimensional data (or with
            # no data), they'll cause problems later
            continue
        print("{} OK".format(filename))
print("Done.")
