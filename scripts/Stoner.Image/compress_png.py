# -*- coding: utf-8 -*-
"""
Script for compressing png images using PIL optimize.

Given the folder it will compress all files in all the subfolders. This
compression is lossless. The images will not lose any definition and loaded
images will be exactly the same.

CHANGE THE FOLDER NAME BEFORE RUNNING
"""

from PIL import Image
import os
from Stoner import DataFolder
from Stoner.compat import python_v3

if python_v3:
    raw_input = input

# CHANGE THE FOLDER NAME BEFORE RUNNING!  #####################################
folder = r"C:\Users\phyrct\KermitData\test2"
#############################################


def get_size(start_path="."):
    """get directory size"""
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def pngsave(im, filename):
    """save a PNG with PIL preserving metadata.

    Thanks to blog http://blog.client9.com/2007/08/28/python-pil-and-png-metadata-take-2.html
    for code
    """
    # these can be automatically added to Image.info dict
    # they are not user-added metadata
    reserved = ("interlace", "gamma", "dpi", "transparency", "aspect")

    # undocumented class
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()

    # copy metadata into new object
    for k, v in im.info.iteritems():
        if k in reserved:
            continue
        meta.add_text(k, v, 0)

    # and save
    im.save(filename, "PNG", pnginfo=meta, optimize=True)


get_sizes = True  # calculate size of folder before and after (may slow things down)

df = DataFolder(folder, pattern="*.png")
df.flatten()
dflen = len(df)

sizebefore = -1
if get_sizes:
    sizebefore = get_size(start_path=folder)

print("found files for compression:")
print("folder: {}".format(folder))
print("no. of files: {}".format(dflen))
print("size before: {}".format(sizebefore))
raw_input("Enter to continue")

for i, fname in enumerate(df.ls):
    print("{}% complete".format(i / float(dflen) * 100))
    try:
        im = Image.open(fname)
        pngsave(im, fname)
    except Exception:
        print("Could not compress file {}".format(fname))
        q = raw_input("Would you like to continue (y/n)?")
        if q != "y":
            raise RuntimeError("Could not compress file {}".format(fname))

sizeafter = -1
if get_sizes:
    sizeafter = get_size(start_path=folder)
print("size after: {}".format(sizeafter))
print("compression: {}".format(sizeafter / float(sizebefore)))
