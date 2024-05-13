# -*- coding: utf-8 -*-
"""Support routine for reading zip files."""
import zipfile as zf
from os import path


def test_is_zip(filename, member=""):
    """Recursively searches for a zipfile in the tree.

    Args:
        filename (str):
            Path to test whether it is a zip file or not.

    Keyword Arguments:
        member (str):
            Used in recursive calls to identify the path within the zip file

    Returns:
        False or (filename,member):
            Returns False if not a zip file, otherwise the actual filename of the zip file and the nanme of the
            member within that
        zipfile.
    """
    if not filename or str(filename) == "":
        return False
    if zf.is_zipfile(filename):
        return filename, member
    part = path.basename(filename)
    newfile = path.dirname(filename)
    if newfile == filename:  # reached the end of the line
        part = filename
        newfile = ""
    if member != "":
        newmember = path.join(part, member)
    else:
        newmember = part
    return test_is_zip(newfile, newmember)
