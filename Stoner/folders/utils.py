# -*- coding: utf-8 -*-
"""Utility functions to support :py:class:`Stoner.folders.core.objectFolder` operations."""
__all__ = [
    "pathsplit",
    "pathjoin",
    "scan_dir",
    "discard_earlier",
    "filter_files",
    "get_pool",
    "removeDisallowedFilenameChars",
]
import os.path as path
import re
import string
import fnmatch
import pathlib
from multiprocessing.pool import ThreadPool

from numpy import array
import multiprocess as multiprocessing

from Stoner.compat import string_types, _pattern_type
from Stoner.tools import get_option


def pathsplit(pth):
    """Split pth into a sequence of individual parts with path.split."""
    pth = pathlib.Path(pth)
    ret = [pth.name]
    ret.extend([x.name for x in pth.parents])
    ret.reverse()
    return [str(x) for x in ret if x != ""]


def pathjoin(*args):
    """Join a path like path.join, but then replace the path separator with a standard /."""
    if len(args) > 1:
        tmp = path.join(args[0], *args[1:])
        return tmp.replace(path.sep, "/")


def scan_dir(root):
    """Gather a list of files and directories."""
    dirs = []
    files = []
    root = pathlib.Path(root)
    for f in root.glob("*"):
        if f.is_dir():
            dirs.append(f.name)
        elif f.is_file():
            files.append(f.name)
    return dirs, files


def discard_earlier(files):
    """Discard files where a similar named file with !#### exists."""
    search = re.compile(r"^(?P<basename>.*)\!(?P<rev>\d+)(?P<ext>\.[^\.]*)$")
    dups = dict()
    ret = []
    for f in files:
        match = search.match(f)
        if match:
            fname = f"{match.groupdict()['basename']}{match.groupdict()['ext']}"
            rev = int(match.groupdict()["rev"])
            entry = dups.get(fname, [])
            entry.append((rev, f))
            dups[fname] = entry
        else:
            entry = dups.get(f, [])
            entry.append((-1, f))
            dups[f] = entry
    for f, revs in dups.items():
        rev = sorted(revs)[-1]
        ret.append(rev[1])
    return ret


def filter_files(files, patterns, keep=True):
    """Filter a list of files against include/exclusion patterns.

    Args:
        files (list of str): Filename to filter
        pattens (list of (str,regular expressions): List of patterns to consider

    Keyword Arguments:
        keep (bool): True (default) to keep matching files.

    Returns:
        (list of str): Files that pass the filter
    """
    dels = []
    for p in patterns:  # Remove excluded files
        if isinstance(p, string_types):
            for f in list(fnmatch.filter(files, p)):
                dels.append(files.index(f))
        if isinstance(p, _pattern_type):
            # For reg expts we iterate over all files, but we can't delete matched
            # files as we go as we're iterating over them - so we store the
            # indices and delete them later.
            for f in files:
                if p.search(f):
                    dels.append(files.index(f))
        index = array([not keep ^ (i in dels) for i in range(len(files))], dtype=bool)
        files = (array(files)[index]).tolist()
    return files


def get_pool(_serial=False):
    """Get a Pool and map implementation depending on options.

    Returns:
        Pool(),map: Pool object if possible and map implementation.
    """
    if get_option("multiprocessing") and not _serial:
        try:
            if get_option("threading"):
                p = ThreadPool(processes=int(multiprocessing.cpu_count() - 1))
            else:
                p = multiprocessing.Pool(int(multiprocessing.cpu_count() / 2))
            imap = p.imap
        except (ArithmeticError, AttributeError, LookupError, RuntimeError, NameError, OSError, TypeError, ValueError):
            # Fallback to non-multiprocessing if necessary
            p = None
            imap = map
    else:
        p = None
        imap = map
    return p, imap


def removeDisallowedFilenameChars(filename):
    """Clean characters in filenames.

    Args:
        filename (string): filename to cleanse

    Returns:
        A filename with non ASCII characters stripped out
    """
    validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return "".join([c for c in filename if c in validFilenameChars])
