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
import fnmatch
import os.path as path
import pathlib
import re
import string
from concurrent import futures
from os import cpu_count

from dask.distributed import Client
from numpy import array

from Stoner.compat import _pattern_type, string_types
from Stoner.tools import get_option


class _fake_future:

    """Minimal class that behaves like a simple future.

    This simply stores the function that should be exectured and its arguments and then delays executing it until
    the result() method is called.
    """

    def __init__(self, fn, *args, **kargs):
        self.fn = fn
        self.args = args
        self.kargs = kargs

    def result(self):
        """Execute the stored function call and return the result."""
        return self.fn(*self.args, **self.kargs)


class _fake_executor:

    """Minimal class to fake the bits of the executor protocol that we need."""

    def __init__(self, *args, **kargs):
        """Fake constructor."""

    def map(self, fn, *iterables):  # pylint: disable=no-self-use
        """Map over the results, yields each result in turn."""
        for item in zip(*iterables):
            yield fn(*item)

    def shutdown(self):  # pylint: disable=no-self-use
        """Fake shutdown method."""

    def submit(self, fn, *args, **kwargs):  # pylint: disable no-self-use
        """Execute a function."""
        return _fake_future(fn(*args, **kwargs))


executor_map = {
    "singleprocess": (_fake_executor, {}),
    "serial": (_fake_executor, {}),
    "threadpool": (futures.ThreadPoolExecutor, {"max_workers": cpu_count()}),
    "processpool": (futures.ProcessPoolExecutor, {"max_workers": cpu_count()}),
    "dask": (Client, {}),
}


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


def get_pool(folder=None, _model=None):
    """Get a concurrent.futures compatible executor.

    Returns:
        (futures.Executor):
            Executor on which to run the distributed job.
    """
    if isinstance(_model, str):
        _model = _model.lower()
    if getattr(folder, "executor", False):
        if folder.executor.name == _model:
            return folder.executor

    if _model is None:
        if get_option("multiprocessing"):
            if get_option("threading"):
                _model = "threadpool"
            else:
                _model = "processpool"
        else:
            _model = "singleprocess"
    executor_class, kwargs = executor_map[_model]
    executor = executor_class(**kwargs)
    executor.name = _model

    if getattr(folder, "executor", False):
        folder.executor.shutdown()
    if folder:
        setattr(folder, "executor", executor)
    return executor


def removeDisallowedFilenameChars(filename):
    """Clean characters in filenames.

    Args:
        filename (string): filename to cleanse

    Returns:
        A filename with non ASCII characters stripped out
    """
    validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return "".join([c for c in filename if c in validFilenameChars])
