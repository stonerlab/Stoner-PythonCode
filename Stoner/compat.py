# -*- coding: utf-8 -*-
"""Stoner.compat ensures a consistent namespace for the rest of the package

Handles differences between python 2.7 and 3.x as well as some optional dependencies.

Created on Tue Jan 14 19:53:11 2014

@author: Gavin Burnell
"""
from sys import version_info as __vi__
from os import walk, makedirs
from os.path import join, commonpath
import fnmatch
from inspect import signature, getfullargspec
from shutil import which

import numpy as np
from matplotlib import __version__ as mpl_version

try:
    from lmfit import Model  # pylint: disable=unused-import

    _lmfit = True
except ImportError:
    Model = object
    _lmfit = False

__all__ = [
    "str2bytes",
    "bytes2str",
    "get_filedialog",
    "getargspec",
    "string_types",
    "int_types",
    "index_types",
    "classproperty",
    "mpl_version",
    "_lmfit",
    "makedirs",
    "cmp",
    "hyperspy_ok",
    "which",
    "commonpath",
]

try:
    import hyperspy as hs

    hsv = [int(x) for x in hs.__version__.split(".")]
    if hsv[0] <= 1 and hsv[1] <= 3:
        raise ImportError("Hyperspy should be version 1.4 or above. Actual version is {}".format(hs.__version__))
    hyperspy_ok = True
except ImportError:
    hyperspy_ok = False


if __vi__[1] < 7:
    from re import _pattern_type  # pylint: disable = E0611
else:
    from re import Pattern as _pattern_type  # pylint: disable = E0611

cmp = None


def getargspec(*args, **kargs):
    """Wrapper for getargspec for Python V3."""
    return getfullargspec(*args, **kargs)[:4]


def get_func_params(func):
    """Get the parameters for a function."""
    sig = signature(func)
    ret = {}
    for i, k in enumerate(sig.parameters):
        if i == 0:
            continue
        ret[k] = sig.parameters[k]
    return list(ret.keys())


string_types = (str,)
int_types = (int,)


def str2bytes(s):
    """Encode a unicode string into UTF-8."""
    return bytes(str(s), "utf-8")


def bytes2str(b):
    """Decode byte string back to univcode."""
    if isinstance(b, bytes):
        return b.decode("utf-8", "ignore")
    return b


def get_filedialog(what="file", **opts):
    """Wrapper around Tk file dialog to mange creating file dialogs in a cross platform way.
    Args:
        what (str): What sort of a dialog to create - options are 'file','directory','save','files'
        **opts (dict): Arguments to pass through to the underlying dialog function.
    Returns:
        A file name or directory or list of files.
    """
    from .tools.widgets import fileDialog

    funcs = {"file": "OpenFile", "directory": "SelectDirectory", "files": "OpenFiles", "save": "SaveFile"}
    if what not in funcs:
        raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
    return fileDialog.openDialog(mode=funcs[what], **opts)


int_types += (np.int, np.int0, np.int8, np.int16, np.int32, np.int64)

index_types = string_types + int_types + (_pattern_type,)


def listdir_recursive(dirname, glob=None):
    """Generator that does a recursive file list with optional globbing."""
    for dp, _, fn in walk(dirname):
        for f in fn:
            ret = join(dp, f)
            if glob is not None:
                if not fnmatch.fnmatch(ret, glob):
                    continue
            yield ret


class ClassPropertyDescriptor:

    """Supports adding class properties."""

    def __init__(self, fget, fset=None):
        """Setup descriptor."""
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        """Implement descriptor getter."""
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()


def classproperty(func):
    """Define a property to be a class property and not an instance property."""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
