# -*- coding: utf-8 -*-
"""Stoner.compat ensures a consistent namespace for the rest of the package

Handles differences between python 2.7 and 3.x as well as some optional dependencies.

Created on Tue Jan 14 19:53:11 2014

@author: Gavin Burnell
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from sys import version_info as __vi__
from matplotlib import __version__ as mpl_version
from distutils.version import LooseVersion
from os import walk
from os.path import join, commonprefix, sep
import fnmatch
import numpy as np

try:
    from lmfit import Model  # pylint: disable=unused-import

    _lmfit = True
except ImportError:
    Model = object
    _lmfit = False


__all__ = [
    "python_v3",
    "str2bytes",
    "bytes2str",
    "get_filedialog",
    "string_types",
    "int_types",
    "index_types",
    "LooseVersion",
    "classproperty",
    "mpl_version",
    "_lmfit",
    "makedirs",
    "cmp",
]

# Nasty hacks to sort out some naming conventions
if __vi__[0] == 2:
    from re import _pattern_type
    from inspect import getargspec
    from __builtin__ import cmp
    from os import getcwd, path, mkdir

    def get_func_params(func):
        ret = []
        for arg in getargspec(func)[0][1:]:
            ret.append(arg)
        return ret

    range = xrange  # NOQA pylint:disable=redefined-builtin, undefined-variable
    string_types = (str, unicode)  # NOQA pylint: disable=undefined-variable
    int_types = (int, long)  # NOQA pylint: disable=undefined-variable
    python_v3 = False

    # |Define symbvols for equivalence to Python 3
    str2bytes = str
    bytes2str = str

    def bytes(arg, *args, **kargs):
        return str(arg)

    def get_filedialog(what="file", **opts):
        """Wrapper around Tk file dialog to mange creating file dialogs in a cross platform way.

        Args:
            what (str): What sort of a dialog to create - options are 'file','directory','save','files'
            **opts (dict): Arguments to pass through to the underlying dialog function.

        Returns:
            A file name or directory or list of files.
        """
        from Tkinter import Tk
        import tkFileDialog as filedialog

        r = Tk()
        r.withdraw()
        funcs = {
            "file": filedialog.askopenfilename,
            "directory": filedialog.askdirectory,
            "files": filedialog.askopenfilenames,
            "save": filedialog.asksaveasfilename,
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)

    def commonpath(paths):
        """Given a sequence of path names, returns the longest common sub-path."""

        if not paths:
            raise ValueError("commonpath() arg is an empty sequence")

        prefix = commonprefix(paths)
        prefix = prefix[::-1]
        split = prefix.index(sep)
        if split > 0:
            prefix = prefix[split:]
        return prefix[::-1]

    def makedirs(name, mode=0o777, exist_ok=False):
        """makedirs(name [, mode=0o777][, exist_ok=False])
        Super-mkdir; create a leaf directory and all intermediate ones.  Works like
        mkdir, except that any intermediate path segment (not just the rightmost)
        will be created if it does not exist. If the target directory already
        exists, raise an OSError if exist_ok is False. Otherwise no exception is
        raised.  This is recursive.
        """
        head, tail = path.split(name)
        if not tail:
            head, tail = path.split(head)
        if head and tail and not path.exists(head):
            try:
                makedirs(head, exist_ok=exist_ok)
            except FileExistsError:
                # Defeats race condition when another thread created the path
                pass
            cdir = getcwd()
            if tail == cdir:  # xxx/newdir/. exists if xxx/newdir exists
                return
        try:
            mkdir(name, mode)
        except OSError:
            # Cannot rely on checking for EEXIST, since the operating system
            # could give priority to other errors like EACCES or EROFS
            if not exist_ok or not path.isdir(name):
                raise


elif __vi__[0] == 3:

    if __vi__[1] < 7:
        from re import _pattern_type
    else:
        from re import Pattern as _pattern_type

    cmp = None
    from builtins import bytes as _bytes
    from os.path import commonpath
    from inspect import signature
    from os import makedirs

    def get_func_params(func):
        sig = signature(func)
        ret = {}
        for i, k in enumerate(sig.parameters):
            if i == 0:
                continue
            ret[k] = sig.parameters[k]
        return list(ret.keys())

    string_types = (str,)
    int_types = (int,)
    python_v3 = True

    def str2bytes(s):
        """Encode a unicode string into UTF-8."""
        return bytes(str(s), "utf-8")

    def bytes2str(b):
        """Decode byte string back to univcode."""
        if isinstance(b, bytes):
            return b.decode("utf-8", "ignore")
        return b

    bytes = _bytes

    def get_filedialog(what="file", **opts):
        """Wrapper around Tk file dialog to mange creating file dialogs in a cross platform way.

        Args:
            what (str): What sort of a dialog to create - options are 'file','directory','save','files'
            **opts (dict): Arguments to pass through to the underlying dialog function.

        Returns:
            A file name or directory or list of files.
        """
        from tkinter import Tk, filedialog

        r = Tk()
        r.withdraw()
        funcs = {
            "file": filedialog.askopenfilename,
            "directory": filedialog.askdirectory,
            "files": filedialog.askopenfilenames,
            "save": filedialog.asksaveasfilename,
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)


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


class ClassPropertyDescriptor(object):

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
