# -*- coding: utf-8 -*-
"""Stoner.compat ensures a consistent namespace for the rest of the package

Handles differences between python 2.7 and 3.x as well as some optional dependencies.

Created on Tue Jan 14 19:53:11 2014

@author: Gavin Burnell
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from sys import version_info as __vi__
from re import _pattern_type
from matplotlib import __version__ as mpl_version
from distutils.version import LooseVersion
from os import walk
from os.path import join
import fnmatch

try:
    from lmfit import Model  # pylint: disable=unused-import
    _lmfit=True
except ImportError:
    Model=object
    _lmfit=False


__all__ = ["python_v3","str2bytes","bytes2str","get_filedialog","string_types","int_types","index_types","LooseVersion","classproperty","mpl_version","_lmfit"]

# Nasty hacks to sort out some naming conventions
if __vi__[0] == 2:
    range = xrange  # NOQA pylint:disable=redefined-builtin, undefined-variable
    string_types = (str, unicode)  # NOQA pylint: disable=undefined-variable
    int_types=(int,long)  # NOQA pylint: disable=undefined-variable
    python_v3 = False

    #|Define symbvols for equivalence to Python 3
    str2bytes = str
    bytes2str = str

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
            "save": filedialog.asksaveasfilename
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)
        
elif __vi__[0] == 3:
    string_types = (str, )
    int_types=(int,)
    python_v3 = True

    def str2bytes(s):
        """Encode a unicode string into UTF-8."""
        return bytes(str(s), "utf-8")

    def bytes2str(b):
        """Decode byte string back to univcode."""
        if isinstance(b, bytes):
            return b.decode("utf-8", "ignore")
        else:
            return b

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
            "save": filedialog.asksaveasfilename
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)


index_types = string_types + int_types +(_pattern_type,)

def listdir_recursive(dirname,glob=None):
    """Generator that does a recursive file list with optional globbing."""
    for dp, _, fn in walk(dirname):
        for f in fn:
            ret=join(dp,f)
            if glob is not None:
                if not fnmatch.fnmatch(ret,glob):
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

