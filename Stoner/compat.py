# -*- coding: utf-8 -*-
"""Ensure a consistent namespace for the rest of the package irrespective of Python language version."""
__all__ = [
    "str2bytes",
    "bytes2str",
    "get_filedialog",
    "string_types",
    "path_types",
    "int_types",
    "index_types",
    "classproperty",
    "mpl_version",
    "_lmfit",
    "makedirs",
    "Hyperspy_ok",
    "hs",
    "hsload",
    "which",
    "commonpath",
    "_jit",
    "_dummy",
]

from sys import version_info as __vi__, modules
from os import walk, makedirs
from os.path import join, commonpath
import fnmatch
from shutil import which
from pathlib import PurePath
from packaging.version import parse as version_parse
from inspect import signature
import re

import numpy as np
import scipy as sp
import matplotlib

_lmfit = True
np_version = version_parse(np.__version__)
sp_version = version_parse(sp.__version__)
mpl_version = version_parse(matplotlib.__version__)

try:  # This only works in PY 3.11 onwards
    modules["sre_parse"] = re._parser
    modules["sre_constants"] = re._constants
    modules["sre_compile"] = re._compiler
except AttributeError:
    pass

try:
    import hyperspy as hs  # Workaround an issue in hs 1.5.2 conda packages

    try:
        hsload = hs.load
    except (RuntimeError, AttributeError):
        try:
            from hyperspy import api

            hsload = api.load
        except (ImportError, AttributeError) as err:
            raise ImportError("Panic over hyperspy") from err

    HuperSpyVersion = [int(x) for x in hs.__version__.split(".")]
    if HuperSpyVersion[0] <= 1 and HuperSpyVersion[1] <= 3:
        raise ImportError(f"Hyperspy should be version 1.4 or above. Actual version is {hs.__version__}")
    Hyperspy_ok = True
except ImportError:
    Hyperspy_ok = False
    hs = None
    hsload = None

if __vi__[1] < 7:
    from re import _pattern_type  # pylint: disable = E0611
else:
    from re import Pattern as _pattern_type  # pylint: disable = E0611


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
path_types = (str, PurePath)

# #### Monkey patch numpy for removed attributes as a compatibiliyu hack
if np_version.minor >= 20:
    np.float = float
    np.bool = np.bool_
    np.str = str
    np.bool8 = np.bool_
    np.int0 = np.intp


def str2bytes(data):
    """Encode a unicode string into UTF-8."""
    if isinstance(data, bytes):
        return data
    return bytes(str(data), "utf-8")


def bytes2str(data):
    """Decode byte string back to univcode."""
    if isinstance(data, bytes):
        return data.decode("utf-8", "ignore")
    return data


def get_filedialog(what="file", **opts):
    """Wrap around Tk file dialog to manage creating file dialogs in a cross platform way.

    Args:
        what (str): What sort of a dialog to create - options are 'file','directory','save','files'
        **opts (dict): Arguments to pass through to the underlying dialog function.

    Returns:
        A file name or directory or list of files.
    """
    from .tools.widgets import fileDialog

    funcs = {"file": "OpenFile", "directory": "SelectDirectory", "files": "OpenFiles", "save": "SaveFile"}
    if what not in funcs:
        raise RuntimeError(f"Unable to recognise required file dialog type:{what}")
    return fileDialog.openDialog(mode=funcs[what], **opts)


if np_version.major >= 2:
    int_types += (int, np.int8, np.int16, np.int32, np.int64)
elif np_version.minor >= 20:
    int_types += (int, np.int0, np.int8, np.int16, np.int32, np.int64)
else:
    int_types += (np.int, np.int0, np.int8, np.int16, np.int32, np.int64)

index_types = string_types + int_types + (_pattern_type,)


def listdir_recursive(dirname, glob=None):
    """Make a recursive file list with optional globbing."""
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
        """Initialise the descriptor."""
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


def _jit(func, *_, **__):
    """Null decorator function."""
    return func


class _dummy:
    """A class that does nothing so that float64 can be an instance of it safely."""

    def jit(self, func, *_, **__):  # pylint: disable=no-self-use
        """Null decorator function."""
        return func

    def __call__(self, *args, **kargs):
        """Handle jit lines that call the type."""
        return self.jit

    def __getitem__(self, *args, **kargs):
        """Hande jit calls to array types."""
        return self
