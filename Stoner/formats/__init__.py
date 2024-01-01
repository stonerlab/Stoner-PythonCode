"""Provides functions to load files in a variety of formats.
"""
__all__ = [
    "instruments",
    "generic",
    "rigs",
    "facilities",
    "simulations",
    "attocube",
    "maximus",
    "register_loader",
    "register_saver",
    "next_loader",
    "best_saver",
    "load",
    "get_loader",
    "get_saver",
    "clear_routine",
]
from copy import copy
import io
from pathlib import Path
from . import instruments, generic, rigs, facilities, simulations, attocube, maximus
from .decorators import register_loader, register_saver, next_loader, best_saver, get_loader, get_saver, clear_routine
from ..core.exceptions import StonerLoadError
from ..tools import make_Data
from ..tools.file import get_mime_type


def load(filename, *args, **kargs):
    """Use the function based loaders to try and load a file from disk."""
    if isinstance(filename, io.IOBase):
        extension = "*"
    elif isinstance(filename, bytes):
        extension = "*"
    elif hasattr(filename, "filename"):
        extension = Path(filename.filename).suffix
    else:
        extension = Path(filename).suffix
    what = kargs.pop("what", "Data")
    mime_type = get_mime_type(filename)
    if filetype := kargs.pop("filetype", False):
        loader = get_loader(filetype)
        print(f"Direct Loading {filetype}")
        data = make_Data(what=what)
        ret = loader(data, filename, *copy(args), *copy(kargs))
        ret["Loaded as"] = loader.name
        return ret
    for loader in next_loader(extension, mime_type=mime_type, what=what):
        print(loader.name)
        data = make_Data(what=what)
        try:
            ret = loader(data, filename, *copy(args), *copy(kargs))
            ret["Loaded as"] = loader.name
            return ret
        except StonerLoadError:
            continue
    raise StonerLoadError(f"Unable to find anything that would load {filename}")
