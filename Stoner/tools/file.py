# -*- coding: utf-8 -*-
"""General file related tools."""

import io
import os
import pathlib
import sys
import urllib
import zipfile
from importlib import import_module
from traceback import format_exc
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

import h5py

from ..compat import bytes2str, path_types, str2bytes, string_types
from ..core.base import SortedMultivalueDict, metadataObject
from ..core.exceptions import StonerLoadError, StonerUnrecognisedFormat
from ..tools.typing import Filename
from .decorators import make_Data, make_Image
from .widgets import fileDialog

__all__ = [
    "get_hdf_loader",
    "file_dialog",
    "get_file_name_type",
    "auto_load_classes",
    "get_mime_type",
    "FileManager",
    "SizedFileManager",
    "URL_SCHEMES",
    "get_filename",
]

_loaders_by_type = SortedMultivalueDict()
_loaders_by_pattern = SortedMultivalueDict()
_loaders_by_name = {}
_savers_by_pattern = SortedMultivalueDict()
_savers_by_name = {}


try:
    from magic import MAGIC_MIME_TYPE
    from magic import Magic as filemagic
except ImportError:
    filemagic = None

URL_SCHEMES = ["http", "https"]


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
    if zipfile.is_zipfile(filename):
        return filename, member
    part = os.path.basename(filename)
    newfile = os.path.dirname(filename)
    if newfile == filename:  # reached the end of the line
        part = filename
        newfile = ""
    if member != "":
        newmember = os.path.join(part, member)
    else:
        newmember = part
    return test_is_zip(newfile, newmember)


def get_hdf_loader(f, default_loader=lambda *args, **kwargs: None):
    """Look inside the open hdf file for details of what class to use to read this group.

    Args:
        f (h5py.File, h5py.Group):
            Open hdf file to look for a type attribute that gives the class to use to read this group.

    Returns:
        (callable):
            Callable function that can produce an object of an appropriate class.
    """
    if "type" not in f.attrs:
        raise StonerLoadError("HDF5 Group does not specify the type attribute used to check we can load it.")
    typ = bytes2str(f.attrs.get("type", ""))
    if (typ not in globals() or not isinstance(globals()[typ], type)) and "module" not in f.attrs:
        raise StonerLoadError(
            "HDF5 Group does not specify a recognized type and does not specify a module to use to load."
        )

    if "module" in f.attrs:
        mod = import_module(bytes2str(f.attrs["module"]))
        cls = getattr(mod, typ)
        return getattr(cls, "read_hdf5", default_loader)
    return getattr(globals()[typ], "read_hdf5", default_loader)


def get_filename(args, kwargs):
    """Extract a filename from the function arguments."""
    if "filename" in kwargs:
        return kwargs.pop("filename"), args, kwargs
    if len(args):
        args = list(args)
        return args.pop(0), args, kwargs
    return None, args, kwargs


def file_dialog(mode: str, filename: Filename, filetype: str) -> Union[pathlib.Path, Sequence[pathlib.Path], None]:
    """Create a file dialog box for loading or saving ~b DataFile objects.

    Args:
        mode (string):
            The mode of the file operation  'r' or 'w'
        filename (str, Path, bool):
            The starting filename
        filetype (subclass of metadataObject, string):
            The filetype to open with - used to selectr file patterns

    Returns:
        (pathlib.PurePath or None):
            A filename to be used for the file operation.
    """
    # Wildcard pattern to be used in file dialogs.

    patterns: Dict = {"*.*": "All Files"}
    patterns.update(dialog_patterns(name=filetype))

    if isinstance(filename, string_types):
        filename = pathlib.Path(filename)

    if filename is not None and not isinstance(filename, bool):
        dirname = filename.parent
        filename = filename.name
    else:
        filename = ""
        dirname = ""
    if "r" in mode:
        mode = "OpenFile"
    elif "w" in mode:
        mode = "SaveFile"
    else:
        mode = "SelectDirectory"
    filename = fileDialog.open_dialog(start=dirname, mode=mode, patterns=patterns)
    return filename if filename else None


def get_file_name_type(
    filename: Filename, filetype: Union[Type[metadataObject], str], parent: Type[metadataObject]
) -> Tuple[pathlib.PurePath, Type[metadataObject]]:
    """Rationalise a filename and filetype."""
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    if isinstance(filetype, str):  # We can specify filetype as part of name
        filetype = get_loader(filetype).name
    if filename is None or (isinstance(filename, bool) and not filename):
        if filetype is None:
            filetype = parent
        filename = file_dialog("r", filename, filetype)
    elif isinstance(filename, io.IOBase):  # Opened file
        filename = filename.name
    # try:
    #     if not filename.exists():
    #         raise IOError(f"Cannot find {filename} to load")
    # except AttributeError as err:
    #     raise IOError(f"Unable to tell if file exists - {type(filename)}") from err
    return filename, filetype


def auto_load_classes(
    filename: Filename,
    baseclass: Type[metadataObject],
    debug: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
) -> Type[metadataObject]:
    """Work through subclasses of parent to find one that will load this file."""
    mimetype = get_mime_type(filename, debug=debug)
    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}
    match filename:
        case str() | pathlib.Path():
            pattern = pathlib.Path(filename).suffix
        case _ if hasattr(filename, "url"):
            pth = urllib.parse.urlparse(filename.url).path
            pattern = pathlib.Path(pth).suffix
        case bytes() | io.StringIO() | io.BytesIO() | bool():
            pattern = "*"
        case io.IOBase() if hasattr(filename, "name"):
            pattern = pathlib.Path(filename.name).suffix
        case h5py.File():
            pattern = pathlib.Path(filename.filename).suffix
        case h5py.Group():
            pattern = pathlib.Path(filename.file.filename).suffix
        case zipfile.ZipFile():
            pattern = pathlib.Path(filename.filename).suffix
        case _:
            raise TypeError(f"Unable to figure out how to deal with {filename}")
    pattern = pattern.lower()
    for loader in next_filer(pattern, mimetype, what=baseclass):
        try:
            match baseclass:
                case "Data":
                    test = make_Data()
                case "Image":
                    test = make_Image()
                case _:
                    raise ValueError(f"Unable to figure out what data type to look at for {baseclass}")
            test = loader(test, filename, *args, **kwargs)
            try:
                kwargs = test._kwargs
                delattr(test, "_kwargs")
            except AttributeError:
                pass

            if debug:
                print("Passed Load")
            if isinstance(test, metadataObject):
                test["Loaded as"] = loader.name
            if debug:
                print(f"Test metadata: {test.metadata}")

            break
        except StonerLoadError as e:
            if debug:
                print(f"Failed Load: {e}")
            continue
        except UnicodeDecodeError:
            print(f"{loader.name, filename} Failed with a uncicode decode error for {format_exc()}\n")
            continue
    else:
        raise StonerUnrecognisedFormat(
            f"Ran out of subclasses to try and load {filename} (mimetype={mimetype}) as."
            + " Recognised filetype are:{','.join(_loaders_by_type.keys())}"
        )
    return test


def get_mime_type(filename: Union[pathlib.Path, str], debug: bool = False) -> Optional[str]:
    """Get the mime type of the file if filemagic is available."""
    if (
        filemagic is not None
        and isinstance(filename, path_types)
        and urllib.parse.urlparse(str(filename)).scheme not in URL_SCHEMES
    ):
        with filemagic(flags=MAGIC_MIME_TYPE) as m:
            mimetype = m.id_filename(str(filename))
        if debug:
            print(f"Mimetype:{mimetype}")
    else:
        mimetype = None
    return mimetype


def next_filer(
    pattern: Optional[str] = "*",
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
    what: Optional[str] = None,
    mode: str = "load",
) -> Callable:
    """Find possible loaders and yield them in turn.

    Keyword Args:
        pattern (str, None):
            (deault None) - if the file to load has an extension, use this.
        mime-type (str,None):
            (default None) - if we have a mime-type for the file, use this.
        what (str, None):
            (default None) - limit the return values to things that can load the specified class. If None, then
            this check is skipped.
        mode (str):
            "load" or "save" - set which caches to look at.

    Yields:
        func (callable):
            The next loader function to try.

    Notes:
        If avaialbe, iterate through all loaders that match that particular mime-type, but also match the pattern
        if it is available (which is should be!). If mime-type is not specified, then just match by pattern and if
        neither are specified, then yse the default no pattern "*".
    """
    if mode == "save":
        cache_by_type = {}
        cache_by_pattern = _savers_by_pattern
        cache_by_name = _savers_by_name
    else:
        cache_by_pattern = _loaders_by_pattern
        cache_by_type = _loaders_by_type
        cache_by_name = _loaders_by_name

    if mime_type is not None and mime_type in cache_by_type:  # Prefer mime-type if available
        for func in cache_by_type.get_value_list(mime_type):
            if pattern and pattern != "*" and pattern not in func.patterns:
                # If we have both pattern and type, match patterns too.
                continue
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    if pattern in cache_by_pattern:  # Fall back to specific pattern
        for func in cache_by_pattern.get_value_list(pattern):
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    if name in cache_by_name:
        func = cache_by_name[name]
        if not (what and what != func.what):  # If we are limiting what we can load, do that check
            yield func
    if pattern != "*":  # Fall back again to generic pattern
        for func in cache_by_pattern.get_value_list("*"):
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    return StopIteration


def dialog_patterns(pattern: str = "*", name: Optional[str] = None, what: str = "Data", mode: str = "load"):
    """Build a list of patterns and names for a dialog box."""
    out = {}
    for func in next_filer(pattern=pattern, name=name, what=what, mode=mode):
        for pat in func.patterns:
            if pat in out:
                out[pat] += f",{func.name}"
            else:
                out[pat] = f"{func.name}"
    return out


def best_saver(filename: str, name: Optional[str], what: Optional[str] = None) -> Callable:
    """Figure out the best saving routine registerd with the package."""
    if name and name in _savers_by_name:
        return _savers_by_name[name]
    match filename:
        case str() | pathlib.Path():
            extension = pathlib.Path(filename).suffix
        case io.IOBase() if hasattr(filename, "name"):
            extension = pathlib.Path(filename.name).suffix
        case h5py.File():
            extension = pathlib.Path(filename.filename).suffix
        case h5py.Group():
            extension = pathlib.Path(filename.file.filename).suffix
        case zipfile.ZipFile():
            extension = pathlib.Path(filename.filename).suffix
        case _:
            raise TypeError(f"Don't know how to get an extension pattern for a {type(filename)}")
    if extension.lower() in _savers_by_pattern:
        for _, func in _savers_by_pattern[extension.lower()]:
            if what is None or (what and what == func.what):
                return func
    for _, func in _savers_by_pattern["*"]:
        if what is None or (what and what == func.what):
            return func
    raise ValueError(f"Unable to find a saving routine for {filename}")


def get_loader(filetype, silent=False):
    """Return the loader function by name.

    Args:
        filetype (str): Filetype to get loader for.
        silent (bool): If False (default) raise a StonerLoadError if filetype doesn't have a loade.

    Returns:
        (callable): Matching loader.

    Notes:
        If the filetype is not found and contains a . then it tries to import a module with the same name int he
        hope that that defines the missing loader. If that fails to work, then either raises StonerLoadError or
        returns None, depending on *silent*.
    """
    try:
        return _loaders_by_name[filetype]
    except KeyError as err:
        if "." in filetype:
            try:
                module_name = ".".join(filetype.split(".")[:-1])
                if module_name not in sys.modules:
                    import_module(module_name)
                ret = _loaders_by_name.get(filetype, _loaders_by_name.get(filetype.split(".")[-1], None))
                if ret:
                    return ret
            except ImportError:
                pass
        if not silent:
            raise StonerLoadError(f"Cannot locate a loader function for {filetype}") from err
    return None


def get_saver(filetype, silent=False):
    """Return the saver function by name.

    Args:
        filetype (str): Filetype to get saver for.
        silent (bool): If False (default) raise a Stonersaverror if filetype doesn't have a loade.

    Returns:
        (callable): Matching saver.

    Notes:
        If the filetype is not found and contains a . then it tries to import a module with the same name int he
        hope that that defines the missing saver. If that fails to work, then either raises Stonersaverror or
        returns None, depending on *silent*.
    """
    try:
        return _savers_by_name[filetype]
    except KeyError as err:
        if "." in filetype:
            try:
                module_name = ".".join(filetype.split(".")[:-1])
                if module_name not in sys.modules:
                    import_module(module_name)
                ret = _savers_by_name.get(filetype, _savers_by_name.get(filetype.split(".")[-1], None))
                if ret:
                    return ret
            except ImportError:
                pass
        if not silent:
            raise StonerLoadError(f"Cannot locate a loader function for {filetype}") from err
    return None


def clear_routine(name, loader=True, saver=True):
    """Remove the routine with the specified name from the registered loaders and/or savers.

    Args:
        name (str): Name of routine to remove

    Keyword Arguments:
        loader (bool):
            Whether to remove tyhe loader (default True)
        saver (bool):
            Whether to remove the saver routne (default True)

    Returns:
        (dict):
            Removed kiader and saver routines.
    """
    ret = {}
    for k, lookup, pattern_lookup, type_lookup in zip(
        ["loader", "saver"],
        [_loaders_by_name, _savers_by_name],
        [_loaders_by_pattern, _savers_by_pattern],
        [_loaders_by_type, None],
    ):
        if not {"loader": loader, "saver": saver}[k]:
            continue
        if not locals()[k]:
            continue
        func = lookup.pop(name, None)
        if func is None:
            continue
        ret[k] = func
        for lookup_dict in [pattern_lookup, type_lookup]:
            if not isinstance(lookup_dict, dict):
                continue
            for _, values in _loaders_by_pattern.items():
                remove = []
                for ix, (_, loade_f) in enumerate(values):
                    if loade_f is func:
                        remove.append(ix)
                if remove:
                    remove.reverse()
                    for ix in remove:
                        del values[ix]
    return ret


class FileManager:
    """Simple context manager that allows opening files or working with already open string buffers."""

    def __init__(self, filename, *args, **kwargs):
        """Store the parameters passed to the context manager."""
        self.filename = filename
        self.args = args
        self.kwargs = kwargs
        self.file = None
        self.binary = len(args) > 0 and args[0].endswith("b")
        if isinstance(filename, path_types):
            parsed = urllib.parse.urlparse(str(filename))
            if parsed.scheme not in URL_SCHEMES:
                filename = pathlib.Path(filename)
            else:
                filename = urllib.request.urlopen(filename)  # pylint: disable=consider-using-with
        match filename:
            case str() | pathlib.Path():
                self.mode = "open"
            case io.IOBase() if hasattr(filename, "response"):
                if self.binary:
                    self.mode = "bytes"
                    self.filename = str2bytes(filename.response)
                else:
                    self.filename = bytes2str(filename.response)
                    self.mode = "text"
            case io.IOBase():
                if self.binary:
                    self.mode = "bytes"
                    self.filename = str2bytes(filename.read())
                else:
                    self.filename = bytes2str(filename.read())
                    self.mode = "text"
                filename.response = self.filename
            case bytes():
                if (len(args) > 0 and args[0][-1] == "b") or self.kwargs.pop("mode", "").endswith("b"):
                    self.filename = filename
                    self.mode = "bytes"
                else:
                    self.filename = bytes2str(filename)
                    self.mode = "text"
            case _:
                raise TypeError(f"Unrecognised filename type {type(filename)}")

    def __enter__(self):
        """Either open the file or reset the buffer."""
        if self.mode == "open":
            if len(self.args) > 0 and "b" not in self.args[0]:
                self.kwargs.setdefault("encoding", "utf-8")
            self.file = open(self.filename, *self.args, **self.kwargs)  # pylint: disable=unspecified-encoding
        elif self.mode == "text":
            self.file = io.StringIO(self.filename)
        elif self.mode == "bytes":
            self.file = io.BytesIO(self.filename)
        elif self.mode in ["bytesio", "textio"]:
            self.file = self.filename
        else:
            raise TypeError(f"Unrecognised filename type {type(self.filename)}")
        return self.file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the open file, or reset the buffer position."""
        if not self.file.closed and self.file.seekable():
            self.file.seek(0)
        if self.mode == "open":
            self.file.close()


class SizedFileManager(FileManager):
    """Context manager that figures out the size of the file as well as opening it."""

    def __enter__(self):
        """Add the file length information to the context variable."""
        super().__enter__()
        if self.mode == "open":
            length = os.stat(self.filename).st_size
        elif self.mode in ["textio", "bytesio"]:
            if self.file.seekable():
                pos = self.file.tell()
                self.file.seek(0, 2)
                length = self.file.tell()
                self.file.seek(pos)
            else:
                length = -1
        elif self.mode in ["text", "bytes"]:
            length = len(self.filename)
        else:
            length = len(self.file)
        return self.file, length


class HDFFileManager:
    """Context manager for HDF5 files."""

    def __init__(self, filename, mode="r"):
        """Initialise context handler.

        Works out the filename and group in cases the input flename includes a path to a sub group.

        Checks the file is actually an h4py file that is openable with the given mode.
        """
        self.mode = mode
        self.handle = None
        self.file = None
        self.close = True
        self.group = ""
        # Handle the case we're passed an already open h5py object
        if not isinstance(filename, path_types) or mode == "w":  # passed an already open h5py object
            self.filename = filename
            return
        # Here we deal with a string or path filename
        parts = str(filename).split(os.path.sep)
        bits = len(parts)
        for ix in range(bits):
            testname = "/".join(parts[: bits - ix])
            if os.path.exists(testname):
                filename = testname
                break

        try:
            if not mode.startswith("w"):
                with h5py.File(filename, "r"):
                    pass
        except (IOError, OSError) as err:
            raise StonerLoadError(f"{filename} not at HDF5 File") from err
        self.filename = filename

    def __enter__(self):
        """Open the hdf file with given mode and navigate to the group."""
        if isinstance(self.filename, (h5py.File, h5py.Group)):  # passed an already open h5py object
            self.handle = self.filename
            if isinstance(self.filename, h5py.Group):
                self.file = self.filename.file
            else:
                self.file = self.filename
            self.close = False
        elif isinstance(self.filename, path_types):  # Passed something to open
            handle = h5py.File(self.filename, self.mode)
            self.file = handle
            for grp in self.group.split("/"):
                if grp.strip() != "":
                    handle = handle[grp]
            self.handle = handle
        else:
            raise StonerLoadError("Note a resource that can be handled with HDF")
        return self.handle

    def __exit__(self, _type, _value, _traceback):
        """Ensure we close the hdf file no matter what."""
        if self.file is not None and self.close:
            self.file.close()
