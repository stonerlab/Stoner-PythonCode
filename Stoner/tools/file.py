# -*- coding: utf-8 -*-
"""General fle related tools."""
from importlib import import_module
import io
import os
import pathlib
import urllib
from traceback import format_exc
from typing import Union, Sequence, Dict, Type, Tuple, Optional

from ..compat import string_types, path_types, bytes2str, str2bytes
from .widgets import fileDialog
from .classes import subclasses
from ..core.exceptions import StonerLoadError, StonerUnrecognisedFormat
from ..core.base import regexpDict, metadataObject

from ..core.Typing import Filename

__all__ = [
    "file_dialog",
    "get_file_name_type",
    "auto_load_classes",
    "get_mime_type",
    "FileManager",
    "SizedFileManager",
    "URL_SCHEMES",
]

try:
    from magic import Magic as filemagic, MAGIC_MIME_TYPE
except ImportError:
    filemagic = None

URL_SCHEMES = ["http", "https"]


def file_dialog(
    mode: str, filename: Filename, filetype: Union[Type[metadataObject], str], baseclass: Type[metadataObject]
) -> Union[pathlib.Path, Sequence[pathlib.Path], None]:
    """Create a file dialog box for loading or saving ~b DataFile objects.

    Args:
        mode (string):
            The mode of the file operation  'r' or 'w'
        filename (str, Path, bool):
            Tje starting filename
        filetype (submclass of metadataObject, string):
            The filetype to open with - used to selectr file patterns
        basclass (subclass of metadataObject):
            Type of object we're looking to create'

    Returns:
        (pathlib.PurePath or None):
            A filename to be used for the file operation.
    """
    # Wildcard pattern to be used in file dialogs.

    descs: Dict = {"*.*": "All Files"}
    for p in filetype.patterns:  # pylint: disable=not-an-iterable
        descs[p] = filetype.__name__ + " file"
    for c in subclasses(baseclass):  # pylint: disable=E1136, E1133
        for p in subclasses(baseclass)[c].patterns:  # pylint: disable=unsubscriptable-object
            if p in descs:
                descs[p] += (
                    ", " + subclasses(baseclass)[c].__name__ + " file"  # pylint: disable=E1136
                )  # pylint: disable=unsubscriptable-object
            else:
                descs[p] = subclasses(baseclass)[c].__name__ + " file"  # pylint: disable=unsubscriptable-object

    patterns = descs

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
    filename = fileDialog.openDialog(start=dirname, mode=mode, patterns=patterns)
    return filename if filename else None


def get_file_name_type(
    filename: Filename, filetype: Union[Type[metadataObject], str], parent: Type[metadataObject]
) -> Tuple[pathlib.PurePath, Type[metadataObject]]:
    """Rationalise a filename and filetype."""
    if isinstance(filename, string_types):
        filename = pathlib.Path(filename)
    if isinstance(filetype, string_types):  # We can specify filetype as part of name
        try:
            filetype = regexpDict(subclasses(parent))[filetype]  # pylint: disable=E1136
        except KeyError:
            parts = filetype.split(".")
            mod = ".".join(parts[:-1])
            try:
                mod = import_module(mod)
                filetype = getattr(mod, parts[-1])
            except (ImportError, AttributeError) as err:
                raise ValueError(f"Unable to import {filetype}") from err
            if not issubclass(filetype, parent):
                raise ValueError(f"{filetype} is  not a subclass of DataFile.")
    if filename is None or (isinstance(filename, bool) and not filename):
        if filetype is None:
            filetype = parent
        filename = file_dialog("r", filename, filetype, parent)
    elif isinstance(filename, io.IOBase):  # Opened file
        filename = filename.name
    try:
        if not filename.exists():
            raise IOError(f"Cannot find {filename} to load")
    except AttributeError as err:
        raise IOError(f"Unable to tell if file exists - {type(filename)}") from err
    return filename, filetype


def auto_load_classes(
    filename: Filename,
    baseclass: Type[metadataObject],
    debug: bool = False,
    args: Optional[Tuple] = None,
    kargs: Optional[Dict] = None,
) -> Type[metadataObject]:
    """Work through subclasses of parent to find one that will load this file."""
    mimetype = get_mime_type(filename, debug=debug)
    args = args if args is not None else ()
    kargs = kargs if kargs is not None else {}
    for cls in subclasses(baseclass).values():  # pylint: disable=E1136, E1101
        cls_name = cls.__name__
        if debug:
            print(cls_name)
        try:
            if mimetype is not None and mimetype not in cls.mime_type:  # short circuit for non-=matching mime-types
                if debug:
                    print(f"Skipping {cls_name} due to mismatcb mime type {cls.mime_type}")
                continue
            test = cls()
            if "_load" not in cls.__dict__:  # No local _load method
                continue
            if debug and filemagic is not None:
                print(f"Trying: {cls_name} =mimetype {test.mime_type}")

            test = test._load(filename, auto_load=False, *args, **kargs)
            if test is None:
                raise SyntaxError(f"Class {cls_name}'s _load returned None !!")
            try:
                kargs = test._kargs
                delattr(test, "_kargs")
            except AttributeError:
                pass

            if debug:
                print("Passed Load")
            if isinstance(test, metadataObject):
                test["Loaded as"] = cls_name
            if debug:
                print(f"Test matadata: {test.metadata}")

            break
        except StonerLoadError as e:
            if debug:
                print(f"Failed Load: {e}")
            continue
        except UnicodeDecodeError:
            print(f"{cls, filename} Failed with a uncicode decode error for { format_exc()}\n")
            continue
    else:
        raise StonerUnrecognisedFormat(
            f"Ran out of subclasses to try and load {filename} (mimetype={mimetype}) as."
            + f" Recognised filetype are:{list(subclasses(baseclass).keys())}"  # pylint: disable=E1101
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


class FileManager:

    """Simple context manager that allows opening files or working with alreadt open string buffers."""

    def __init__(self, filename, *args, **kargs):
        """Store the parameters passed to the context manager."""
        self.filename = filename
        self.args = args
        self.kargs = kargs
        self.file = None
        self.binary = len(args) > 0 and args[0][-1] == "b"
        if isinstance(filename, path_types):
            parsed = urllib.parse.urlparse(str(filename))
            if parsed.scheme not in URL_SCHEMES:
                filename = pathlib.Path(filename)
            else:
                filename = urllib.request.urlopen(filename)
        if isinstance(filename, path_types):
            self.mode = "open"
        elif isinstance(filename, io.IOBase):
            if not hasattr(filename, "response"):
                if self.binary:
                    self.mode = "bytes"
                    self.filename = str2bytes(filename.read())
                else:
                    self.filename = bytes2str(filename.read())
                    self.mode = "text"
                filename.response = self.filename
            else:
                if self.binary:
                    self.mode = "bytes"
                    self.filename = str2bytes(filename.response)
                else:
                    self.filename = bytes2str(filename.response)
                    self.mode = "text"
        elif isinstance(filename, bytes):
            if len(args) > 0 and args[0][-1] == "b":
                self.filename = filename
                self.mode = "bytes"
            else:
                self.filename = bytes2str(filename)
                self.mode = "text"
        else:
            raise TypeError(f"Unrecognised filename type {type(filename)}")

    def __enter__(self):
        """Either open the file or reset the buffer."""
        if self.mode == "open":
            self.file = open(self.filename, *self.args, **self.kargs)
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
