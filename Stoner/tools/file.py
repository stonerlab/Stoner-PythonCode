# -*- coding: utf-8 -*-
"""General fle related tools."""
import http
import io
import os
import pathlib
import urllib
from importlib import import_module
from traceback import format_exc
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

import requests

from ..compat import bytes2str, path_types, str2bytes, string_types
from ..core.base import metadataObject, regexpDict
from ..core.exceptions import StonerLoadError, StonerUnrecognisedFormat
from ..core.Typing import Filename
from .classes import subclasses
from .decorators import lookup_index, lookup_loaders, lookup_savers
from .null import null
from .widgets import fileDialog

__all__ = [
    "file_dialog",
    "get_file_name_type",
    "get_file_type",
    "auto_load_classes",
    "get_mime_type",
    "FileManager",
    "SizedFileManager",
    "URL_SCHEMES",
]

try:
    from magic import MAGIC_MIME_TYPE
    from magic import Magic as filemagic
except ImportError:
    filemagic = None
URL_SCHEMES = ["http", "https"]


def file_dialog_registry(
    mode: str, filename: Filename, filetype: Union[None, Callable]
) -> Union[pathlib.Path, Sequence[pathlib.Path], None]:
    """Create a file dialog box for loading or saving ~b DataFile objects using the registry.

    Args:
        mode (string):
            The mode of the file operation  'r' or 'w'
        filename (str, Path, bool):
            Tje starting filename
        filetype (string or Callable Loader):
            The filetype to open with - used to selectr file patterns

    Returns:
        (pathlib.PurePath or None):
            A filename to be used for the file operation.
    """
    # Wildcard pattern to be used in file dialogs.

    descs: Dict = {"*.*": "All Files"} if mode == "r" else {}
    if filetype is not None:  # specific filetype passed in
        record = lookup_index(filetype)
        if (mode == "r" and record.loader is not None) or (mode == "w" and record.savers is not None):
            for p in record.patterns:
                descs[p] = record.description
    else:  # All registered filetypes
        lookup = lookup_loaders if mode == "r" else lookup_savers
        for filetype in lookup():
            record = lookup_index(filetype)
            for p in record.patterns:
                descs[p] = record.description
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


def file_dialog(
    mode: str, filename: Filename, filetype: Union[Type[metadataObject], str], baseclass: Type[metadataObject],
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


def get_loader_key(filename):
    """Get something  that might be used as a key to lookup a file loader."""
    if isinstance(filename, pathlib.Path):
        key = get_mime_type(filename)
        if not key:
            key = f"*{filename.suffix}"
        return key

    if isinstance(filename, requests.Response):
        return filename.headers["Content-type"]

    if isinstance(filename, io.IOBase):  # file-like object
        try:
            key = f"*{pathlib.Path(filename.name).suffix}"
        except (AttributeError, ValueError):
            key = "*.*"
        return key

    return "*.*"


def get_file_name(
    filename: Filename, filetype: Optional[Union[Callable, str]] = None
) -> Tuple[pathlib.PurePath, Type[metadataObject]]:
    """Rationalise a filename and filetype."""
    if isinstance(filename, string_types):
        filename = pathlib.Path(filename)
    if filename is None or (isinstance(filename, bool) and not filename):
        filename = file_dialog_registry("r", filename, filetype,)
    elif isinstance(filename, io.IOBase):  # Opened file
        filename = filename.name
    try:
        if not filename.exists():
            raise IOError(f"Cannot find {filename} to load")
    except AttributeError as err:
        raise IOError(f"Unable to tell if file exists - {type(filename)}") from err
    return filename


def get_file_name_type(
    filename: Filename, filetype: Union[Type[metadataObject], str], parent: Type[metadataObject],
) -> Tuple[pathlib.PurePath, Type[metadataObject]]:
    """Rationalise a filename and filetype."""
    if isinstance(filename, string_types):
        filename = pathlib.Path(filename)
    filetype = get_file_type(filetype, parent)
    if filename is None or (isinstance(filename, bool) and not filename):
        filename = file_dialog("r", filename, filetype, parent)
    elif isinstance(filename, io.IOBase):  # Opened file
        filename = filename.name
    try:
        if not filename.exists():
            raise IOError(f"Cannot find {filename} to load")
    except AttributeError as err:
        raise IOError(f"Unable to tell if file exists - {type(filename)}") from err
    return filename, filetype


def get_file_type(filetype: Union[Type[metadataObject], str], parent: Type[metadataObject]) -> Type[metadataObject]:
    """Try to ensure that the filetype parameter is an appropriate filetype class.

    Args:
        filetype (str or ubclass of metadataObject):
            The requested type to use for loading.
        parent (sublclass of metadataObject):
            The type of object we're trying to create for which this file must be a subclass.

    Returns:
        (metadataObject subclass):
            The requested  subclass.

    Raises:
        (ValueError):
            If the requested filetype is a string and cannot be imported, or the filetype isn't a subclass of the
            requested parent class.
    """
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
    if filetype is None:
        filetype = parent
    if not issubclass(filetype, parent):
        raise ValueError(f"{filetype} is  not a subclass of DataFile.")
    return filetype


def _handle_url_response(resp):
    """Decode a response object to either a bytes or str depending on the context type."""
    # See if we already have a buffer of data or not
    if hasattr(resp, "buffer"):
        data = resp.buffer
    elif isinstance(resp, http.client.HTTPResponse):
        data = resp.read()
        resp.buffer = data
    else:
        data = resp.content
        resp.buffer = data
    content_type = [x.strip() for x in resp.headers.get("Content-Type", "text/plain; charset=utf-8").split(";")]
    typ, substyp = content_type[0].split("/")
    if len(content_type) > 1 and "charset" in content_type[1] and typ == "text":
        charset = content_type[1][8:]
        data = data.decode(charset)
    elif typ == "text":
        data = bytes2str(data)
    return data, typ != "text"


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
    debug = kargs.get("debug", False)
    if isinstance(filename, io.IOBase):  # We need to stop the autoloading classes closing the file
        if not filename.seekable():  # Replace the filename with a seekable buffer
            data = _handle_url_response(filename)
            if isinstance(data, bytes):
                filename = io.BytesIO(data)
                if debug:
                    print("Replacing non seekable buffer with BytesIO")
            else:
                filename = io.StringIO(data)
                if debug:
                    print("Replacing non seekable buffer with BytesIO")
        if debug:
            print("Replacing close method to prevent unintended stream closure")
        original_close = filename.close
        filename.close = null()
    else:
        original_close = None
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
    if original_close is not None:
        filename.close = original_close  # Restore the close method now we're done messing
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
        self.buffer = None
        self.args = args
        self.kargs = kargs
        self.file = None
        self.binary = len(args) > 0 and args[0][-1] == "b"
        if isinstance(filename, path_types):
            parsed = urllib.parse.urlparse(str(filename))
            if parsed.scheme not in URL_SCHEMES:
                filename = pathlib.Path(filename)
            else:
                filename = requests.get(filename)
        if isinstance(filename, path_types):
            self.mode = "open"
        elif isinstance(filename, (http.client.HTTPResponse, requests.Response)):
            self.mode = "buffer"
            self.buffer, self.binary = _handle_url_response(filename)
        elif isinstance(filename, io.IOBase):
            self.mode = "buffer"
            try:
                pos = filename.tell()
            except (ValueError, io.UnsupportedOperation, AttributeError):
                pos = 0
            self.buffer = getattr(filename, "buffer", filename.read())
            if len(args) == 0:
                self.binary = isinstance(self.buffer, bytes)
            if filename.seekable():
                filename.seek(pos)
        elif isinstance(filename, bytes):
            self.mode = "bytes"
            self.buffer = filename
        else:
            raise TypeError(f"Unrecognised filename type {type(filename)}")
        if len(args) > 0:
            mode = kargs.get("mode", args[0])
        else:
            mode = kargs.get("mode", "r" + ("b" if self.binary else ""))
        if self.binary and mode[-1] != "b":
            raise StonerLoadError("Binary stream opened when mode was not a binary mode")
        if not self.binary and mode[-1] == "b":
            raise StonerLoadError("Text stream opened when mode was a binary mode")

    def __enter__(self):
        """Either open the file or reset the buffer."""
        if self.mode == "open":
            self.file = open(self.filename, *self.args, **self.kargs)
        elif self.buffer is not None and self.binary:
            self.file = io.BytesIO(str2bytes(self.buffer))
        elif self.buffer is not None and not self.binary:
            self.file = io.StringIO(bytes2str(self.buffer))
        else:
            raise TypeError(f"Unrecognised filename type {type(self.filename)}")
        return self.file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the open file, or reset the buffer position."""
        if self.mode == "open":
            self.file.close()


class SizedFileManager(FileManager):

    """Context manager that figures out the size of the file as well as opening it."""

    def __enter__(self):
        """Add the file length information to the context variable."""
        ret = super().__enter__()
        if self.mode == "open":
            length = os.stat(self.filename).st_size
        elif self.file.seekable():
            pos = self.file.tell()
            self.file.seek(0, 2)
            length = self.file.tell()
            self.file.seek(pos, 0)
        else:
            length = -1
        return ret, length
