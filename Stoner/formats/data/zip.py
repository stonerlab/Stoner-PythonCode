# -*- coding: utf-8 -*-
"""Loader for zip files."""
import zipfile as zf
import os.path as path
from traceback import format_exc

from ...compat import str2bytes, path_types
from ...core.exceptions import StonerLoadError
from ...tools import copy_into, make_Data
from ...tools.file import get_filename
from ..decorators import register_loader, register_saver
from ..utils.zip import test_is_zip


@register_loader(patterns=(".zip", 16), mime_types=("application/zip", 16), name="ZippedFile", what="Data")
def load_zipfile(new_data, *args, **kargs):
    """Load a file from the zip file, opening it as necessary."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    try:
        if isinstance(new_data.filename, zf.ZipFile):  # Loading from an ZipFile
            if not new_data.filename.fp:  # Open zipfile if necessary
                other = zf.ZipFile(new_data.filename.filename, "r")
                close_me = True
            else:  # Zip file is already open
                other = new_data.filename
                close_me = False
            member = kargs.get("member", other.namelist()[0])
            solo_file = len(other.namelist()) == 1
        elif isinstance(new_data.filename, path_types) and zf.is_zipfile(
            new_data.filename
        ):  # filename is a string that is a zip file
            other = zf.ZipFile(new_data.filename, "a")
            member = kargs.get("member", other.namelist()[0])
            close_me = True
            solo_file = len(other.namelist()) == 1
        else:
            raise StonerLoadError(f"{new_data.filename} does  not appear to be a real zip file")
    except StonerLoadError:
        raise
    except Exception as err:  # pylint: disable=W0703 # Catching everything else here
        try:
            exc = format_exc()
            other.close()
        except (AttributeError, NameError, ValueError, TypeError, zf.BadZipFile, zf.LargeZipFile):
            pass
        raise StonerLoadError(f"{new_data.filename} threw an error when opening\n{exc}") from err
    # Ok we can try reading now
    info = other.getinfo(member)
    data = other.read(info)  # In Python 3 this would be a bytes
    tmp = make_Data() << data.decode("utf-8")
    copy_into(tmp, new_data)
    # new_data.__init__(tmp << data)
    new_data.filename = path.join(other.filename, member)
    if close_me:
        other.close()
    if solo_file:
        new_data.filename = str(filename)
    return new_data


@register_saver(patterns=(".zip", 16), name="ZippedFile", what="Data")
def save(save_data, filename=None, **kargs):
    """Override the save method to allow ZippedFile to be written out to disc (as a mininmalist output).

    Args:
        filename (string or zipfile.ZipFile instance):
            Filename to save as (using the same rules as for the load routines)

    Returns:
        A copy of itsave_data.
    """
    if filename is None:
        filename = save_data.filename
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = save_data.__file_dialog("w")
    compression = kargs.pop("compression", zf.ZIP_DEFLATED)
    try:
        if isinstance(filename, path_types):  # We;ve got a string filename
            if test_is_zip(filename):  # We can find an existing zip file somewhere in the filename
                zipfile, member = test_is_zip(filename)
                zipfile = zf.ZipFile(zipfile, "a")
                close_me = True
            elif path.exists(filename):  # The fiule exists but isn't a zip file
                raise IOError(f"{filename} Should either be a zip file or a new zip file")
            else:  # Path doesn't exist, use extension of file part to find where the zip file should be
                parts = path.split(filename)
                for i, part in enumerate(parts):
                    if path.splitext(part)[1].lower() == ".zip":
                        break
                else:
                    raise IOError(f"Can't figure out where the zip file is in {filename}")
                zipfile = zf.ZipFile(path.join(*parts[: i + 1]), "w", compression, True)
                close_me = True
                member = path.join("/", *parts[i + 1 :])
        elif isinstance(filename, zf.ZipFile):  # Handle\ zipfile instance, opening if necessary
            if not filename.fp:
                filename = zf.ZipFile(filename.filename, "a")
                close_me = True
            else:
                close_me = False
            zipfile = filename
            member = ""

        if member == "" or member == "/":  # Is our file object a bare zip file - if so create a default member name
            if len(zipfile.namelist()) > 0:
                member = zipfile.namelist()[-1]
                save_data.filename = path.join(filename, member)
            else:
                member = "DataFile.txt"
                save_data.filename = filename

        zipfile.writestr(member, str2bytes(str(save_data)))
        if close_me:
            zipfile.close()
    except (zipfile.BadZipFile, IOError, TypeError, ValueError) as err:
        error = format_exc()
        try:
            zipfile.close()
        finally:
            raise IOError(f"Error saving zipfile\n{error}") from err
    return save_data
