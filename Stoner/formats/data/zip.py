# -*- coding: utf-8 -*-
"""Loader for zip files."""

import fnmatch
import json
import pathlib
import zipfile as zf
from os import path
from traceback import format_exc

import chardet
import pandas as pd

from ...compat import path_types, str2bytes
from ...core.data import Data
from ...core.exceptions import StonerLoadError
from ...tools import copy_into
from ...tools.file import get_filename
from ...tools.typing import Args, Filename, Kwargs
from ..decorators import register_loader, register_saver
from ..utils.zip import test_is_zip

from ...tools.json import flatten_json, find_paths, find_parent_dicts


def _split_filename(filename: Filename, **kwargs: Kwargs) -> Filename:
    """Try to get the member and filename parts."""
    filename = pathlib.Path(filename)
    if filename.suffix == ".zip":
        return filename
    for bit in filename.parents:
        if bit.suffix == ".zip":
            kwargs["member"] = str(filename.relative_to(bit))
            return bit
    return filename


@register_loader(patterns=(".mlseq", 16), mime_types=("application/zip", 16), name="MeasureLinkFile", what="Data")
def load_measure_linkfile(new_data: Data, *args: Args, **kwargs: Kwargs) -> Data:
    """Load a MeasureLink sequence file and assemble as a data object.

    Args:
        new_data (Data):
            Data instance into whoch to load the new data.
        *args:
            Other positional arguments passed to get_filename.

    Keyword Arguments:
        **kwargs:
            Other keyword arguments passed to get_filename.

    Returns:
        (Data):
            Loaded Data instance.

    Notes:
        `.mlseq` files are actually zip archives containing a collection of json files and a flat list of sub-folders
        The subfolders contain json for the node operations and optionally (if the key HasData is True) a csv file.
    """
    filename, args, kwargs = get_filename(args, kwargs)
    if not test_is_zip(filename):
        raise StonerLoadError("Must be a zip file to load as a measurement sequence.")
    with zf.ZipFile(filename, "r") as seq:
        if "FileInfo.json" not in seq.namelist():
            raise StonerLoadError("Missing the Measurelink Sequence FileInfo.json entry")
        with seq.open("FileInfo.json", "r") as fileinfo_json:
            fileinfo = fileinfo_json.read()
            fileinfo = fileinfo.decode(chardet.detect(fileinfo)["encoding"])
        fileinfo = json.loads(fileinfo)
        new_data.metadata.update(flatten_json(fileinfo))
        with seq.open("Model.json", "r") as model_json:
            model = model_json.read()
            model = model.decode(chardet.detect(model)["encoding"])
        model = json.loads(model)
        # new_data.metadata.update(flatten_json(model))
        for ix, pth in enumerate(fnmatch.filter(seq.namelist(), "*.csv")):
            with seq.open(pth) as dataframe:
                df = pd.read_csv(dataframe)
            if ix == 0:
                data = df
            else:
                data = pd.concat([data, df])

        data = data.select_dtypes(include="number")
        new_data.data = data.values
        new_data.column_headers = list(data.columns)

        has_data = find_paths(model, "HasData", True)

    new_data.filename = filename
    return new_data


@register_loader(patterns=(".zip", 24), mime_types=("application/zip", 16), name="ZippedFile", what="Data")
def load_zipfile(new_data: Data, *args: Args, **kwargs: Kwargs) -> Data:
    """Load a file from the zip file, opening it as necessary.

    Args:
        new_data (Data):
            Data instance into whoch to load the new data.
        *args:
            Other positional arguments passed to get_filename.

    Keyword Arguments:
        **kwargs:
            Other keyword arguments passed to get_filename.

    Returns:
        (Data):
            Loaded Data instance.
    """
    filename, args, kwargs = get_filename(args, kwargs)
    if isinstance(filename, path_types):
        filename = _split_filename(filename, **kwargs)

    new_data.filename = filename
    other = None
    close_me = False
    try:
        try:
            if isinstance(filename, zf.ZipFile):
                if filename.fp:
                    other = filename
                else:
                    other = zf.ZipFile(filename.filename, "r")
                    close_me = True
            elif isinstance(filename, path_types) and zf.is_zipfile(filename):
                other = zf.ZipFile(filename, "r")
                close_me = True
            else:
                raise StonerLoadError(f"{filename} does not appear to be a real zip file")
            names = other.namelist()
            if not names:
                raise StonerLoadError("ZIP archive contains no members")
            member = kwargs.get("member", names[0])
            solo_file = len(names) == 1
            data = other.read(other.getinfo(member))
        except (OSError, KeyError, zf.BadZipFile, zf.LargeZipFile, RuntimeError) as err:
            raise StonerLoadError(f"Unable to read ZIP member: {err}") from err

        try:
            text = data.decode("utf-8")
            header = text.split("\n", 1)[0].split("\t", 1)[0].strip()
            if header not in {"TDI Format 1.5", "TDI Format=Text 1.0", "TDI Format 2.0"}:
                raise StonerLoadError("ZIP member is not supported TDI text")
            tmp = Data() << text
        except (UnicodeDecodeError, ValueError, IndexError, StopIteration) as err:
            raise StonerLoadError(f"Invalid TDI data in ZIP member: {err}") from err
        copy_into(tmp, new_data)
        new_data.filename = path.join(other.filename, member)
        if solo_file:
            new_data.filename = str(filename)
        return new_data
    finally:
        if close_me and other is not None:
            other.close()


@register_saver(patterns=(".zip", 16), name="ZippedFile", what="Data")
def save(save_data: Data, *args: Args, **kwargs: Kwargs) -> Data:
    """Override the save method to allow ZippedFile to be written out to disc (as a mininmalist output).

    Args:
        save_data (Data):
            Data instance to be saved.
        *args:
            Other positional arguments are passed to get_filename to work out the filename.

    Keyword Arguments:
        **kwargs:
            Other keyword arguments are passed to get_filename to work out the filename.

    Returns:
        A copy of the instance of Data that was saved.
    """
    filename, args, kwargs = get_filename(args, kwargs)
    compression = kwargs.pop("compression", zf.ZIP_DEFLATED)
    try:
        if isinstance(filename, path_types):  # We;ve got a string filename
            if test_is_zip(filename):  # We can find an existing zip file somewhere in the filename
                zipfile, member = test_is_zip(filename)
                zipfile = zf.ZipFile(zipfile, "a")
                close_me = True
            elif path.exists(filename):  # The fiule exists but isn't a zip file
                raise IOError(f"{filename} Should either be a zip file or a new zip file")
            else:  # Path doesn't exist, use extension of file part to find where the zip file should be
                parts = pathlib.Path(filename).parts
                for i, part in enumerate(parts):
                    if path.splitext(part)[1].lower() == ".zip":
                        break
                else:
                    raise IOError(f"Can't figure out where the zip file is in {filename}")
                zipfile = zf.ZipFile(  # pylint: disable=consider-using-with
                    path.join(*parts[: i + 1]), "w", compression, True
                )
                close_me = True
                member = path.join("/", *parts[i + 1 :])
        elif isinstance(filename, zf.ZipFile):  # Handle\ zipfile instance, opening if necessary
            if not filename.fp:
                filename = zf.ZipFile(filename.filename, "a")  # pylint: disable=consider-using-with
                close_me = True
            else:
                close_me = False
            zipfile = filename
            member = ""

        if member in ["", "/"]:  # Is our file object a bare zip file - if so create a default member name
            if len(zipfile.namelist()) > 0:
                member = zipfile.namelist()[-1]
                save_data.filename = path.join(filename, member)
            else:
                member = "DataFile.txt"
                save_data.filename = filename

        zipfile.writestr(member, str2bytes(str(save_data)))
        if close_me:
            zipfile.close()
    except (zf.BadZipFile, IOError, TypeError, ValueError) as err:
        error = format_exc()
        try:
            zipfile.close()
        finally:
            raise IOError(f"Error saving zipfile\n{error}") from err
    return save_data
