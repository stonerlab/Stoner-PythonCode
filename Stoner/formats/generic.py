#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile classes for soem generic file formats."""
__all__ = ["CSVFile", "HyperSpyFile", "KermitPNGFile", "TDMSFile"]
import csv
import io
import os
import re
import warnings
from collections.abc import Mapping

import numpy as np
import PIL

from ..compat import Hyperspy_ok, hs, str2bytes
from ..Core import DataFile
from ..core.exceptions import StonerLoadError
from ..core.utils import tab_delimited
from ..tools.decorators import make_Data, register_loader, register_saver, lookup_index
from ..tools.file import FileManager


@register_loader(patterns=["*.txt", "*.tdi", "*.dat"], priority=1, description="Stoner native Tagged Data file format")
def TDIFile(filename, **kargs):
    """Actually load the data from disc assuming a .tdi file format.

    Args:
        filename (str):
            Path to filename to be loaded. If None or False, a dialog bax is raised to ask for the filename.

    Returns:
        DataFile:
            A copy of the newly loaded :py:class`DataFile` object.

    Exceptions:
        StonerLoadError:
            Raised if the first row does not start with 'TDI Format 1.5' or 'TDI Format=1.0'.

    Note:
        The *_load* methods shouldbe overidden in each child class to handle the process of loading data from
        disc. If they encounter unexpected data, then they should raise StonerLoadError to signal this, so that
        the loading class can try a different sub-class instead.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    with FileManager(instance.filename, "r", encoding="utf-8", errors="ignore") as datafile:
        line = datafile.readline()
        if line.startswith("TDI Format 1.5"):
            fmt = 1.5
        elif line.startswith("TDI Format=Text 1.0"):
            fmt = 1.0
        else:
            raise StonerLoadError("Not a TDI File")

        datafile.seek(0)
        reader = csv.reader(datafile, dialect=tab_delimited())
        cols = 0
        for ix, metadata in enumerate(reader):
            if ix == 0:
                row = metadata
                continue
            if len(metadata) < 1:
                continue
            if cols == 0:
                cols = len(metadata)
            if len(metadata) > 1:
                max_rows = ix + 1
            if "=" in metadata[0]:
                instance.metadata.import_key(metadata[0])
        col_headers_tmp = [x.strip() for x in row[1:]]
        for ix, c in enumerate(col_headers_tmp):
            header = c
            i = 1
            while header in col_headers_tmp[:ix]:
                header = f"{c}_{i}"
                i += 1
            col_headers_tmp[ix] = header
        with warnings.catch_warnings():
            datafile.seek(0)
            warnings.filterwarnings("ignore", "Some errors were detected !")
            data = np.genfromtxt(
                datafile,
                skip_header=1,
                usemask=True,
                delimiter="\t",
                usecols=range(1, cols),
                invalid_raise=False,
                comments="\0",
                missing_values=[""],
                filling_values=[np.nan],
                max_rows=max_rows,
            )
    if data.ndim < 2:
        data = np.ma.atleast_2d(data)
    retain = np.all(np.isnan(data), axis=1)
    instance.data = data[~retain]
    instance["TDI Format"] = fmt
    if instance.data.ndim == 2 and instance.data.shape[1] > 0:
        instance.column_headers = col_headers_tmp
    return instance


@register_saver(name="TDIFile", priority=1)
def TDISaver(instance, filename=None, **kargs):
    """Save a string representation of the current DataFile object into the file 'filename'.

    Args:
        filename (string, bool or None):
            Filename to save data as, if this is None then the current filename for the object is used. If this
            is not set, then then a file dialog is used. If filename is False then a file dialog is forced.
        as_loaded (bool,str):
            If True, then the *Loaded as* key is inspected to see what the original class of the DataFile was
            and then this class' save method is used to save the data. If a str then
            the keyword value is interpreted as the name of a subclass of the the current DataFile.

    Returns:
        instance:
            The current :py:class:`DataFile` object
    """
    if filename is None:
        filename = instance.filename
    # Normalise the extension to ensure it's something we like...
    info = lookup_index("TDIFile")
    filename, ext = os.path.splitext(filename)
    if f"*.{ext}" not in info.patterns:  # pylint: disable=unsupported-membership-test
        ext = info.patterns[0][2:]  # pylint: disable=unsubscriptable-object
    filename = f"{filename}.{ext}"
    header = ["TDI Format 1.5"]
    header.extend(instance.column_headers[: instance.data.shape[1]])
    header = "\t".join(header)
    mdkeys = sorted(instance.metadata)
    if len(mdkeys) > len(instance):
        mdremains = mdkeys[len(instance) :]
        mdkeys = mdkeys[0 : len(instance)]
    else:
        mdremains = []
    mdtext = np.array([instance.metadata.export(k) for k in mdkeys])
    if len(mdtext) < len(instance):
        mdtext = np.append(mdtext, np.zeros(len(instance) - len(mdtext), dtype=str))
    data_out = np.column_stack([mdtext, instance.data.values])
    fmt = ["%s"] * data_out.shape[1]
    with io.open(filename, "w", errors="replace") as f:
        np.savetxt(f, data_out, fmt=fmt, header=header, delimiter="\t", comments="")
        for k in mdremains:
            f.write(instance.metadata.export(k) + "\n")  # (str2bytes(instance.metadata.export(k) + "\n"))

    instance.filename = filename
    return instance


def _delim_detect(line):
    """Detect a delimiter in a line.

    Args:
        line(str):
            String to search for delimiters in.

    Returns:
        (str):
            Delimiter to use.

    Raises:
        StnerLoadError:
            If delimiter cannot be located.
    """
    quotes = re.compile(r"([\"\'])[^\1]*\1")
    line = quotes.sub("", line)  # Remove quoted strings first
    current = (None, len(line))
    for delim in "\t ,;":
        try:
            idx = line.index(delim)
        except ValueError:
            continue
        if idx < current[1]:
            current = (delim, idx)
    if current[0] is None:
        raise StonerLoadError("Unable to find a delimiter in the line")
    return current[0]


@register_loader(
    patterns=["*.csv", "*.txt"], mime_types=["application/csv", "text/plain"], priority=128, description="csv file",
)
def CSVFile(filename, **kargs):
    """Load generic deliminated files.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Keyword Arguments:
        header_line (int): The line in the file that contains the column headers.
            If None, then column headers are auotmatically generated.
        data_line (int): The line on which the data starts
        data_delim (string): Thge delimiter used for separating data values
        header_delim (strong): The delimiter used for separating header values

    Returns:
        A copy of the current object after loading the data.
    """
    _defaults = kargs.pop("_defaults", {"header_line": 0, "data_line": 1, "header_delim": ",", "data_delim": ","},)
    header_line = kargs.pop("header_line", _defaults["header_line"])
    data_line = kargs.pop("data_line", _defaults["data_line"])
    data_delim = kargs.pop("data_delim", _defaults["data_delim"])
    header_delim = kargs.pop("header_delim", _defaults["header_delim"])
    instance = kargs.pop("instance", make_Data())

    if filename is None or not filename:
        instance.get_filename("r")
    else:
        instance.filename = filename
    if data_delim is None or header_delim is None:  #
        with FileManager(instance.filename, "r") as datafile:
            lines = datafile.readlines()
        if header_line is not None and header_delim is None:
            header_delim = _delim_detect(lines[header_line])
        if data_line is not None and data_delim is None:
            data_delim = _delim_detect(lines[data_line])
    with FileManager(instance.filename, "r") as datafile:
        if header_line is not None:
            try:
                for ix, line in enumerate(datafile):
                    if ix == header_line:
                        break
                else:
                    raise StonerLoadError("Ran out of file before readching header")
                header = line.strip()
                column_headers = next(csv.reader(io.StringIO(header), delimiter=header_delim))
                data = np.genfromtxt(datafile, delimiter=data_delim, skip_header=data_line - header_line)
            except (TypeError, ValueError, csv.Error, StopIteration, UnicodeDecodeError,) as err:
                try:
                    filename.seek(0)
                except AttributeError:
                    pass
                raise StonerLoadError("Header and data on the same line") from err
        else:  # Generate
            try:
                data = np.genfromtxt(datafile, delimiter=data_delim, skip_header=data_line)
            except (TypeError, ValueError) as err:
                try:
                    filename.seek(0)
                except AttributeError:
                    pass
                raise StonerLoadError("Failed to open file as CSV File") from err
            column_headers = ["Column" + str(x) for x in range(np.shape(data)[1])]
    instance.data = data
    instance.column_headers = column_headers
    instance._kargs = kargs
    return instance


@register_loader(
    patterns=["*.csv", "*.txt"],
    mime_types=["application/csv", "text/plain"],
    priority=256,
    description="just numbers",
)
def JustNumbers(filename, **kargs):
    kargs.setdefault(
        "_defaults", {"header_line": None, "data_line": 0, "header_delim": None, "data_delim": None},
    )
    return CSVFile(filename, **kargs)


def _png_check_signature(instance, filename):
    """Check that this is a PNG file and raie a StonerLoadError if not."""
    try:
        with FileManager(filename, "rb") as test:
            sig = test.read(8)
        sig = [x for x in sig]
        if getattr(instance, "debug", False):
            print(sig)
        if sig != [137, 80, 78, 71, 13, 10, 26, 10]:
            raise StonerLoadError("Signature mismatrch")
    except (StonerLoadError, IOError, io.UnsupportedOperation) as err:
        raise StonerLoadError("Not a PNG file!")
    return True


@register_loader(
    patterns=["*.png"], mime_types=["image/png"], priority=16, description="Kerr Microscope PNG File",
)
def KermitPNGFile(filename, **kargs):
    """PNG file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    _png_check_signature(instance, filename)
    try:
        with PIL.Image.open(instance.filename, "r") as img:
            for k in img.info:
                instance.metadata[k] = img.info[k]
            instance.data = np.asarray(img)
    except IOError as err:
        try:
            filename.seek(0)
        except AttributeError:
            pass
        raise StonerLoadError("Unable to read as a PNG file.") from err
    return instance


@register_saver(name="KermitPNGFile", priority=16)
def to_png(instance, filename=None, **kargs):
    """Override the save method to allow KermitPNGFiles to be written out to disc.

    Args:
        filename (string): Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string): Record deliniminator (defaults to a comma)

    Returns:
        A copy of itinstance.
    """
    if filename is None:
        filename = instance.filename
    metadata = PIL.PngImagePlugin.PngInfo()
    for k in instance.metadata:
        parts = instance.metadata.export(k).split("=")
        key = parts[0]
        val = str2bytes("=".join(parts[1:]))
        metadata.add_text(key, val)
    img = PIL.Image.fromarray(instance.data)
    img.save(filename, "png", pnginfo=metadata)
    instance.filename = filename
    return instance


try:  # Optional tdms support
    from nptdms import TdmsFile

    @register_loader(
        patterns=["*.tdms"], mime_types=["application/octet-stream"], priority=16, description="NI TDMS File",
    )
    def TDMSFile(filename, **kargs):
        """TDMS file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itinstance after loading the data.
        """
        instance = kargs.pop("instance", make_Data())
        instance.filename = filename
        # Open the file and read the main file header and unpack into a dict

        try:
            if isinstance(filename, io.IOBase):  # We need to clone this into a spare buffer to stop it being closed:
                if filename.seekable():
                    pos = filename.tell()
                buffer = getattr(filename, "buffer", filename.read())
                filename.buffer = buffer
                if isinstance(filename, io.TextIOBase):
                    buffer = io.StringIO(buffer)
                else:
                    buffer = io.BytesIO(buffer)
                if filename.seekable():
                    filename.seek(pos)
            else:
                buffer = filename
            f = TdmsFile(buffer)
            for grp in f.groups():
                if grp.path == "/":
                    pass  # skip the rooot group
                elif grp.path == "/'TDI Format 1.5'":
                    tmp = DataFile(grp.as_dataframe())
                    instance.data = tmp.data
                    instance.column_headers = tmp.column_headers
                    instance.metadata.update(grp.properties)
                else:
                    tmp = DataFile(grp.as_dataframe())
                    instance.data = tmp.data
                    instance.column_headers = tmp.column_headers
        except (IOError, ValueError, TypeError, StonerLoadError) as err:
            raise StonerLoadError("Not a TDMS File") from err
        return instance


except ImportError:
    TDMSFile = make_Data()
if Hyperspy_ok:

    def _unpack_meta(instance, root, value):
        """Recursively unpack a nested dict of metadata and append keys to instance.metadata."""
        if isinstance(value, Mapping):
            for item in value.keys():
                if root != "":
                    _unpack_meta(instance, f"{root}.{item}", value[item])
                else:
                    _unpack_meta(instance, f"{item}", value[item])
        else:
            instance.metadata[root] = value

    def _unpack_axes(instance, ax_manager):
        """Unpack the axes managber as metadata."""
        for ax in ax_manager.signal_axes:
            for k in instance._axes_keys:
                instance.metadata[f"{ax.name}.{k}"] = getattr(ax, k)

    @register_loader(
        patterns=["*.emd", "*.dm4"], mime_types=["application/x-hdf"], priority=64, description="Hyperspy file",
    )
    def HyperSpyFile(filename, **kargs):
        """Load HyperSpy file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itinstance after loading the data.
        """
        _axes_keys = [
            "name",
            "scale",
            "low_index",
            "low_value",
            "high_index",
            "high_value",
        ]
        instance = kargs.pop("instance", make_Data())
        instance.filename = filename
        # Open the file and read the main file header and unpack into a dict
        try:
            load = hs.load
        except AttributeError:
            try:
                from hyperspy import api

                load = api.load
            except (ImportError, AttributeError) as err:
                raise ImportError("Panic over hyperspy") from err
        try:
            signal = load(instance.filename)
            if not isinstance(signal, hs.signals.Signal2D):
                raise StonerLoadError("Not a 2D signal object - aborting!")
        except Exception as err:  # pylint: disable=W0703 Pretty generic error catcher
            try:
                filename.seek(0)
            except (io.UnsupportedOperation, AttributeError):
                pass
            raise StonerLoadError(f"Not readable by HyperSpy error was {err}") from err
        instance.data = signal.data
        _unpack_meta(instance, "", signal.metadata.as_dictionary())
        _unpack_axes(instance, signal.axes_manager)

        return instance


else:
    HyperSpyFile = DataFile
