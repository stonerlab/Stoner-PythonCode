#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile classes for some generic file formats."""
import csv
import contextlib
import copy
import io
from pathlib import Path
import re
from collections.abc import Mapping
import sys
import logging
from typing import Optional
import warnings

import PIL
import numpy as np

from ...compat import str2bytes, Hyperspy_ok, hs, hsload
from ...core.array import DataArray
from ...core.exceptions import StonerLoadError
from ...core.utils import tab_delimited
from ...tools import make_Data

from ...tools.file import FileManager, get_filename

from ..decorators import register_loader, register_saver


class _refuse_log(logging.Filter):
    """Refuse to log all records."""

    def filter(self, record):
        """Do not log anything."""
        return False


@contextlib.contextmanager
def catch_sysout(*args):
    """Temporarily redirect sys.stdout and.sys.stdin."""
    stdout, stderr = sys.stdout, sys.stderr
    out = io.StringIO()
    sys.stdout, sys.stderr = out, out
    logger = logging.getLogger("hyperspy.io")
    logger.addFilter(_refuse_log)
    yield None
    logger.removeFilter(_refuse_log)
    sys.stdout, sys.stderr = stdout, stderr
    return


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
    patterns=[(".dat", 8), (".txt", 8), ("*", 8)],
    mime_types=[("application/tsv", 8), ("text/plain", 8), ("text/tab-separated-values", 8)],
    name="DataFile",
    what="Data",
)
def load_tdi_format(new_data, *args, **kargs):
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
        The *_load* methods shouldbe overridden in each child class to handle the process of loading data from
        disc. If they encounter unexpected data, then they should raise StonerLoadError to signal this, so that
        the loading class can try a different sub-class instead.
    """
    filename, args, kargs = get_filename(args, kargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    with FileManager(new_data.filename, "r", encoding="utf-8", errors="ignore") as datafile:
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
        max_rows = 0
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
                new_data.metadata.import_key(metadata[0])
        col_headers_tmp = [x.strip() for x in row[1:]]
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
    new_data.data = DataArray(data[~retain])
    new_data["TDI Format"] = fmt
    new_data["Stoner.class"] = "Data"
    if new_data.data.ndim == 2 and new_data.data.shape[1] > 0:
        new_data.column_headers = col_headers_tmp
    new_data.metadata = copy.deepcopy(new_data.metadata)  # This fixes some type issues TODO - work out why!
    return new_data


@register_saver(
    patterns=[(".dat", 8), (".txt", 8), ("*", 8)],
    name="DataFile",
    what="Data",
)
def save_tdi_format(save_data, filename):
    """Write out a DataFile to a tab delimited tdi text file."""

    header = ["TDI Format 1.5"]
    header.extend(save_data.column_headers[: save_data.data.shape[1]])
    header = "\t".join(header)
    mdkeys = sorted(save_data.metadata)
    if len(mdkeys) > len(save_data):
        mdremains = mdkeys[len(save_data) :]
        mdkeys = mdkeys[0 : len(save_data)]
    else:
        mdremains = []
    mdtext = np.array([save_data.metadata.export(k) for k in mdkeys])
    if len(mdtext) < len(save_data):
        mdtext = np.append(mdtext, np.zeros(len(save_data) - len(mdtext), dtype=str))
    data_out = np.column_stack([mdtext, save_data.data])
    fmt = ["%s"] * data_out.shape[1]
    with io.open(filename, "w", errors="replace", encoding="utf-8") as f:
        np.savetxt(f, data_out, fmt=fmt, header=header, delimiter="\t", comments="")
        for k in mdremains:
            f.write(save_data.metadata.export(k) + "\n")  # (str2bytes(save_data.metadata.export(k) + "\n"))

    save_data.filename = filename
    return save_data


@register_loader(
    patterns=[(".csv", 32), (".txt", 256)],
    mime_types=[("application/csv", 16), ("text/plain", 256), ("text/csv", 16)],
    name="CSVFile",
    what="Data",
)
def load_csvfile(new_data: "DataFile", *args, **kargs) -> "DataFile":
    """Load generic deliminated files.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Keyword Arguments:
        header_line (int): The line in the file that contains the column headers.
            If None, then column headers are automatically generated.
        data_line (int): The line on which the data starts
        data_delim (string): The delimiter used for separating data values
        header_delim (strong): The delimiter used for separating header values

    Returns:
        A copy of the current object after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    _defaults = {"header_line": 0, "data_line": 1, "header_delim": ",", "data_delim": ","}

    header_line = kargs.pop("header_line", _defaults["header_line"])
    data_line = kargs.pop("data_line", _defaults["data_line"])
    data_delim = kargs.pop("data_delim", _defaults["data_delim"])
    header_delim = kargs.pop("header_delim", _defaults["header_delim"])

    new_data.filename = filename

    if data_delim is None or header_delim is None:  #
        with FileManager(new_data.filename, "r") as datafile:
            lines = datafile.readlines()
        if header_line is not None and header_delim is None:
            header_delim = _delim_detect(lines[header_line])
        if data_line is not None and data_delim is None:
            data_delim = _delim_detect(lines[data_line])

    with FileManager(new_data.filename, "r") as datafile:
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
            except (TypeError, ValueError, csv.Error, StopIteration, UnicodeDecodeError) as err:
                raise StonerLoadError("Header and data on the same line") from err
        else:  # Generate
            try:
                data = np.genfromtxt(datafile, delimiter=data_delim, skip_header=data_line)
            except (TypeError, ValueError) as err:
                raise StonerLoadError("Failed to open file as CSV File") from err
            column_headers = ["Column" + str(x) for x in range(np.shape(data)[1])]

    new_data.data = data
    new_data.column_headers = column_headers
    new_data.metadata |= kargs
    return new_data


@register_loader(
    patterns=[(".csv", 512), (".txt", 512)],
    mime_types=[("application/csv", 512), ("text/plain", 512), ("text/csv", 512)],
    name="JustNumbers",
    what="Data",
)
def load_numbersfile(new_data: "DataFile", *args, **kargs) -> "DataFile":
    """Simple pass through for csv file."""
    return load_csvfile(new_data, *args, header_line=None, data_line=0, header_delim=None, data_delim=None)


@register_saver(patterns=[(".csv", 32), (".txt", 256)], name="CSVFile", what="Data")
def save_csvfile(save_data: "DataFile", *args, **kargs):
    """Override the save method to allow CSVFiles to be written out to disc (as a mininmalist output).

    Args:
        filename (string): Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string): Record deliniminator (defaults to a comma)
        no_header (bool): Whether to skip the headers, defaults to False (include colu,n headers)

    Returns:
        A copy of itsave_data.
    """
    filename, args, kargs = get_filename(args, kargs)
    delimiter = kargs.pop("deliminator", ",")
    if filename is None:
        filename = save_data.filename
    no_header = kargs.get("no_header", False)
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = save_data.__file_dialog("w")
    with FileManager(filename, "w") as outfile:
        spamWriter = csv.writer(outfile, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        if not no_header:
            spamWriter.writerow(save_data.column_headers)
        while i < save_data.data.shape[0]:
            spamWriter.writerow(save_data.data[i, :])
            i += 1
    save_data.filename = filename
    return save_data


@register_loader(
    patterns=[(".csv", 64), (".txt", 512)],
    mime_types=[("application/csv", 64), ("text/plain", 512), ("text/csv", 64)],
    name="JustNumbers",
    what="Data",
)
def load_justnumbers(new_data: "DataFile", *args, **kargs) -> "DataFile":
    """Load generic deliminated files with no headers or metadata.

    Args:
        filename (str):
            File to load.

    Keyword Arguments:
        data_delim (string): The delimiter used for separating data values
        header_delim (strong): The delimiter used for separating header values

    Returns:
        A copy of the current object after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    _defaults = {"header_line": None, "data_line": 0, "data_delim": None}
    for karg, val in _defaults.items():
        kargs.setdefault(karg, val)
    return load_csvfile(new_data, filename, *args, **kargs)


@register_saver(patterns=[(".csv", 32), (".txt", 256)], name="JustNumbersFile", what="Data")
def save_justnumbers(save_data: "DataFile", filename: Optional[str] = None, **kargs):
    """Override the save method to allow JustNumbersFiles to be written out to disc (as a very mininmalist output).

    Args:
        filename (string): Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string): Record deliniminator (defaults to a comma)

    Returns:
        A copy of itsave_data.
    """
    kargs["no_header"] = False
    return save_csvfile(save_data, filename, **kargs)


def _check_png_signature(filename):
    """Check that this is a PNG file and raie a StonerLoadError if not."""
    try:
        with FileManager(filename, "rb") as test:
            sig = test.read(8)
        sig = [x for x in sig]
        if sig != [137, 80, 78, 71, 13, 10, 26, 10]:
            raise StonerLoadError("Signature mismatrch")
    except (StonerLoadError, IOError) as err:
        from traceback import format_exc

        raise StonerLoadError(f"Not a PNG file!>\n{format_exc()}") from err
    return True


@register_loader(
    patterns=(".png", 16),
    mime_types=("image/png", 16),
    name="KermitPNGFile",
    what="Data",
)
def load_pngfile(new_data, *args, **kargs):
    """PNG file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    _check_png_signature(filename)
    try:
        with PIL.Image.open(new_data.filename, "r") as img:
            for k in img.info:
                new_data.metadata[k] = img.info[k]
            new_data.data = np.asarray(img)
    except IOError as err:
        raise StonerLoadError("Unable to read as a PNG file.") from err
    new_data["Stoner.class"] = "Data"  # Reset my load class
    new_data.metadata = copy.deepcopy(new_data.metadata)  # Fixes up some data types (see TDI loader)
    return new_data


@register_saver(patterns=(".png", 8), name="ImageFile", what="Image")
@register_saver(patterns=(".png", 16), name="KermitPngFile", what="Data")
def save_pngfile(save_data, filename=None, **kargs):
    """Override the save method to allow KermitPNGFiles to be written out to disc.

    Args:
        filename (string): Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string): Record deliniminator (defaults to a comma)

    Returns:
        A copy of itsave_data.
    """
    if filename is None:
        filename = save_data.filename
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = save_data.__file_dialog("w")

    metadata = PIL.PngImagePlugin.PngInfo()
    for k in save_data.metadata:
        parts = save_data.metadata.export(k).split("=")
        key = parts[0]
        val = str2bytes("=".join(parts[1:]))
        metadata.add_text(key, val)
    img = PIL.Image.fromarray(save_data.data)
    img.save(filename, "png", pnginfo=metadata)
    save_data.filename = filename
    return save_data


try:  # Optional tdms support
    from nptdms import TdmsFile

    @register_loader(
        patterns=[(".tdms", 16), (".tdms_index", 16)],
        mime_types=("application/octet-stream", 16),
        name="TDMSFile",
        what="Data",
    )
    def load_tdms(new_data, *args, **kargs):
        """TDMS file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itnew_data after loading the data.
        """
        filename, args, kargs = get_filename(args, kargs)
        filename = Path(filename)
        if filename.suffix == ".tdms_index":
            filename = filename.parent / f"{filename.stem}.tdms"  # rewrite filename for not the index file!
        new_data.filename = filename
        # Open the file and read the main file header and unpack into a dict
        try:
            f = TdmsFile(new_data.filename)
            for grp in f.groups():
                if grp.path == "/":
                    pass  # skip the rooot group
                elif grp.path == "/'TDI Format 1.5'":
                    tmp = make_Data(grp.as_dataframe())
                    new_data.data = tmp.data
                    new_data.column_headers = tmp.column_headers
                    new_data.metadata.update(grp.properties)
                else:
                    tmp = make_Data(grp.as_dataframe())
                    new_data.data = tmp.data
                    new_data.column_headers = tmp.column_headers
        except (IOError, ValueError, TypeError, StonerLoadError) as err:
            from traceback import format_exc

            raise StonerLoadError(f"Not a TDMS File \n{format_exc()}") from err

        return new_data

except ImportError:
    pass

if Hyperspy_ok:

    def _unpack_hyperspy_meta(new_data, root, value):
        """Recursively unpack a nested dict of metadata and append keys to new_data.metadata."""
        if isinstance(value, Mapping):
            for item in value.keys():
                if root != "":
                    _unpack_hyperspy_meta(new_data, f"{root}.{item}", value[item])
                else:
                    _unpack_hyperspy_meta(new_data, f"{item}", value[item])
        else:
            new_data.metadata[root] = value

    def _unpack_hyperspy_axes(newdata, ax_manager):
        """Unpack the axes managber as metadata."""
        _axes_keys = ["name", "scale", "low_index", "low_value", "high_index", "high_value"]
        for ax in ax_manager.signal_axes:
            for k in _axes_keys:
                newdata.metadata[f"{ax.name}.{k}"] = getattr(ax, k)

    @register_loader(
        patterns=[(".emd", 32), (".dm4", 32)],
        mime_types=[("application/x-hdf", 64), ("application/x-hdf5", 64)],
        name="HyperSpyFile",
        what="Data",
    )
    def hypersput_load(new_data, *args, **kargs):
        """Load HyperSpy file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itnew_data after loading the data.
        """
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            new_data.get_filename("r")
        else:
            new_data.filename = filename
        # Open the file and read the main file header and unpack into a dict
        try:
            with catch_sysout():
                signal = hsload(new_data.filename)
            if hasattr(hs, "signals"):
                Signal2D = hs.signals.Signal2D
            else:
                Signal2D = hs.api.signals.Signal2D
            if not isinstance(signal, Signal2D):
                raise StonerLoadError("Not a 2D signal object - aborting!")
        except Exception as err:  # pylint: disable=W0703 Pretty generic error catcher
            raise StonerLoadError(f"Not readable by HyperSpy error was {err}") from err
        new_data.data = signal.data
        _unpack_hyperspy_meta(new_data, "", signal.metadata.as_dictionary())
        _unpack_hyperspy_axes(new_data, signal.axes_manager)
        del new_data["General.FileIO.0.timestamp"]
        return new_data
