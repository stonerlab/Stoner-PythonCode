#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile classes for soem generic file formats."""
__all__ = ["CSVFile", "HyperSpyFile", "KermitPNGFile", "TDMSFile"]
import csv
import io
import linecache
import re
from copy import copy
from collections.abc import Mapping

import PIL
import numpy as np

from ..Core import DataFile
from ..compat import str2bytes, Hyperspy_ok, hs
from ..core.exceptions import StonerLoadError


class CSVFile(DataFile):

    """A subclass of DataFiule for loading generic deliminated text fiules without metadata."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 128  # Rather generic file format so make it a low priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.csv", "*.txt"]  # Recognised filename patterns

    _defaults = {"header_line": 0, "data_line": 1, "header_delim": ",", "data_delim": ","}

    mime_type = ["application/csv", "text/plain"]

    def _load(self, filename, *args, **kargs):
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
        defaults = copy(self._defaults)
        defaults.update(kargs)
        keep = set(kargs.keys()) - (set(self._defaults.keys()) | set(["auto_load", "filetype"]))
        kargs = {k: kargs[k] for k in keep}
        header_line = defaults["header_line"]
        data_line = defaults["data_line"]
        data_delim = defaults["data_delim"]
        header_delim = defaults["header_delim"]
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        if header_line is not None:
            try:
                header_string = linecache.getline(self.filename, header_line + 1)
                header_string = re.sub(r'["\n]', "", header_string)
                header_string.index(header_delim)
            except (ValueError, SyntaxError) as err:
                linecache.clearcache()
                raise StonerLoadError("No Delimiters in header line") from err
            column_headers = [x.strip() for x in header_string.split(header_delim)]
        else:
            column_headers = ["Column" + str(x) for x in range(np.shape(self.data)[1])]
            try:
                data_test = linecache.getline(self.filename, data_line + 1)
            except SyntaxError as err:
                raise StonerLoadError("linecache fell over - not a text file?") from err
            if data_delim is None:
                for data_delim in ["\t", ",", ";", " "]:
                    if data_delim in data_test:
                        break
                else:
                    raise StonerLoadError("No delimiters in data lines")
            elif data_delim not in data_test:
                linecache.clearcache()
                raise StonerLoadError("No delimiters in data lines")

        self.data = np.genfromtxt(self.filename, dtype="float", delimiter=data_delim, skip_header=data_line)
        self.column_headers = column_headers
        linecache.clearcache()
        self._kargs = kargs
        return self

    def save(self, filename=None, **kargs):
        """Override the save method to allow CSVFiles to be written out to disc (as a mininmalist output).

        Args:
            filename (string): Fielname to save as (using the same rules as for the load routines)

        Keyword Arguments:
            deliminator (string): Record deliniminator (defaults to a comma)

        Returns:
            A copy of itself.
        """
        delimiter = kargs.pop("deliminator", ",")
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")
        with open(filename, "w") as outfile:
            spamWriter = csv.writer(outfile, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            i = 0
            spamWriter.writerow(self.column_headers)
            while i < self.data.shape[0]:
                spamWriter.writerow(self.data[i, :])
                i += 1
        self.filename = filename
        return self


class JustNumbersFile(CSVFile):

    """A reader format for things which are just a block of numbers with no headers or metadata."""

    priority = 256  # Rather generic file format so make it a low priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.csv", "*.txt"]  # Recognised filename patterns

    _defaults = {"header_line": None, "data_line": 0, "header_delim": None, "data_delim": None}


class KermitPNGFile(DataFile):

    """Loads PNG files with additional metadata embedded in them and extracts as metadata."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # We're checking for a the specoific PNG signature
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.png"]  # Recognised filename patterns

    mime_type = "image/png"

    def _check_signature(self, filename):
        """Check that this is a PNG file and raie a StonerLoadError if not."""
        try:
            with io.open(filename, "rb") as test:
                sig = test.read(8)
            sig = [x for x in sig]
            if self.debug:
                print(sig)
            if sig != [137, 80, 78, 71, 13, 10, 26, 10]:
                raise StonerLoadError("Signature mismatrch")
        except (StonerLoadError, IOError) as err:
            from traceback import format_exc

            raise StonerLoadError(f"Not a PNG file!>\n{format_exc()}") from err
        return True

    def _load(self, filename=None, *args, **kargs):
        """PNG file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        self._check_signature(filename)
        try:
            with PIL.Image.open(self.filename, "r") as img:
                for k in img.info:
                    self.metadata[k] = img.info[k]
                self.data = np.asarray(img)
        except IOError as err:
            raise StonerLoadError("Unable to read as a PNG file.") from err

        return self

    def save(self, filename=None, **kargs):
        """Override the save method to allow KermitPNGFiles to be written out to disc.

        Args:
            filename (string): Filename to save as (using the same rules as for the load routines)

        Keyword Arguments:
            deliminator (string): Record deliniminator (defaults to a comma)

        Returns:
            A copy of itself.
        """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")

        metadata = PIL.PngImagePlugin.PngInfo()
        for k in self.metadata:
            parts = self.metadata.export(k).split("=")
            key = parts[0]
            val = str2bytes("=".join(parts[1:]))
            metadata.add_text(key, val)
        img = PIL.Image.fromarray(self.data)
        img.save(filename, "png", pnginfo=metadata)
        self.filename = filename
        return self


try:  # Optional tdms support
    from nptdms import TdmsFile

    class TDMSFile(DataFile):

        """First stab at writing a file that will import TDMS files."""

        #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
        #   .. note::
        #      Subclasses with priority<=32 should make some positive identification that they have the right
        #      file type before attempting to read data.
        priority = 16  # Makes a positive ID of its file contents
        #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
        # the file load/save dialog boxes.
        patterns = ["*.tdms"]  # Recognised filename patterns

        mime_type = "application/octet-stream"

        def _load(self, filename=None, *args, **kargs):
            """TDMS file loader routine.

            Args:
                filename (string or bool): File to load. If None then the existing filename is used,
                    if False, then a file dialog will be used.

            Returns:
                A copy of the itself after loading the data.
            """
            if filename is None or not filename:
                self.get_filename("r")
            else:
                self.filename = filename
            # Open the file and read the main file header and unpack into a dict
            try:
                f = TdmsFile(self.filename)
                for grp in f.groups():
                    if grp.path == "/":
                        pass  # skip the rooot group
                    elif grp.path == "/'TDI Format 1.5'":
                        tmp = DataFile(grp.as_dataframe())
                        self.data = tmp.data
                        self.column_headers = tmp.column_headers
                        self.metadata.update(grp.properties)
                    else:
                        tmp = DataFile(grp.as_dataframe())
                        self.data = tmp.data
                        self.column_headers = tmp.column_headers
            except (IOError, ValueError, TypeError, StonerLoadError) as err:
                from traceback import format_exc

                raise StonerLoadError(f"Not a TDMS File \n{format_exc()}") from err

            return self


except ImportError:
    TDMSFile = DataFile

if Hyperspy_ok:

    class HyperSpyFile(DataFile):

        """Wrap the HyperSpy file to map to DataFile."""

        priority = 64  # Makes an ID check but is quite generic

        patterns = ["*.emd", "*.dm4"]

        mime_type = "application/x-hdf"  # Really an HDF5 file

        _axes_keys = ["name", "scale", "low_index", "low_value", "high_index", "high_value"]

        def _unpack_meta(self, root, value):
            """Recursively unpack a nested dict of metadata and append keys to self.metadata."""
            if isinstance(value, Mapping):
                for item in value.keys():
                    if root != "":
                        self._unpack_meta(f"{root}.{item}", value[item])
                    else:
                        self._unpack_meta(f"{item}", value[item])
            else:
                self.metadata[root] = value

        def _unpack_axes(self, ax_manager):
            """Unpack the axes managber as metadata."""
            for ax in ax_manager.signal_axes:
                for k in self._axes_keys:
                    self.metadata[f"{ax.name}.{k}"] = getattr(ax, k)

        def _load(self, filename=None, *args, **kargs):
            """Load HyperSpy file loader routine.

            Args:
                filename (string or bool): File to load. If None then the existing filename is used,
                    if False, then a file dialog will be used.

            Returns:
                A copy of the itself after loading the data.
            """
            if filename is None or not filename:
                self.get_filename("r")
            else:
                self.filename = filename
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
                signal = load(self.filename)
                if not isinstance(signal, hs.signals.Signal2D):
                    raise StonerLoadError("Not a 2D signal object - aborting!")
            except Exception as err:  # pylint: disable=W0703 Pretty generic error catcher
                raise StonerLoadError(f"Not readable by HyperSpy error was {err}") from err
            self.data = signal.data
            self._unpack_meta("", signal.metadata.as_dictionary())
            self._unpack_axes(signal.axes_manager)

            return self


else:
    HyperSpyFile = DataFile
