#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile like classes for Various experimental rigs."""

__all__ = ["BigBlueFile", "BirgeIVFile", "MokeFile", "FmokeFile", "EasyPlotFile", "PinkLibFile"]
import io
import re
import csv

import numpy as np

from Stoner.compat import bytes2str
from Stoner.core.base import string_to_type
import Stoner.Core as Core
from .generic import CSVFile


class BigBlueFile(CSVFile):

    """Extends CSVFile to load files from Nick Porter's old BigBlue code."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 64  # Also rather generic file format so make a lower priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat", "*.iv", "*.rvt"]  # Recognised filename patterns

    def _load(self, filename, *args, **kargs):
        """Just call the parent class but with the right parameters set.

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

        super()._load(self.filename, *args, header_line=3, data_line=7, data_delim=" ", header_delim=",")
        if np.all(np.isnan(self.data)):
            raise Core.StonerLoadError("All data was NaN in Big Blue format")
        return self


class BirgeIVFile(Core.DataFile):

    """Implements the IV File format used by the Birge Group in Michigan State University Condesned Matter Physiscs."""

    patterns = ["*.dat"]

    def _load(self, filename, *args, **kargs):
        """File loader for PinkLib.

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
        ix = 0
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
            if not re.compile(r"\d{1,2}/\d{1,2}/\d{4}").match(f.readline()):
                raise Core.StonerLoadError("Not a BirgeIVFile as no date on first line")
            data = f.readlines()
            expected = ["Vo(-))", "Vo(+))", "Ic(+)", "Ic(-)"]
            for length, m in zip(data[-4:], expected):
                if not length.startswith(m):
                    raise Core.StonerLoadError("Not a BirgeIVFile as wrong footer line")
                key = length[: len(m)]
                val = length[len(m) :]
                if "STDEV" in val:
                    ix2 = val.index("STDEV")
                    key2 = val[ix2 : ix2 + 4 + len(key)]
                    val2 = val[ix2 + 4 + len(key) :]
                    self.metadata[key2] = string_to_type(val2.strip())
                    val = val[:ix2]
                self.metadata[key] = string_to_type(val.strip())
            for ix, line in enumerate(data):  # Scan the ough lines to get metadata
                if ":" in line:
                    parts = line.split(":")
                    self.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
                elif "," in line:
                    for part in line.split(","):
                        parts = part.split(" ")
                        self.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
                elif line.startswith("H "):
                    self.metadata["H"] = string_to_type(line.split(" ")[1].strip())
                else:
                    headers = [x.strip() for x in line.split(" ")]
                    break
            else:
                raise Core.StonerLoadError("Oops ran off the end of the file!")
        self.data = np.genfromtxt(filename, skip_header=ix + 2, skip_footer=4)
        self.column_headers = headers

        self.setas = "xy"
        return self


class MokeFile(Core.DataFile):

    """Class that extgends Core.DataFile to load files from the Leeds MOKE system."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priotity = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat", "*.txt"]

    def _load(self, filename, *args, **kargs):
        """Leeds  MOKE file loader routine.

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
        with io.open(self.filename, mode="rb") as f:
            line = bytes2str(f.readline()).strip()
            if line != "#Leeds CM Physics MOKE":
                raise Core.StonerLoadError("Not a Core.DataFile from the Leeds MOKE")
            while line.startswith("#") or line == "":
                parts = line.split(":")
                if len(parts) > 1:
                    key = parts[0][1:]
                    data = ":".join(parts[1:]).strip()
                    self[key] = data
                line = bytes2str(f.readline()).strip()
            column_headers = [x.strip() for x in line.split(",")]
            self.data = np.genfromtxt(f, delimiter=",")
        self.setas = "xy.de"
        self.column_headers = column_headers
        return self


class FmokeFile(Core.DataFile):

    """Extends Core.DataFile to open Fmoke Files."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Makes a positive ID check of its contents so give it priority in autoloading
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename, *args, **kargs):
        """Sheffield Focussed MOKE file loader routine.

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
        with io.open(self.filename, mode="rb") as f:
            try:
                value = [float(x.strip()) for x in bytes2str(f.readline()).split("\t")]
            except (TypeError, ValueError) as err:
                f.close()
                raise Core.StonerLoadError("Not an FMOKE file?") from err
            label = [x.strip() for x in bytes2str(f.readline()).split("\t")]
            if label[0] != "Header:":
                f.close()
                raise Core.StonerLoadError("Not a Focussed MOKE file !")
            del label[0]
            for k, v in zip(label, value):
                self.metadata[k] = v  # Create metatdata from first 2 lines
            column_headers = [x.strip() for x in bytes2str(f.readline()).split("\t")]
            self.data = np.genfromtxt(f, dtype="float", delimiter="\t", invalid_raise=False)
            self.column_headers = column_headers
        return self


class EasyPlotFile(Core.DataFile):

    """A class that will extract as much as it can from an EasyPlot save File."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 32  # Fairly generic, but can do some explicit testing

    def _load(self, filename, *args, **kargs):
        """Private loader method."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename

        datastart = -1
        dataend = -1

        i = 0
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as data:
            if "******** EasyPlot save file ********" not in data.read(1024):
                raise Core.StonerLoadError("Not an EasyPlot Save file?")
            data.seek(0)
            for i, line in enumerate(data):
                line = line.strip()
                if line == "":
                    continue
                if line[0] not in "-0123456789" and dataend < 0 <= datastart:
                    dataend = i
                if line.startswith('"') and ":" in line:
                    parts = [x.strip() for x in line.strip('"').split(":")]
                    self[parts[0]] = string_to_type(":".join(parts[1:]))
                elif line.startswith("/"):  # command
                    parts = [x.strip('"') for x in next(csv.reader([line], delimiter=" ")) if x != ""]
                    cmd = parts[0].strip("/")
                    if len(cmd) > 1:
                        cmdname = f"_{cmd}_cmd"
                        if cmdname in dir(self):  # If this command is implemented as a function run it
                            cmd = getattr(self, f"_{cmd}_cmd")
                            cmd(parts[1:])
                        else:
                            if len(parts[1:]) > 1:
                                cmd = cmd + "." + parts[1]
                                value = ",".join(parts[2:])
                            elif len(parts[1:]) == 1:
                                value = parts[1]
                            else:
                                value = True
                            self[cmd] = value
                elif line[0] in "-0123456789" and datastart < 0:  # start of data
                    datastart = i
                    if "," in line:
                        delimiter = ","
                    else:
                        delimiter = None
        if dataend < 0:
            dataend = i
        self.data = np.genfromtxt(self.filename, skip_header=datastart, skip_footer=i - dataend, delimiter=delimiter)
        if self.data.shape[1] == 2:
            self.setas = "xy"
        return self

    def _extend_columns(self, i):
        """Ensure the column headers are at least i long."""
        if len(self.column_headers) < i:
            length = len(self.column_headers)
            self.data = np.append(
                self.data, np.zeros((self.shape[0], i - length)), axis=1
            )  # Need to expand the array first
            self.column_headers.extend([f"Column {x}" for x in range(length, i)])

    def _et_cmd(self, parts):
        """Handle axis labellling command."""
        if parts[0] == "x":
            self._extend_columns(1)
            self.column_headers[0] = parts[1]
        elif parts[0] == "y":
            self._extend_columns(2)
            self.column_headers[1] = parts[1]
        elif parts[0] == "g":
            self["title"] = parts[1]

    def _td_cmd(self, parts):
        self.setas = parts[0]

    def _sa_cmd(self, parts):
        """Implement the sa (set-axis?) command."""
        if parts[0] == "l":  # Legend
            col = int(parts[2])
            self._extend_columns(col + 1)
            self.column_headers[col] = parts[1]


class PinkLibFile(Core.DataFile):

    """Extends Core.DataFile to load files from MdV's PINK library - as used by the GMR anneal rig."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 32  # reasonably generic format
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """File loader for PinkLib.

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
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
            if "PINKlibrary" not in f.readline():
                raise Core.StonerLoadError("Not a PINK file")
            f = f.readlines()
            happened_before = False
            for i, line in enumerate(f):
                if line[0] != "#" and not happened_before:
                    header_line = i - 2  # -2 because there's a commented out data line
                    happened_before = True
                    continue  # want to get the metadata at the bottom of the file too
                if any(s in line for s in ("Start time", "End time", "Title")):
                    tmp = line.strip("#").split(":")
                    self.metadata[tmp[0].strip()] = ":".join(tmp[1:]).strip()
            column_headers = f[header_line].strip("#\t ").split("\t")
        data = np.genfromtxt(self.filename, dtype="float", delimiter="\t", invalid_raise=False, comments="#")
        self.data = data[:, 0:-2]  # Deal with an errant tab at the end of each line
        self.column_headers = column_headers
        if np.all([h in column_headers for h in ("T (C)", "R (Ohm)")]):
            self.setas(x="T (C)", y="R (Ohm)")  # pylint: disable=not-callable
        return self
