#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement DataFile like classes for Various experimental rigs
"""
__all__ = ["BigBlueFile", "BirgeIVFile", "MokeFile", "FmokeFile"]
import io
import re

import numpy as np

import Stoner.Core as Core
from .generic import CSVFile
from Stoner.compat import bytes2str
from Stoner.core.base import string_to_type


class BigBlueFile(CSVFile):

    """Extends CSVFile to load files from Nick Porter's old BigBlue code"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 64  # Also rather generic file format so make a lower priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat", "*.iv", "*.rvt"]  # Recognised filename patterns

    def _load(self, filename, *args, **kargs):
        """Just call the parent class but with the right parameters set

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

        super(BigBlueFile, self)._load(
            self.filename, *args, header_line=3, data_line=7, data_delim=" ", header_delim=","
        )
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
            for l, m in zip(data[-4:], expected):
                if not l.startswith(m):
                    raise Core.StonerLoadError("Not a BirgeIVFile as wrong footer line")
                key = l[: len(m)]
                val = l[len(m) :]
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

    """Extends Core.DataFile to open Fmoke Files"""

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
            except Exception:
                f.close()
                raise Core.StonerLoadError("Not an FMOKE file?")
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
