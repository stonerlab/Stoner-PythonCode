#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements classes to load file formats from various simulation packages."""

__all__ = ["GenXFile", "OVFFile"]
import re
import io

import numpy as np

from ..Core import DataFile
from ..core.exceptions import StonerLoadError, assertion
from ..core.base import string_to_type
from ..tools.file import FileManager


def _read_line(data, metadata):
    """Read a single line and add to a metadata dictionary."""
    line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
    if line == "" or ":" not in line:
        return True
    parts = line.split(":")
    field = parts[0].strip()
    value = ":".join(parts[1:]).strip()
    if field == "Begin" and value.startswith("Data "):
        value = value.split(" ")
        metadata["representation"] = value[1]
        if value[1] == "Binary":
            metadata["representation size"] = value[2]
        return False
    if field not in ["Begin", "End"]:
        metadata[field] = value
    return True


class GenXFile(DataFile):

    """Extends DataFile for GenX Exported data."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        pattern = re.compile(r'# Dataset "([^\"]*)" exported from GenX on (.*)$')
        pattern2 = re.compile(r"#\sFile\sexported\sfrom\sGenX\'s\sReflectivity\splugin")
        i = 0
        ix = 0
        with FileManager(self.filename, "r", errors="ignore", encoding="utf-8") as datafile:
            line = datafile.readline()
            match = pattern.match(line)
            match2 = pattern2.match(line)
            if match is not None:
                dataset = match.groups()[0]
                date = match.groups()[1]
                self["date"] = date
                i = 2
            elif match2 is not None:
                line = datafile.readline()
                self["date"] = line.split(":")[1].strip()
                dataset = datafile.readline()[1:].strip()
                i = 3
            else:
                raise StonerLoadError("Not a GenXFile")
            for ix, line in enumerate(datafile):
                line = line.strip()
                if line in ["# Headers:", "# Column lables:"]:
                    line = next(datafile)[1:].strip()
                    break
            else:
                raise StonerLoadError("Cannot find headers")
        skip = ix + i + 2
        column_headers = [f.strip() for f in line.strip().split("\t")]
        self.data = np.real(np.genfromtxt(self.filename, skip_header=skip, dtype=complex))
        self["dataset"] = dataset
        if "sld" in dataset.lower():
            self["type"] = "SLD"
        elif "asymmetry" in dataset.lower():
            self["type"] = "Asymmetry"
        elif "dd" in dataset.lower():
            self["type"] = "Down"
        elif "uu" in dataset.lower():
            self["type"] = "Up"
        self.column_headers = column_headers
        return self


class OVFFile(DataFile):

    """A class that reads OOMMF vector format files and constructs x,y,z,u,v,w data.

    OVF 1 and OVF 2 files with text or binary data and only files with a meshtype rectangular are supported
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.ovf"]  # Recognised filename patterns

    mime_type = ["text/plain", "application/octet-stream"]

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards.

        Notes:
            This code can handle only the first segment in the data file.
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        # Reading in binary, converting to utf-8 where we should be in the text header
        with FileManager(self.filename, "rb") as data:
            line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
            if line not in ["OOMMF: rectangular mesh v1.0", "OOMMF OVF 2.0"]:
                raise StonerLoadError("Not an OVF 1.0 or 2.0 file.")
            while _read_line(data, self.metadata):
                pass  # Read the file until we reach the start of the data block.
            if self.metadata["representation"] == "Binary":
                size = (  # Work out the size of the data block to read
                    self.metadata["xnodes"]
                    * self.metadata["ynodes"]
                    * self.metadata["znodes"]
                    * self.metadata["valuedim"]
                    + 1
                ) * self.metadata["representation size"]
                bin_data = data.read(size)
                numbers = np.frombuffer(bin_data, dtype=f"<f{self.metadata['representation size']}")
                chk = numbers[0]
                if (
                    chk != [1234567.0, 123456789012345.0][self.metadata["representation size"] // 4 - 1]
                ):  # If we have a good check number we can carry on, otherwise try the other endianess
                    numbers = np.frombuffer(bin_data, dtype=f">f{self.metadata['representation size']}")
                    chk = numbers[0]
                    if chk != [1234567.0, 123456789012345.0][self.metadata["representation size"] // 4 - 1]:
                        raise StonerLoadError("Bad binary data for ovf gile.")

                data = np.reshape(
                    numbers[1:], (self.metadata["xnodes"] * self.metadata["ynodes"] * self.metadata["znodes"], 3),
                )
            else:
                data = np.genfromtxt(
                    data, max_rows=self.metadata["xnodes"] * self.metadata["ynodes"] * self.metadata["znodes"],
                )
        xmin, xmax, xstep = (
            self.metadata["xmin"],
            self.metadata["xmax"],
            self.metadata["xstepsize"],
        )
        ymin, ymax, ystep = (
            self.metadata["ymin"],
            self.metadata["ymax"],
            self.metadata["ystepsize"],
        )
        zmin, zmax, zstep = (
            self.metadata["zmin"],
            self.metadata["zmax"],
            self.metadata["zstepsize"],
        )
        Z, Y, X = np.meshgrid(
            np.arange(zmin + zstep / 2, zmax, zstep) * 1e9,
            np.arange(ymin + ystep / 2, ymax, ystep) * 1e9,
            np.arange(xmin + xstep / 2, xmax, xstep) * 1e9,
            indexing="ij",
        )
        self.data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), data))

        column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
        self.setas = "xyzuvw"
        self.column_headers = column_headers
        return self
