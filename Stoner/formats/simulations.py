#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements classes to load file formats from various simulation packages."""

__all__ = ["GenXFile", "OVFFile"]
import re
import io

import numpy as np

from Stoner.Core import DataFile
from Stoner.core.exceptions import StonerLoadError, assertion
from Stoner.core.base import string_to_type


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
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as datafile:
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

    def __init__(self, *args, **kargs):
        """Set some instance attributes."""
        super().__init__(*args, **kargs)
        self._ptr = None

    def _read_uvwdata(self, filename, fmt, lineno):
        """Read the numerical data taking account of the format."""
        if fmt == "Text":
            uvwdata = np.genfromtxt(self.filename, skip_header=lineno + 2)
        elif fmt == "Binary 4":
            if self["version"] == 1:
                dt = np.dtype(">f4")
            else:
                dt = np.dtype("<f4")
            with io.open(filename, "rb") as bindata:
                bindata.seek(self._ptr)
                uvwdata = np.fromfile(
                    bindata, dtype=dt, count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"]
                )
                assertion(
                    uvwdata[0] == 1234567.0, f"Binary 4 format check value incorrect ! Actual Value was {uvwdata[0]}"
                )
            uvwdata = uvwdata[1:]
            uvwdata = np.reshape(uvwdata, (-1, self["valuedim"]))
        elif fmt == "Binary 8":
            if self["version"] == 1:
                dt = np.dtype(">f8")
            else:
                dt = np.dtype("<f8")
            with io.open(filename, "rb") as bindata:
                bindata.seek(self._ptr)
                uvwdata = np.fromfile(
                    bindata, dtype=dt, count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"]
                )
                assertion(
                    (uvwdata[0] == 123456789012345.0),
                    f"Binary 4 format check value incorrect ! Actual Value was {uvwdata[0]}",
                )
            uvwdata = np.reshape(uvwdata, (-1, self["valuedim"]))
        else:
            raise StonerLoadError(f"Unknow OVF Format {fmt}")
        return uvwdata

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename

        self._ptr = 0
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
            line = next(data)
            self._ptr += len(line)
            line = line.strip()
            if "OOMMF: rectangular mesh" in line:
                if "v1.0" in line:
                    self["version"] = 1
                elif "v2.0" in line:
                    self["version"] = 2
                else:
                    raise StonerLoadError("Cannot determine version of OOMMFF file")
            else:  # bug out oif we don't like the header
                raise StonerLoadError(f"Not n OOMMF OVF File: opening line eas {line}")
            pattern = re.compile(r"#\s*([^\:]+)\:\s+(.*)$")
            i = None
            for i, line in enumerate(data):
                self._ptr += len(line)
                line.strip()
                if line.startswith("# Begin: Data"):  # marks the start of the trext
                    break
                elif line.startswith("# Begin:") or line.startswith("# End:"):
                    continue
                else:
                    res = pattern.match(line)
                    if res is not None:
                        key = res.group(1)
                        val = res.group(2)
                        self[key] = string_to_type(val)
                    else:
                        raise StonerLoadError("Failed to understand metadata")
            fmt = re.match(r".*Data\s+(.*)", line).group(1).strip()
            assertion(
                (self["meshtype"] == "rectangular"),
                "Sorry only OVF files with rectnagular meshes are currently supported.",
            )
            if self["version"] == 1:
                if self["meshtype"] == "rectangular":
                    self["valuedim"] = 3
                else:
                    self["valuedim"] = 6
            uvwdata = self._read_uvwdata(filename, fmt, i)

        x = (np.linspace(self["xmin"], self["xmax"], self["xnode"] + 1)[:-1] + self["xbase"]) * 1e9
        y = (np.linspace(self["ymin"], self["ymax"], self["ynode"] + 1)[:-1] + self["ybase"]) * 1e9
        z = (np.linspace(self["zmin"], self["zmax"], self["znode"] + 1)[:-1] + self["zbase"]) * 1e9
        (y, z, x) = (np.ravel(i) for i in np.meshgrid(y, z, x))
        self.data = np.column_stack((x, y, z, uvwdata))
        column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
        self.setas = "xyzuvw"
        self.column_headers = column_headers
        return self
