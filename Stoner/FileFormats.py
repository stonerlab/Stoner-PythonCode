"""Provides extra classes that can load data from various instruments into _SC_.DataFile type objects.

You do not need to use these classes directly, they are made available to :py:class:`Stoner.Core.Data` which
will load each of them in turn when asked to load an unknown data file.

Each class has a priority attribute that is used to determine the order in which
they are tried by :py:class:`Stoner.Core.Data` and friends where trying to load data.
High priority is run last (so is a bit of a misnomer!).

Eacg class should implement a load() method and optionally a save() method. Classes should make every effort to
positively identify that the file is one that they understand and throw a :py:exception:Stoner.Core._SC_.StonerLoadError` if not.
"""
from __future__ import print_function

__all__ = [
    "BNLFile",
    "BigBlueFile",
    "BirgeIVFile",
    "CSVFile",
    "EasyPlotFile",
    "FmokeFile",
    "GenXFile",
    "HyperSpyFile",
    "KermitPNGFile",
    "LSTemperatureFile",
    "MDAASCIIFile",
    "MokeFile",
    "OVFFile",
    "OpenGDAFile",
    "PIL",
    "PinkLibFile",
    "QDFile",
    "RasorFile",
    "RigakuFile",
    "SNSFile",
    "SPCFile",
    "TDMSFile",
    "VSMFile",
    "XRDFile",
]
# pylint: disable=unused-argument
import Stoner.Core as _SC_
from Stoner.formats.instruments import LSTemperatureFile, QDFile, RigakuFile, SPCFile, VSMFile, XRDFile
from Stoner.formats.facilities import BNLFile, MDAASCIIFile, OpenGDAFile, RasorFile, SNSFile
from Stoner.formats.generic import CSVFile, KermitPNGFile, TDMSFile, HyperSpyFile
from Stoner.formats.rigs import BigBlueFile, BirgeIVFile, MokeFile, FmokeFile
from .core.exceptions import assertion
from .core.base import string_to_type
import re
import numpy as _np_
import csv
import io

import PIL
import PIL.PngImagePlugin as png

# Expand png size limits as we have big text blocks full of metadata
png.MAX_TEXT_CHUNK = 2 ** 22
png.MAX_TEXT_MEMORY = 2 ** 28


class GenXFile(_SC_.DataFile):

    """Extends _SC_.DataFile for GenX Exported data."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 64
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
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as datafile:
            line = datafile.readline()
            match = pattern.match(line)
            match2 = pattern2.match(line)
            if match is not None:
                dataset = match.groups()[0]
                date = match.groups()[1]
                line = datafile.readline()
                line = datafile.readline()
                line = line[1:]
                self["date"] = date
            elif match2 is not None:
                line = datafile.readline()
                self["date"] = line.split(":")[1].strip()
                datafile.readline()
                line = datafile.readline()
                line = line[1:]
                dataset = "asymmetry"
            else:
                raise _SC_.StonerLoadError("Not a GenXFile")
        column_headers = [f.strip() for f in line.strip().split("\t")]
        self.data = _np_.genfromtxt(self.filename, skip_header=4)
        self["dataset"] = dataset
        self.setas = "xye"
        self.column_headers = column_headers
        return self


class OVFFile(_SC_.DataFile):

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

    def _read_uvwdata(self, filename, fmt, lineno):
        """Read the numerical data taking account of the format."""
        if fmt == "Text":
            uvwdata = _np_.genfromtxt(self.filename, skip_header=lineno + 2)
        elif fmt == "Binary 4":
            if self["version"] == 1:
                dt = _np_.dtype(">f4")
            else:
                dt = _np_.dtype("<f4")
            with io.open(filename, "rb") as bindata:
                bindata.seek(self._ptr)
                uvwdata = _np_.fromfile(
                    bindata, dtype=dt, count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"]
                )
                assertion(
                    uvwdata[0] == 1234567.0,
                    "Binary 4 format check value incorrect ! Actual Value was {}".format(uvwdata[0]),
                )
            uvwdata = uvwdata[1:]
            uvwdata = _np_.reshape(uvwdata, (-1, self["valuedim"]))
        elif fmt == "Binary 8":
            if self["version"] == 1:
                dt = _np_.dtype(">f8")
            else:
                dt = _np_.dtype("<f8")
            with io.open(filename, "rb") as bindata:
                bindata.seek(self._ptr)
                uvwdata = _np_.fromfile(
                    bindata, dtype=dt, count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"]
                )
                assertion(
                    (uvwdata[0] == 123456789012345.0),
                    "Binary 4 format check value incorrect ! Actual Value was {}".format(uvwdata[0]),
                )
            uvwdata = _np_.reshape(uvwdata, (-1, self["valuedim"]))
        else:
            raise _SC_.StonerLoadError("Unknow OVF Format {}".format(fmt))
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
                    raise _SC_.StonerLoadError("Cannot determine version of OOMMFF file")
            else:  # bug out oif we don't like the header
                raise _SC_.StonerLoadError("Not n OOMMF OVF File: opening line eas {}".format(line))
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
                        raise _SC_.StonerLoadError("Failed to understand metadata")
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

        x = (_np_.linspace(self["xmin"], self["xmax"], self["xnode"] + 1)[:-1] + self["xbase"]) * 1e9
        y = (_np_.linspace(self["ymin"], self["ymax"], self["ynode"] + 1)[:-1] + self["ybase"]) * 1e9
        z = (_np_.linspace(self["zmin"], self["zmax"], self["znode"] + 1)[:-1] + self["zbase"]) * 1e9
        (y, z, x) = (_np_.ravel(i) for i in _np_.meshgrid(y, z, x))
        self.data = _np_.column_stack((x, y, z, uvwdata))
        column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
        self.setas = "xyzuvw"
        self.column_headers = column_headers
        return self


class EasyPlotFile(_SC_.DataFile):

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
                raise _SC_.StonerLoadError("Not an EasyPlot Save file?")
            else:
                data.seek(0)
            for i, line in enumerate(data):
                line = line.strip()
                if line == "":
                    continue
                if line[0] not in "-0123456789" and datastart > 0 and dataend < 0:
                    dataend = i
                if line.startswith('"') and ":" in line:
                    parts = [x.strip() for x in line.strip('"').split(":")]
                    self[parts[0]] = string_to_type(":".join(parts[1:]))
                elif line.startswith("/"):  # command
                    parts = [x.strip('"') for x in next(csv.reader([line], delimiter=" ")) if x != ""]
                    cmd = parts[0].strip("/")
                    if len(cmd) > 1:
                        cmdname = "_{}_cmd".format(cmd)
                        if cmdname in dir(self):  # If this command is implemented as a function run it
                            cmd = getattr(self, "_{}_cmd".format(cmd))
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
        self.data = _np_.genfromtxt(self.filename, skip_header=datastart, skip_footer=i - dataend, delimiter=delimiter)
        if self.data.shape[1] == 2:
            self.setas = "xy"
        return self

    def _extend_columns(self, i):
        """Ensure the column headers are at least i long."""
        if len(self.column_headers) < i:
            l = len(self.column_headers)
            self.data = _np_.append(
                self.data, _np_.zeros((self.shape[0], i - l)), axis=1
            )  # Need to expand the array first
            self.column_headers.extend(["Column {}".format(x) for x in range(l, i)])

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
        """The sa (set-axis?) command."""
        if parts[0] == "l":  # Legend
            col = int(parts[2])
            self._extend_columns(col + 1)
            self.column_headers[col] = parts[1]


class PinkLibFile(_SC_.DataFile):

    """Extends _SC_.DataFile to load files from MdV's PINK library - as used by the GMR anneal rig."""

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
                raise _SC_.StonerLoadError("Not a PINK file")
            f = f.readlines()
            happened_before = False
            for i, line in enumerate(f):
                if line[0] != "#" and not happened_before:
                    header_line = i - 2  # -2 because there's a commented out data line
                    happened_before = True
                    continue  # want to get the metadata at the bottom of the file too
                elif any(s in line for s in ("Start time", "End time", "Title")):
                    tmp = line.strip("#").split(":")
                    self.metadata[tmp[0].strip()] = ":".join(tmp[1:]).strip()
            column_headers = f[header_line].strip("#\t ").split("\t")
        data = _np_.genfromtxt(self.filename, dtype="float", delimiter="\t", invalid_raise=False, comments="#")
        self.data = data[:, 0:-2]  # Deal with an errant tab at the end of each line
        self.column_headers = column_headers
        if _np_.all([h in column_headers for h in ("T (C)", "R (Ohm)")]):
            self.setas(x="T (C)", y="R (Ohm)")  # pylint: disable=not-callable
        return self
