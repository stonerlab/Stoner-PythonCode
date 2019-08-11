#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement DataFile like classes to represent Instrument Manufacturer's File Formats'
"""
__all__ = ["LSTemperatureFile", "QDFile", "RigakuFile", "SPCFile", "XRDFile"]

# Standard Library imports
from datetime import datetime
import io
import os
import re
import struct

import numpy as np

import Stoner.Core as Core
from Stoner.compat import python_v3, str2bytes, bytes2str
from Stoner.core.exceptions import StonerAssertionError, assertion
from Stoner.core.base import string_to_type


class LSTemperatureFile(Core.DataFile):

    """A class that reads and writes Lakeshore Temperature Calibration Curves.

    .. warning::

        This class works for cernox curves in Log Ohms/Kelvin and Log Ohms/Log Kelvin. It may or may not work with any
        other temperature calibration data !

    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.340"]

    def _load(self, filename=None, *args, **kargs):
        """Data loader function for 340 files."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename

        with io.open(self.filename, "rb") as data:
            keys = []
            vals = []
            for line in data:
                line = bytes2str(line)
                if line.strip() == "":
                    break
                parts = [p.strip() for p in line.split(":")]
                if len(parts) != 2:
                    raise Core.StonerLoadError("Header doesn't contain two parts at {}".format(line.strip()))
                else:
                    keys.append(parts[0])
                    vals.append(parts[1])
            else:
                raise Core.StonerLoadError("Overan the end of the file")
            if keys != [
                "Sensor Model",
                "Serial Number",
                "Data Format",
                "SetPoint Limit",
                "Temperature coefficient",
                "Number of Breakpoints",
            ]:
                raise Core.StonerLoadError("Header did not contain recognised keys.")
            for (k, v) in zip(keys, vals):
                v = v.split()[0]
                self.metadata[k] = string_to_type(v)
            headers = bytes2str(next(data)).strip().split()
            column_headers = headers[1:]
            dat = np.genfromtxt(data)
            self.data = dat[:, 1:]
        self.column_headers = column_headers
        return self

    def save(self, filename=None, **kargs):
        """Overrides the save method to allow CSVFiles to be written out to disc (as a mininmalist output)

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
        if self.shape[1] == 2:  # 2 columns, let's hope they're the right way round!
            cols = [0, 1]
        elif (
            self.setas.has_xcol and self.setas.has_ycol
        ):  # Use ycol, x col but assume x is real temperature and y is resistance
            cols = [self.setas.ycol[0], self.setas.xcol]
        else:
            cols = range(self.shape[1])
        with io.open(filename, "w", errors="ignore", encoding="utf-8", newline="\r\n") as f:
            for k, v in (
                ("Sensor Model", "CX-1070-SD"),
                ("Serial Number", "Unknown"),
                ("Data Format", 4),
                ("SetPoint Limit", 300.0),
                ("Temperature coefficient", 1),
                ("Number of Breakpoints", len(self)),
            ):
                if k in ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit"]:
                    kstr = "{:16s}".format(k + ":")
                else:
                    kstr = "{}:   ".format(k)
                v = self.get(k, v)
                if k == "Data Format":
                    units = ["()", "()", "()", "()", "(Log Ohms/Kelvin)", "(Log Ohms/Log Kelvin)"]
                    vstr = "{}      {}".format(v, units[int(v)])
                elif k == "SetPointLimit":
                    vstr = "{}      (Kelvin)".format(v)
                elif k == "Temperature coefficient":
                    vstr = "{} {}".format(v, ["(positive)", "(negative)"][v])
                elif k == "Number of Breakpoints":
                    vstr = str(len(self))
                else:
                    vstr = str(v)
                f.write(u"{}{}\n".format(kstr, vstr))
            f.write(u"\n")
            f.write(u"No.   ")
            for i in cols:
                f.write(u"{:11s}".format(self.column_headers[i]))
            f.write(u"\n\n")
            for i in range(
                len(self.data)
            ):  # This is a slow way to write the data, but there should only ever be 200 lines
                line = "\t".join(["{:<10.8f}".format(n) for n in self.data[i, cols]])
                f.write(u"{}\t".format(i))
                f.write(u"{}\n".format(line))
        self.filename = filename
        return self


class QDFile(Core.DataFile):

    """Extends Core.DataFile to load files from Quantum Design Systems - including PPMS, MPMS and SQUID-VSM"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 15  # Is able to make a positive ID of its file content, so get priority to check
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """QD system file loader routine.

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
        setas = {}
        i = 0
        with io.open(self.filename, "r", encoding="utf-8", errors="ignore") as f:  # Read filename linewise
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line != "[Header]":
                    raise Core.StonerLoadError("Not a Quantum Design File !")
                elif line == "[Header]" or line.startswith(";") or line == "":
                    continue
                elif "[Data]" in line:
                    break
                elif "," not in line:
                    raise Core.StonerLoadError("No data in file!")
                parts = [x.strip() for x in line.split(",")]
                if parts[1].split(":")[0] == "SEQUENCE FILE":
                    key = parts[1].split(":")[0].title()
                    value = parts[1].split(":")[1]
                elif parts[0] == "INFO":
                    if parts[1] == "APPNAME":
                        parts[1], parts[2] = parts[2], parts[1]
                    if len(parts) > 2:
                        key = "{}.{}".format(parts[0], parts[2])
                    else:
                        raise Core.StonerLoadError("No data in file!")
                    key = key.title()
                    value = parts[1]
                elif parts[0] in ["BYAPP", "FILEOPENTIME"]:
                    key = parts[0].title()
                    value = " ".join(parts[1:])
                elif parts[0] == "FIELDGROUP":
                    key = "{}.{}".format(parts[0], parts[1]).title()
                    value = "[{}]".format(",".join(parts[2:]))
                elif parts[0] == "STARTUPAXIS":
                    axis = parts[1][0].lower()
                    setas[axis] = setas.get(axis, []) + [int(parts[2])]
                    key = "Startupaxis-{}".format(parts[1].strip())
                    value = parts[2].strip()
                else:
                    key = parts[0] + "," + parts[1]
                    key = key.title()
                    value = " ".join(parts[2:])
                self.metadata[key] = string_to_type(value)
            else:
                raise Core.StonerLoadError("No data in file!")
            if "Byapp" not in self:
                raise Core.StonerLoadError("Not a Quantum Design File !")

            if python_v3:
                column_headers = f.readline().strip().split(",")
                if "," not in f.readline():
                    raise Core.StonerLoadError("No data in file!")
            else:
                column_headers = f.next().strip().split(",")
                if "," not in f.next():
                    raise Core.StonerLoadError("No data in file!")
            data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", delimiter=",", invalid_raise=False)
            if data.shape[1] != len(column_headers):  # Trap for buggy QD software not giving ewnough columns of data
                data = np.append(data, np.ones((data.shape[0], len(column_headers) - data.shape[1])) * np.NaN, axis=1)
            self.data = data
        self.column_headers = column_headers
        s = self.setas
        for k in setas:
            for ix in setas[k]:
                s[ix - 1] = k
        self.setas = s
        return self


class RigakuFile(Core.DataFile):

    """Loads a .ras file as produced by Rigaku X-ray diffractormeters"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Can make a positive id of file from first line
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.ras"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Reads an Rigaku ras file including handling the metadata nicely

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        from ast import literal_eval

        if filename is None or not filename:
            self.get_filename("rb")
        else:
            self.filename = filename
        sh = re.compile(r"^\*([^\s]+)\s+(.*)$")  # Regexp to grab the keys
        ka = re.compile(r"(.*)\-(\d+)$")
        header = dict()
        i = 0
        with io.open(self.filename, "rb") as f:
            for i, line in enumerate(f):
                line = bytes2str(line).strip()
                if i == 0 and line != "*RAS_DATA_START":
                    raise Core.StonerLoadError("Not a Rigaku file!")
                if line == "*RAS_HEADER_START":
                    break
            i2 = None
            for i2, line in enumerate(f):
                line = bytes2str(line).strip()
                m = sh.match(line)
                if m:
                    key = m.groups()[0].lower().replace("_", ".")
                    try:
                        value = m.groups()[1].decode("utf-8", "ignore")
                    except AttributeError:
                        value = m.groups()[1]
                    header[key] = value
                if "*RAS_INT_START" in line:
                    break
            keys = list(header.keys())
            keys.sort()
            for key in keys:
                m = ka.match(key)
                value = header[key].strip()
                try:
                    newvalue = literal_eval(value.strip('"'))
                except Exception:
                    newvalue = literal_eval(value)
                if m:
                    key = m.groups()[0]
                    if key in self.metadata and not (isinstance(self[key], (np.ndarray, list))):
                        if isinstance(self[key], str):
                            self[key] = list([self[key]])
                        else:
                            self[key] = np.array(self[key])
                    if key not in self.metadata:
                        if isinstance(newvalue, str):
                            self[key] = list([newvalue])
                        else:
                            self[key] = np.array([newvalue])
                    else:
                        if isinstance(self[key][0], str) and isinstance(self[key], list):
                            self[key].append(newvalue)
                        else:
                            self[key] = np.append(self[key], newvalue)
                else:
                    self.metadata[key] = newvalue

        with io.open(self.filename, "rb") as data:
            self.data = np.genfromtxt(
                data, dtype="float", delimiter=" ", invalid_raise=False, comments="*", skip_header=i + i2 + 1
            )
        column_headers = ["Column" + str(i) for i in range(self.data.shape[1])]
        column_headers[0:2] = [self.metadata["meas.scan.unit.x"], self.metadata["meas.scan.unit.y"]]
        for key in self.metadata:
            if isinstance(self[key], list):
                self[key] = np.array(self[key])
        self.setas = "xy"
        self.column_headers = column_headers
        return self

    def to_Q(self, l=1.540593):
        """Adds an additional function to covert an angualr scale to momentum transfer.

        Returns:
            a copy of itself.
        """
        self.add_column((4 * np.pi / l) * np.sin(np.pi * self.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)")


class SPCFile(Core.DataFile):

    """Extends Core.DataFile to load SPC files from Raman"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Can't make a positive ID of itself
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.spc"]  # Recognised filename patterns

    mime_type = ["application/octet-stream"]

    def _read_xdata(self, f):
        """Read the xdata from the spc file."""
        self._pts = self._header["fnpts"]
        if self._header["ftflgs"] & 128:  # We need to read some X Data
            if 4 * self._pts > self._filesize - f.tell():
                raise Core.StonerLoadError("Trying to read too much data!")
            xvals = f.read(4 * self._pts)  # I think storing X vals directly implies that each one is 4 bytes....
            xdata = np.array(struct.unpack(str2bytes(str(self._pts) + "f"), xvals))
        else:  # Generate the X Data ourselves
            first = self._header["ffirst"]
            last = self._header["flast"]
            if self._pts > 1e6:  # Something not right here !
                raise Core.StonerLoadError("More than 1 million points requested. Bugging out now!")
            xdata = np.linspace(first, last, self._pts)
        return xdata

    def _read_ydata(self, f, data, column_headers):
        """Read the y data and column headers from spc file."""
        n = self._header["fnsub"]
        subhdr_keys = (
            "subflgs",
            "subexp",
            "subindx",
            "subtime",
            "subnext",
            "subnois",
            "subnpts",
            "subscan",
            "subwlevel",
            "subresv",
        )
        if self._header["ftflgs"] & 1:
            y_width = 2
            y_fmt = "h"
            divisor = 2 ** 16
        else:
            y_width = 4
            y_fmt = "i"
            divisor = 2 ** 32
        if n * (y_width * self._pts + 32) > self._filesize - f.tell():
            raise Core.StonerLoadError("No good, going to read too much data!")
        for j in range(n):  # We have n sub-scans
            # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
            subhdr = struct.unpack(b"BBHfffIIf4s", f.read(32))
            subheader = dict(zip(["scan" + str(j) + ":" + x for x in subhdr_keys], subhdr))

            # Now read the y-data
            exponent = subheader["scan" + str(j) + ":subexp"]
            if int(exponent) & -128:  # Data is unscaled direct floats
                ydata = np.array(struct.unpack(str2bytes(str(self._pts) + "f"), f.read(self._pts * y_width)))
            else:  # Data is scaled by exponent
                yvals = struct.unpack(str2bytes(str(self._pts) + y_fmt), f.read(self._pts * y_width))
                ydata = np.array(yvals, dtype="float64") * (2 ** exponent) / divisor
            data[:, j + 1] = ydata
            self._header = dict(self._header, **subheader)
            column_headers.append("Scan" + str(j) + ":" + self._yvars[self._header["fytype"]])

        return data

    def _read_loginfo(self, f):
        """Read the log info section of the spc file."""
        logstc = struct.unpack(b"IIIII44s", f.read(64))
        logstc_keys = ("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
        logheader = dict(zip(logstc_keys, logstc))
        self._header = dict(self._header, **logheader)

        # Can't handle either binary log information or ion disk log information (wtf is this anyway !)
        if self._header["logbins"] + self._header["logdsks"] > self._filesize - f.tell():
            raise Core.StonerLoadError("Too much logfile data to read")
        f.read(self._header["logbins"] + self._header["logdsks"])

        # The renishaw seems to put a 16 character timestamp next - it's not in the spec but never mind that.
        self._header["Date-Time"] = f.read(16)
        # Now read the rest of the file as log text
        logtext = f.read()
        # We expect things to be single lines terminated with a CR-LF of the format key=value
        for line in re.split(b"[\r\n]+", logtext):
            if b"=" in line:
                parts = line.split(b"=")
                key = parts[0].decode()
                value = parts[1].decode()
                self._header[key] = value

    def _load(self, filename=None, *args, **kargs):
        """Reads a .scf file produced by the Renishaw Raman system (amongs others)

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.

        Todo:
            Implement the second form of the file that stores multiple x-y curves in the one file.

        Notes:
            Metadata keys are pretty much as specified in the spc.h file that defines the filerformat.
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        self._filesize = os.stat(self.filename).st_size
        with io.open(filename, "rb") as f:
            spchdr = struct.unpack(b"BBBciddiBBBBi9s9sH8f30s130siiBBHf48sfifB187s", f.read(512))
            keys = (
                "ftflgs",
                "fversn",
                "fexper",
                "fexp",
                "fnpts",
                "ffirst",
                "flast",
                "fnsub",
                "fxtype",
                "fytype",
                "fztype",
                "fpost",
                "fres",
                "fsource",
                "fpeakpt",
                "fspare1",
                "fspare2",
                "fspare3",
                "fspare4",
                "fspare5",
                "fspare6",
                "fspare7",
                "fspare8",
                "fcm",
                "nt",
                "fcatx",
                "flogoff",
                "fmods",
                "fprocs",
                "flevel",
                "fsampin",
                "ffactor",
                "fmethod",
                "fzinc",
                "fwplanes",
                "fwinc",
                "fwtype",
                "fwtype",
                "fresv",
            )
            self._xvars = [
                "Arbitrary",
                "Wavenumber (cm-1)",
                "Micrometers (um)",
                "Nanometers (nm)",
                "Seconds",
                "Minutes",
                "Hertz (Hz)",
                "Kilohertz (KHz)",
                "Megahertz (MHz)",
                "Mass (M/z)",
                "Parts per million (PPM)",
                "Days",
                "Years",
                "Raman Shift (cm-1)",
                "Raman Shift (cm-1)",
                "eV",
                "XYZ text labels in fcatxt (old 0x4D version only)",
                "Diode Number",
                "Channel",
                "Degrees",
                "Temperature (F)",
                "Temperature (C)",
                "Temperature (K)",
                "Data Points",
                "Milliseconds (mSec)",
                "Microseconds (uSec)",
                "Nanoseconds (nSec)",
                "Gigahertz (GHz)",
                "Centimeters (cm)",
                "Meters (m)",
                "Millimeters (mm)",
                "Hours",
                "Hours",
            ]
            self._yvars = [
                "Arbitrary Intensity",
                "Interferogram",
                "Absorbance",
                "Kubelka-Monk",
                "Counts",
                "Volts",
                "Degrees",
                "Milliamps",
                "Millimeters",
                "Millivolts",
                "Log(1/R)",
                "Percent",
                "Percent",
                "Intensity",
                "Relative Intensity",
                "Energy",
                "Decibel",
                "Temperature (F)",
                "Temperature (C)",
                "Temperature (K)",
                "Index of Refraction [N]",
                "Extinction Coeff. [K]",
                "Real",
                "Imaginary",
                "Complex",
                "Complex",
                "Transmission (ALL HIGHER MUST HAVE VALLEYS!)",
                "Reflectance",
                "Arbitrary or Single Beam with Valley Peaks",
                "Emission",
                "Emission",
            ]

            self._header = dict(zip(keys, spchdr))
            n = self._header["fnsub"]

            if self._header["ftflgs"] & 64 == 64 or not (
                75 <= self._header["fversn"] <= 77
            ):  # This is the multiple XY curves in file flag.
                raise Core.StonerLoadError(
                    "Filetype not implemented yet ! ftflgs={ftflgs}, fversn={fversn}".format(**self._header)
                )
            else:  # A single XY curve in the file.
                # Read the xdata and add it to the file.
                xdata = self._read_xdata(f)
                data = np.zeros((self._pts, (n + 1)))  # initialise the data soace
                data[:, 0] = xdata  # Put in the X-Data
                column_headers = [self._xvars[self._header["fxtype"]]]  # And label the X column correctly

                # Now we're going to read the Y-data
                data = self._read_ydata(f, data, column_headers)
                if self._header["flogoff"] != 0:  # Ok, we've got a log, so read the log header and merge into metadata
                    self._read_loginfo(f)
            # Ok now build the Stoner.Core.DataFile instance to return
            self.data = data
            # The next bit generates the metadata. We don't just copy the metadata because we need to figure out the typehints first - hence the loop
            # here to call Core.DataFile.__setitem()
            for x in self._header:
                self[x] = self._header[x]
            self.column_headers = column_headers
            if len(self.column_headers) == 2:
                self.setas = "xy"
            return self


class VSMFile(Core.DataFile):

    """Extends Core.DataFile to open VSM Files"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Now makes a positive ID of its contents
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.fld"]  # Recognised filename patterns

    def __parse_VSM(self, header_line=3, data_line=3, header_delim=","):
        """An intrernal function for parsing deliminated data without a leading column of metadata.copy

        Keyword Arguments:
            header_line (int): The line in the file that contains the column headers.
                If None, then column headers are auotmatically generated.
            data_line (int): The line on which the data starts
            header_delim (strong): The delimiter used for separating header values

        Returns:
            Nothing, but modifies the current object.

        Note:
            The default values are configured fir read VSM data files
        """
        try:
            with io.open(self.filename, errors="ignore", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        self["Timestamp"] = line.strip()
                        check = datetime.strptime(self["Timestamp"], "%a %b %d %H:%M:%S %Y")
                        if check is None:
                            raise Core.StonerLoadError("Not a VSM file ?")
                    elif i == 1:
                        assertion(line.strip() == "")
                    elif i == 2:
                        header_string = line.strip()
                    elif i == header_line:
                        unit_string = line.strip()
                        column_headers = [
                            "{} ({})".format(h.strip(), u.strip())
                            for h, u in zip(header_string.split(header_delim), unit_string.split(header_delim))
                        ]
                    elif i > 3:
                        break
        except (StonerAssertionError, ValueError, AssertionError, TypeError) as e:
            raise Core.StonerLoadError("Not a VSM File" + str(e.args))
        self.data = np.genfromtxt(
            self.filename,
            dtype="float",
            usemask=True,
            skip_header=data_line - 1,
            missing_values=["6:0", "---"],
            invalid_raise=False,
        )

        self.data = np.ma.mask_rows(self.data)
        cols = self.data.shape[1]
        self.data = np.reshape(self.data.compressed(), (-1, cols))
        self.column_headers = column_headers
        self.setas(x="H_vsm (T)", y="m (emu)")  # pylint: disable=not-callable

    def _load(self, filename=None, *args, **kargs):
        """VSM file loader routine.

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
        self.__parse_VSM()
        return self


class XRDFile(Core.DataFile):

    """Loads Files from a Brucker D8 Discovery X-Ray Diffractometer"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Makes a positive id of its file contents
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dql"]  # Recognised filename patterns

    def __init__(self, *args, **kargs):
        """Add a public attribute to XRD File."""
        super(XRDFile, self).__init__(*args, **kargs)
        self._public_attrs = {"four_bounce": bool}

    def _load(self, filename=None, *args, **kargs):
        """Reads an XRD Core.DataFile as produced by the Brucker diffractometer

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.

        Notes:
            Format is ini file like but not enough to do standard inifile processing - in particular
            one can have multiple sections with the same name (!)
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        sh = re.compile(r"\[(.+)\]")  # Regexp to grab section name
        with io.open(self.filename, errors="ignore", encoding="utf-8") as f:  # Read filename linewise
            if f.readline().strip() != ";RAW4.00":  # Check we have the corrrect fileformat
                raise Core.StonerLoadError("File Format Not Recognized !")
            drive = 0
            for line in f:  # for each line
                m = sh.search(line)
                if m:  # This is a new section
                    section = m.group(1)
                    if section == "Drive":  # If this is a Drive section we need to know which Drive Section it is
                        section = section + str(drive)
                        drive = drive + 1
                    elif section == "Data":  # Data section contains the business but has a redundant first line
                        if python_v3:
                            f.readline()
                        else:
                            f.next()
                    for line in f:  # Now start reading lines in this section...
                        if (
                            line.strip() == ""
                        ):  # A blank line marks the end of the section, so go back to the outer loop which will handle a new section
                            break
                        elif section == "Data":  # In the Data section read lines of data value,vale
                            parts = line.split(",")
                            angle = parts[0].strip()
                            counts = parts[1].strip()
                            dataline = np.array([float(angle), float(counts)])
                            self.data = np.append(self.data, dataline)
                        else:  # Other sections contain metadata
                            parts = line.split("=")
                            key = parts[0].strip()
                            data = parts[1].strip()
                            # Keynames in main metadata are section:key - use theCore.DataFile magic to do type determination
                            self[section + ":" + key] = string_to_type(data)
            column_headers = ["Angle", "Counts"]  # Assume the columns were Angles and Counts

        self.data = np.reshape(self.data, (-1, 2))
        self.setas = "xy"
        self.four_bounce = self["HardwareConfiguration:Monochromator"] == 1
        self.column_headers = column_headers
        return self

    def to_Q(self, l=1.540593):
        """Adds an additional function to covert an angualr scale to momentum transfer

        returns a copy of itself.
        """
        self.add_column((4 * np.pi / l) * np.sin(np.pi * self.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)")
