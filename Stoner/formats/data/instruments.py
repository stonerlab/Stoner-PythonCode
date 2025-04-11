#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile like classes to represent Instrument Manufacturer's File Formats'."""

# Standard Library imports
from datetime import datetime
import re
import struct
from ast import literal_eval

import numpy as np

from ...compat import str2bytes, bytes2str
from ...core.exceptions import StonerAssertionError, assertion, StonerLoadError
from ...core.base import string_to_type
from ...tools.file import FileManager, SizedFileManager, get_filename

from ..decorators import register_loader, register_saver


@register_loader(patterns=(".340", 16), mime_types=("text/plain", 32), name="LSTemperatureFile", what="Data")
def load_340(new_data, *args, **kargs):
    """Load data for 340 files."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    with FileManager(new_data.filename, "rb") as data:
        keys = []
        vals = []
        for line in data:
            line = bytes2str(line)
            if line.strip() == "":
                break
            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 2:
                raise StonerLoadError(f"Header doesn't contain two parts at {line.strip()}")
            keys.append(parts[0])
            vals.append(parts[1])
        else:
            raise StonerLoadError("Overan the end of the file")
        if keys != [
            "Sensor Model",
            "Serial Number",
            "Data Format",
            "SetPoint Limit",
            "Temperature coefficient",
            "Number of Breakpoints",
        ]:
            raise StonerLoadError("Header did not contain recognised keys.")
        for k, v in zip(keys, vals):
            v = v.split()[0]
            new_data.metadata[k] = string_to_type(v)
        headers = bytes2str(next(data)).strip().split()
        column_headers = headers[1:]
        dat = np.genfromtxt(data)
        new_data.data = dat[:, 1:]
    new_data.column_headers = column_headers
    return new_data


@register_saver(patterns=".340", name="LSTemperatureFile", what="Data")
def save_340(save_data, filename=None, **kargs):
    """Override the save method to allow CSVFiles to be written out to disc (as a mininmalist output).

    Args:
        filename (string):
            Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string):
            Record deliniminator (defaults to a comma)

    Returns:
        A copy of itsave_data.
    """
    if filename is None:
        filename = save_data.filename
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = save_data.__file_dialog("w")
    if save_data.shape[1] == 2:  # 2 columns, let's hope they're the right way round!
        cols = [0, 1]
    elif (
        save_data.setas.has_xcol and save_data.setas.has_ycol
    ):  # Use ycol, x col but assume x is real temperature and y is resistance
        cols = [save_data.setas.ycol[0], save_data.setas.xcol]
    else:
        cols = range(save_data.shape[1])
    with FileManager(filename, "w", errors="ignore", encoding="utf-8", newline="\r\n") as f:
        for k, v in (
            ("Sensor Model", "CX-1070-SD"),
            ("Serial Number", "Unknown"),
            ("Data Format", 4),
            ("SetPoint Limit", 300.0),
            ("Temperature coefficient", 1),
            ("Number of Breakpoints", len(save_data)),
        ):
            if k in ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit"]:
                kstr = f"{k+':':16s}"
            else:
                kstr = f"{k}:   "
            v = save_data.get(k, v)
            if k == "Data Format":
                units = ["()", "()", "()", "()", "(Log Ohms/Kelvin)", "(Log Ohms/Log Kelvin)"]
                vstr = f"{v}      {units[int(v)]}"
            elif k == "SetPointLimit":
                vstr = f"{v}      (Kelvin)"
            elif k == "Temperature coefficient":
                vstr = f"{v} {['(positive)', '(negative)'][v]}"
            elif k == "Number of Breakpoints":
                vstr = str(len(save_data))
            else:
                vstr = str(v)
            f.write(f"{kstr}{vstr}\n")
        f.write("\n")
        f.write("No.   ")
        for i in cols:
            f.write(f"{save_data.column_headers[i]:11s}")
        f.write("\n\n")
        for i in range(
            len(save_data.data)
        ):  # This is a slow way to write the data, but there should only ever be 200 lines
            line = "\t".join([f"{n:<10.8f}" for n in save_data.data[i, cols]])
            f.write(f"{i}\t")
            f.write(f"{line}\n")
    save_data.filename = filename
    return save_data


@register_loader(
    patterns=(".dat", 16),
    mime_types=[("application/x-wine-extension-ini", 15), ("text/plain", 16)],
    name="QDFile",
    what="Data",
)
def load_qdfile(new_data, *args, **kargs):
    """QD system file loader routine.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    setas = {}
    i = 0
    new_data.filename = filename
    with FileManager(new_data.filename, "r", encoding="utf-8", errors="ignore") as f:  # Read filename linewise
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line != "[Header]":
                raise StonerLoadError("Not a Quantum Design File !")
            if line == "[Header]" or line.startswith(";") or line == "":
                continue
            if "[Data]" in line:
                break
            if "," not in line:
                raise StonerLoadError("No data in file!")
            parts = [x.strip() for x in line.split(",")]
            if parts[1].split(":")[0] == "SEQUENCE FILE":
                key = parts[1].split(":")[0].title()
                value = parts[1].split(":")[1]
            elif parts[0] == "INFO":
                if parts[1] == "APPNAME":
                    parts[1], parts[2] = parts[2], parts[1]
                if len(parts) > 2:
                    key = f"{parts[0]}.{parts[2]}"
                else:
                    raise StonerLoadError("No data in file!")
                key = key.title()
                value = parts[1]
            elif parts[0] in ["BYAPP", "FILEOPENTIME"]:
                key = parts[0].title()
                value = " ".join(parts[1:])
            elif parts[0] == "FIELDGROUP":
                key = f"{parts[0]}.{parts[1]}".title()
                value = f'[{",".join(parts[2:])}]'
            elif parts[0] == "STARTUPAXIS":
                axis = parts[1][0].lower()
                setas[axis] = setas.get(axis, []) + [int(parts[2])]
                key = f"Startupaxis-{parts[1].strip()}"
                value = parts[2].strip()
            else:
                key = parts[0] + "," + parts[1]
                key = key.title()
                value = " ".join(parts[2:])
            new_data.metadata[key] = string_to_type(value)
        else:
            raise StonerLoadError("No data in file!")
        if "Byapp" not in new_data:
            raise StonerLoadError("Not a Quantum Design File !")

        column_headers = f.readline().strip().split(",")
        data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", delimiter=",", invalid_raise=False)
        if data.shape[0] == 0:
            raise StonerLoadError("No data in file!")
        if data.shape[1] < len(column_headers):  # Trap for buggy QD software not giving ewnough columns of data
            data = np.append(data, np.ones((data.shape[0], len(column_headers) - data.shape[1])) * np.nan, axis=1)
        elif data.shape[1] > len(column_headers):  # too much data
            data = data[:, : len(column_headers) - data.shape[1]]
        new_data.data = data
    new_data.column_headers = column_headers
    s = new_data.setas
    for k in setas:
        for ix in setas[k]:
            s[ix - 1] = k
    new_data.setas = s
    return new_data


def _to_Q(new_data, wavelength=1.540593):
    """Add an additional function to covert an angualr scale to momentum transfer.

    Returns:
        a copy of itnew_data.
    """
    new_data.add_column(
        (4 * np.pi / wavelength) * np.sin(np.pi * new_data.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)"
    )


@register_loader(
    patterns=(".ras", 16),
    mime_types=[("application/x-wine-extension-ini", 16), ("text/plain", 16)],
    name="RigakuFile",
    what="Data",
)
def load_rigaku(new_data, *args, **kargs):
    """Read a Rigaku ras file including handling the metadata nicely.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    sh = re.compile(r"^\*([^\s]+)\s+(.*)$")  # Regexp to grab the keys
    ka = re.compile(r"(.*)\-(\d+)$")
    header = {}
    i = 0
    new_data.filename = filename
    with SizedFileManager(new_data.filename, "rb") as (f, end):
        for i, line in enumerate(f):
            line = bytes2str(line).strip()
            if i == 0 and not line.startswith("*RAS_"):
                raise StonerLoadError("Not a Rigaku file!")
            if line == "*RAS_HEADER_START":
                break
        for line in f:
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
            except (TypeError, ValueError, SyntaxError):
                newvalue = literal_eval(value)
            if newvalue == "-":
                newvalue = np.nan  # trap for missing float value
            if m:
                key = m.groups()[0]
                idx = int(m.groups()[1])
                if key in new_data.metadata and not (isinstance(new_data[key], (np.ndarray, list))):
                    if isinstance(new_data[key], str):
                        new_data[key] = list([new_data[key]])
                        if idx > 1:
                            new_data[key].extend([""] * idx - 1)
                    else:
                        new_data[key] = np.array(new_data[key])
                        if idx > 1:
                            new_data[key] = np.append(new_data[key], np.ones(idx - 1) * np.nan)
                if key not in new_data.metadata:
                    if isinstance(newvalue, str):
                        listval = [""] * (idx + 1)
                        listval[idx] = newvalue
                        new_data[key] = listval
                    else:
                        arrayval = np.ones(idx + 1) * np.nan
                        arrayval = arrayval.astype(type(newvalue))
                        arrayval[idx] = newvalue
                        new_data[key] = arrayval
                else:
                    if isinstance(new_data[key][0], str) and isinstance(new_data[key], list):
                        if len(new_data[key]) < idx + 1:
                            new_data[key].extend([""] * (idx + 1 - len(new_data[key])))
                        new_data[key][idx] = newvalue
                    else:
                        if idx + 1 > new_data[key].size:
                            new_data[key] = np.append(
                                new_data[key],
                                (np.ones(idx + 1 - new_data[key].size) * np.nan).astype(new_data[key].dtype),
                            )
                        try:
                            new_data[key][idx] = newvalue
                        except ValueError:
                            pass
            else:
                new_data.metadata[key] = newvalue

        pos = f.tell()
        max_rows = 0
        for max_rows, line in enumerate(f):
            line = bytes2str(line).strip()
            if "RAS_INT_END" in line:
                break
        f.seek(pos)
        if max_rows > 0:
            new_data.data = np.genfromtxt(
                f, dtype="float", delimiter=" ", invalid_raise=False, comments="*", max_rows=max_rows
            )
            column_headers = ["Column" + str(i) for i in range(new_data.data.shape[1])]
            column_headers[0:2] = [new_data.metadata["meas.scan.unit.x"], new_data.metadata["meas.scan.unit.y"]]
            for key in new_data.metadata:
                if isinstance(new_data[key], list):
                    new_data[key] = np.array(new_data[key])
            new_data.setas = "xy"
            new_data.column_headers = column_headers
        pos = f.tell()
    if pos < end:  # Trap for Rigaku files with multiple scans in them.
        new_data["_endpos"] = pos
        if hasattr(filename, "seekable") and filename.seekable():
            filename.seek(pos)
    if kargs.pop("add_Q", False):
        _to_Q(new_data)
    return new_data


def _read_spc_xdata(new_data, f):
    """Read the xdata from the spc file."""
    new_data._pts = new_data._header["fnpts"]
    if new_data._header["ftflgs"] & 128:  # We need to read some X Data
        if 4 * new_data._pts > new_data._filesize - f.tell():
            raise StonerLoadError("Trying to read too much data!")
        xvals = f.read(4 * new_data._pts)  # I think storing X vals directly implies that each one is 4 bytes....
        xdata = np.array(struct.unpack(str2bytes(str(new_data._pts) + "f"), xvals))
    else:  # Generate the X Data ourselves
        first = new_data._header["ffirst"]
        last = new_data._header["flast"]
        if new_data._pts > 1e6:  # Something not right here !
            raise StonerLoadError("More than 1 million points requested. Bugging out now!")
        xdata = np.linspace(first, last, new_data._pts)
    return xdata


def _read_spc_ydata(new_data, f, data, column_headers):
    """Read the y data and column headers from spc file."""
    n = new_data._header["fnsub"]
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
    if new_data._header["ftflgs"] & 1:
        y_width = 2
        y_fmt = "h"
        divisor = 2**16
    else:
        y_width = 4
        y_fmt = "i"
        divisor = 2**32
    if n * (y_width * new_data._pts + 32) > new_data._filesize - f.tell():
        raise StonerLoadError("No good, going to read too much data!")
    for j in range(n):  # We have n sub-scans
        # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
        subhdr = struct.unpack(b"BBHfffIIf4s", f.read(32))
        subheader = dict(zip(["scan" + str(j) + ":" + x for x in subhdr_keys], subhdr))

        # Now read the y-data
        exponent = subheader["scan" + str(j) + ":subexp"]
        if int(exponent) & -128:  # Data is unscaled direct floats
            ydata = np.array(struct.unpack(str2bytes(str(new_data._pts) + "f"), f.read(new_data._pts * y_width)))
        else:  # Data is scaled by exponent
            yvals = struct.unpack(str2bytes(str(new_data._pts) + y_fmt), f.read(new_data._pts * y_width))
            ydata = np.array(yvals, dtype="float64") * (2**exponent) / divisor
        data[:, j + 1] = ydata
        new_data._header = dict(new_data._header, **subheader)
        column_headers.append("Scan" + str(j) + ":" + new_data._yvars[new_data._header["fytype"]])

    return data


def _read_spc_loginfo(new_data, f):
    """Read the log info section of the spc file."""
    logstc = struct.unpack(b"IIIII44s", f.read(64))
    logstc_keys = ("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
    logheader = dict(zip(logstc_keys, logstc))
    new_data._header = dict(new_data._header, **logheader)

    # Can't handle either binary log information or ion disk log information (wtf is this anyway !)
    if new_data._header["logbins"] + new_data._header["logdsks"] > new_data._filesize - f.tell():
        raise StonerLoadError("Too much logfile data to read")
    f.read(new_data._header["logbins"] + new_data._header["logdsks"])

    # The renishaw seems to put a 16 character timestamp next - it's not in the spec but never mind that.
    new_data._header["Date-Time"] = f.read(16)
    # Now read the rest of the file as log text
    logtext = f.read()
    # We expect things to be single lines terminated with a CR-LF of the format key=value
    for line in re.split(b"[\r\n]+", logtext):
        if b"=" in line:
            parts = line.split(b"=")
            key = parts[0].decode()
            value = parts[1].decode()
            new_data._header[key] = value


@register_loader(patterns=(".spc", 16), mime_types=("application/octet-stream", 16), name="SPCFile", what="Data")
def load_spc(new_data, *args, **kargs):
    """Read a .scf file produced by the Renishaw Raman system (among others).

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.

    Todo:
        Implement the second form of the file that stores multiple x-y curves in the one file.

    Notes:
        Metadata keys are pretty much as specified in the spc.h file that defines the filerformat.
    """
    filename, args, kargs = get_filename(args, kargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    # Open the file and read the main file header and unpack into a dict
    with SizedFileManager(filename, "rb") as (f, length):
        new_data._filesize = length
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
        new_data._xvars = [
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
        new_data._yvars = [
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

        new_data._header = dict(zip(keys, spchdr))
        n = new_data._header["fnsub"]

        if new_data._header["ftflgs"] & 64 == 64 or not (
            75 <= new_data._header["fversn"] <= 77
        ):  # This is the multiple XY curves in file flag.
            raise StonerLoadError(
                f"Filetype not implemented yet ! {new_data._header['ftflgs']=}, {new_data._header['fversn']=}"
            )
        # Read the xdata and add it to the file.
        xdata = _read_spc_xdata(new_data, f)
        data = np.zeros((new_data._pts, (n + 1)))  # initialise the data soace
        data[:, 0] = xdata  # Put in the X-Data
        column_headers = [new_data._xvars[new_data._header["fxtype"]]]  # And label the X column correctly

        # Now we're going to read the Y-data
        data = _read_spc_ydata(new_data, f, data, column_headers)
        if new_data._header["flogoff"] != 0:  # Ok, we've got a log, so read the log header and merge into metadata
            _read_spc_loginfo(new_data, f)
        # Ok now build the Stoner.DataFile instance to return
        new_data.data = data
        # The next bit generates the metadata. We don't just copy the metadata because we need to figure out
        # the typehints first - hence the loop
        # here to call DataFile.__setitem()
        for x in new_data._header:
            new_data[x] = new_data._header[x]
        new_data.column_headers = column_headers
        if len(new_data.column_headers) == 2:
            new_data.setas = "xy"
        return new_data


@register_loader(patterns=[(".fld", 16), (".dat", 32)], mime_types=("text/plain", 16), name="VSMFile", what="Data")
def load_vsm(new_data, *args, header_line=3, data_line=3, header_delim=",", **kargs):
    """VSM file loader routine.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Keyword Arguments:
        header_line (int):
            The line in the file that contains the column headers. If None, then column headers are automatically
            generated.
        data_line (int):
            The line on which the data starts
        header_delim (strong):
            The delimiter used for separating header values

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    try:
        with FileManager(filename, errors="ignore", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    first = line.strip()
                    new_data["Timestamp"] = first
                    check = datetime.strptime(first, "%a %b %d %H:%M:%S %Y")
                    if check is None:
                        raise StonerLoadError("Not a VSM file ?")
                elif i == 1:
                    assertion(line.strip() == "")
                elif i == 2:
                    header_string = line.strip()
                elif i == header_line:
                    unit_string = line.strip()
                    column_headers = [
                        f"{h.strip()} ({u.strip()})"
                        for h, u in zip(header_string.split(header_delim), unit_string.split(header_delim))
                    ]
                elif i > 3:
                    break
    except (StonerAssertionError, ValueError, AssertionError, TypeError) as err:
        raise StonerLoadError(f"Not a VSM File {err}") from err
    new_data.data = np.genfromtxt(
        new_data.filename,
        dtype="float",
        usemask=True,
        skip_header=data_line - 1,
        missing_values=["6:0", "---"],
        invalid_raise=False,
    )

    new_data.data = np.ma.mask_rows(new_data.data)
    cols = new_data.data.shape[1]
    new_data.data = np.reshape(new_data.data.compressed(), (-1, cols))
    new_data.column_headers = column_headers
    new_data.setas(x="H_vsm (T)", y="m (emu)")  # pylint: disable=not-callable
    return new_data


@register_loader(
    patterns=".dql",
    mime_types=[("application/x-wine-extension-ini", 16), ("text/plain", 16)],
    name="XRDFile",
    what="Data",
)
def load_xrd(new_data, *args, **kargs):
    """Read an XRD DataFile as produced by the Brucker diffractometer.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.

    Notes:
        Format is ini file like but not enough to do standard inifile processing - in particular
        one can have multiple sections with the same name (!)
    """
    filename, args, kargs = get_filename(args, kargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    sh = re.compile(r"\[(.+)\]")  # Regexp to grab section name
    with FileManager(new_data.filename, errors="ignore", encoding="utf-8") as f:  # Read filename linewise
        if f.readline().strip() != ";RAW4.00":  # Check we have the correct fileformat
            raise StonerLoadError("File Format Not Recognized !")
        drive = 0
        for line in f:  # for each line
            m = sh.search(line)
            if m:  # This is a new section
                section = m.group(1)
                if section == "Drive":  # If this is a Drive section we need to know which Drive Section it is
                    section = section + str(drive)
                    drive = drive + 1
                elif section == "Data":  # Data section contains the business but has a redundant first line
                    f.readline()
                for line in f:  # Now start reading lines in this section...
                    if line.strip() == "":
                        # A blank line marks the end of the section, so go back to the outer loop which will
                        # handle a new section
                        break
                    if section == "Data":  # In the Data section read lines of data value,vale
                        parts = line.split(",")
                        angle = parts[0].strip()
                        counts = parts[1].strip()
                        dataline = np.array([float(angle), float(counts)])
                        new_data.data = np.append(new_data.data, dataline)
                    else:  # Other sections contain metadata
                        parts = line.split("=")
                        key = parts[0].strip()
                        data = parts[1].strip()
                        # Keynames in main metadata are section:key - use theDataFile magic to do type
                        # determination
                        new_data[section + ":" + key] = string_to_type(data)
        column_headers = ["Angle", "Counts"]  # Assume the columns were Angles and Counts

    new_data.data = np.reshape(new_data.data, (-1, 2))
    new_data.setas = "xy"
    new_data._public_attrs = {"four_bounce": bool}
    new_data.four_bounce = new_data["HardwareConfiguration:Monochromator"] == 1
    new_data.column_headers = column_headers
    if kargs.pop("Q", False):
        _to_Q(new_data)
    return new_data
