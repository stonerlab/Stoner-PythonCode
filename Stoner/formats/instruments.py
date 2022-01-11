#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile like classes to represent Instrument Manufacturer's File Formats'."""
__all__ = ["LSTemperatureFile", "QDFile", "RigakuFile", "SPCFile", "XRDFile"]

import re
import struct
from ast import literal_eval

# Standard Library imports
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .. import Core
from ..compat import bytes2str, str2bytes
from ..core.base import string_to_type
from ..core.exceptions import StonerAssertionError, StonerLoadError, assertion
from ..tools.decorators import make_Data, register_loader
from ..tools.file import FileManager, SizedFileManager


def _r_float(f: Any, length: int = 4) -> float:
    """Read 4 bytes and convert to float."""
    return struct.unpack("<f", f.read(length))[0]


def _r_int(f: Any) -> int:
    """Read 2 bytes and convert to int."""
    return int(struct.unpack("<H", f.read(2))[0])


@register_loader(patterns=["*.340"], priority=16, description="Lakeshore Temperature calibration")
def LSTemperatureFile(filename: str, **kargs) -> Core.DataFile:
    """Load data for 340 files."""
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename

    with FileManager(instance.filename, "rb") as data:
        keys = []
        vals = []
        for line in data:
            line = bytes2str(line)
            if line.strip() == "":
                break
            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 2:
                raise Core.StonerLoadError(f"Header doesn't contain two parts at {line.strip()}")
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
            instance[k] = string_to_type(v)
        headers = bytes2str(next(data)).strip().split()
        column_headers = headers[1:]
        dat = np.genfromtxt(data)
        instance.data = dat[:, 1:]
    instance.column_headers = column_headers
    return instance


def to_340(instance: Core.DataFile, filename: str = None, **kargs) -> Core.DataFile:
    """Override the save method to allow CSVFiles to be written out to disc (as a mininmalist output).

    Args:
        filename (string):
            Filename to save as (using the same rules as for the load routines)

    Keyword Arguments:
        deliminator (string):
            Record deliniminator (defaults to a comma)

    Returns:
        A copy of itinstance.
    """
    if filename is None:
        filename = instance.filename
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = instance.__file_dialog("w")
    if instance.shape[1] == 2:  # 2 columns, let's hope they're the right way round!
        cols = [0, 1]
    elif (
        instance.setas.has_xcol and instance.setas.has_ycol
    ):  # Use ycol, x col but assume x is real temperature and y is resistance
        cols = [instance.setas.ycol[0], instance.setas.xcol]
    else:
        cols = range(instance.shape[1])
    with FileManager(filename, "w", errors="ignore", encoding="utf-8", newline="\r\n") as f:
        for k, v in (
            ("Sensor Model", "CX-1070-SD"),
            ("Serial Number", "Unknown"),
            ("Data Format", 4),
            ("SetPoint Limit", 300.0),
            ("Temperature coefficient", 1),
            ("Number of Breakpoints", len(instance)),
        ):
            if k in ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit"]:
                kstr = f"{k+':':16s}"
            else:
                kstr = f"{k}:   "
            v = instance.get(k, v)
            if k == "Data Format":
                units = ["()", "()", "()", "()", "(Log Ohms/Kelvin)", "(Log Ohms/Log Kelvin)"]
                vstr = f"{v}      {units[int(v)]}"
            elif k == "SetPointLimit":
                vstr = f"{v}      (Kelvin)"
            elif k == "Temperature coefficient":
                vstr = f"{v} {['(positive)', '(negative)'][v]}"
            elif k == "Number of Breakpoints":
                vstr = str(len(instance))
            else:
                vstr = str(v)
            f.write(f"{kstr}{vstr}\n")
        f.write("\n")
        f.write("No.   ")
        for i in cols:
            f.write(f"{instance.column_headers[i]:11s}")
        f.write("\n\n")
        for i in range(
            len(instance.data)
        ):  # This is a slow way to write the data, but there should only ever be 200 lines
            line = "\t".join([f"{n:<10.8f}" for n in instance.data[i, cols]])
            f.write(f"{i}\t")
            f.write(f"{line}\n")
    instance.filename = filename
    return instance


@register_loader(
    patterns=["*.dat"],
    mime_types=["application/x-wine-extension-ini", "text/plain"],
    priority=16,
    description="Quantum Design File",
)
def QDFile(filename: str, **kargs) -> Core.DataFile:
    """QD system file loader routine.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    setas = {}
    i = 0
    with FileManager(instance.filename, "r", encoding="utf-8", errors="ignore") as f:  # Read filename linewise
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line != "[Header]":
                raise Core.StonerLoadError("Not a Quantum Design File !")
            if line == "[Header]" or line.startswith(";") or line == "":
                continue
            if "[Data]" in line:
                break
            if "," not in line:
                raise Core.StonerLoadError("No data in file!")
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
                    raise Core.StonerLoadError("No data in file!")
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
            instance[key] = string_to_type(value)
        else:
            raise Core.StonerLoadError("No data in file!")
        if "Byapp" not in instance:
            raise Core.StonerLoadError("Not a Quantum Design File !")

        column_headers = f.readline().strip().split(",")
        data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", delimiter=",", invalid_raise=False)
        if data.shape[0] == 0:
            raise Core.StonerLoadError("No data in file!")
        if data.shape[1] < len(column_headers):  # Trap for buggy QD software not giving ewnough columns of data
            data = np.append(data, np.ones((data.shape[0], len(column_headers) - data.shape[1])) * np.NaN, axis=1)
        elif data.shape[1] > len(column_headers):  # too much data
            data = data[:, : len(column_headers) - data.shape[1]]
        instance.data = data
    instance.column_headers = column_headers
    s = instance.setas
    for k in setas:
        for ix in setas[k]:
            s[ix - 1] = k
    instance.setas = s
    return instance


@register_loader(patterns=["*.ras"], priority=16, description="Rigaku XRD File")
def RigakuFile(filename: str, **kargs) -> Core.DataFile:
    """Read a Rigaku ras file including handling the metadata nicely.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    sh = re.compile(r"^\*([^\s]+)\s+(.*)$")  # Regexp to grab the keys
    ka = re.compile(r"(.*)\-(\d+)$")
    header = dict()
    i = 0
    with SizedFileManager(instance.filename, "rb") as (f, end):
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
                if key in instance.metadata and not (isinstance(instance[key], (np.ndarray, list))):
                    if isinstance(instance[key], str):
                        instance[key] = list([instance[key]])
                        if idx > 1:
                            instance[key].extend([""] * idx - 1)
                    else:
                        instance[key] = np.array(instance[key])
                        if idx > 1:
                            instance[key] = np.append(instance[key], np.ones(idx - 1) * np.nan)
                if key not in instance.metadata:
                    if isinstance(newvalue, str):
                        listval = [""] * (idx + 1)
                        listval[idx] = newvalue
                        instance[key] = listval
                    else:
                        arrayval = np.ones(idx + 1) * np.nan
                        arrayval = arrayval.astype(type(newvalue))
                        arrayval[idx] = newvalue
                        instance[key] = arrayval
                else:
                    if isinstance(instance[key][0], str) and isinstance(instance[key], list):
                        if len(instance[key]) < idx + 1:
                            instance[key].extend([""] * (idx + 1 - len(instance[key])))
                        instance[key][idx] = newvalue
                    else:
                        if idx + 1 > instance[key].size:
                            instance[key] = np.append(
                                instance[key],
                                (np.ones(idx + 1 - instance[key].size) * np.nan).astype(instance[key].dtype),
                            )
                        try:
                            instance[key][idx] = newvalue
                        except ValueError:
                            pass
            else:
                instance[key] = newvalue

        pos = f.tell()
        max_rows = 0
        for max_rows, line in enumerate(f):
            line = bytes2str(line).strip()
            if "RAS_INT_END" in line:
                break
        f.seek(pos)
        if max_rows > 0:
            instance.data = np.genfromtxt(
                f, dtype="float", delimiter=" ", invalid_raise=False, comments="*", max_rows=max_rows
            )
            column_headers = ["Column" + str(i) for i in range(instance.data.shape[1])]
            column_headers[0:2] = [instance["meas.scan.unit.x"], instance["meas.scan.unit.y"]]
            for key in instance.metadata:
                if isinstance(instance[key], list):
                    instance[key] = np.array(instance[key])
            instance.setas = "xy"
            instance.column_headers = column_headers
        pos = f.tell()
    if pos < end:  # Trap for Rigaku files with multiple scans in them.
        instance["_endpos"] = pos
        if hasattr(filename, "seekable") and filename.seekable():
            filename.seek(pos)
    wavelength = instance["*hw.xg.wave.length.alpha1"]
    instance.add_column(
        (4 * np.pi / wavelength) * np.sin(np.pi * instance.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)"
    )
    return instance


def _spc_read_xdata(instance, f: Any, _header: Dict, _filesize: int) -> np.ndarray:
    """Read the xdata from the spc file."""
    _pts = _header["fnpts"]
    if _header["ftflgs"] & 128:  # We need to read some X Data
        if 4 * _pts > _filesize - f.tell():
            raise Core.StonerLoadError("Trying to read too much data!")
        xvals = f.read(4 * _pts)  # I think storing X vals directly implies that each one is 4 bytes....
        xdata = np.array(struct.unpack(str2bytes(str(_pts) + "f"), xvals))
    else:  # Generate the X Data ourselves
        first = _header["ffirst"]
        last = _header["flast"]
        if _pts > 1e6:  # Something not right here !
            raise Core.StonerLoadError("More than 1 million points requested. Bugging out now!")
        xdata = np.linspace(first, last, _pts)
    return xdata


def _spc_read_ydata(
    instance,
    f: Any,
    data: np.ndarray,
    column_headers: List[str],
    _header: Dict,
    _filesize: int,
    _yvars: List[str],
    _pts: int,
) -> np.ndarray:
    """Read the y data and column headers from spc file."""
    n = _header["fnsub"]
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
    if _header["ftflgs"] & 1:
        y_width = 2
        y_fmt = "h"
        divisor = 2 ** 16
    else:
        y_width = 4
        y_fmt = "i"
        divisor = 2 ** 32
    if n * (y_width * _pts + 32) > _filesize - f.tell():
        raise Core.StonerLoadError("No good, going to read too much data!")
    for j in range(n):  # We have n sub-scans
        # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
        subhdr = struct.unpack(b"BBHfffIIf4s", f.read(32))
        subheader = dict(zip(["scan" + str(j) + ":" + x for x in subhdr_keys], subhdr))

        # Now read the y-data
        exponent = subheader["scan" + str(j) + ":subexp"]
        if int(exponent) & -128:  # Data is unscaled direct floats
            ydata = np.array(struct.unpack(str2bytes(str(_pts) + "f"), f.read(_pts * y_width)))
        else:  # Data is scaled by exponent
            yvals = struct.unpack(str2bytes(str(_pts) + y_fmt), f.read(_pts * y_width))
            ydata = np.array(yvals, dtype="float64") * (2 ** exponent) / divisor
        data[:, j + 1] = ydata
        _header.update(subheader)
        column_headers.append("Scan" + str(j) + ":" + _yvars[_header["fytype"]])

    return data


def _spc_read_loginfo(instance, f: Any, _header: Dict, _filesize: int) -> None:
    """Read the log info section of the spc file."""
    logstc = struct.unpack(b"IIIII44s", f.read(64))
    logstc_keys = ("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
    logheader = dict(zip(logstc_keys, logstc))
    _header.update(logheader)

    # Can't handle either binary log information or ion disk log information (wtf is this anyway !)
    if _header["logbins"] + _header["logdsks"] > _filesize - f.tell():
        raise Core.StonerLoadError("Too much logfile data to read")
    f.read(_header["logbins"] + _header["logdsks"])

    # The renishaw seems to put a 16 character timestamp next - it's not in the spec but never mind that.
    _header["Date-Time"] = f.read(16)
    # Now read the rest of the file as log text
    logtext = f.read()
    # We expect things to be single lines terminated with a CR-LF of the format key=value
    for line in re.split(b"[\r\n]+", logtext):
        if b"=" in line:
            parts = line.split(b"=")
            key = parts[0].decode()
            value = parts[1].decode()
            _header[key] = value


@register_loader(
    patterns=["*.spc"], mime_types=["application/octet-stream"], priority=24, description="Spectroscopy (spc) files"
)
def SPCFile(filename, **kargs):
    """Read a .scf file produced by the Renishaw Raman system (amongs others).

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.

    Todo:
        Implement the second form of the file that stores multiple x-y curves in the one file.

    Notes:
        Metadata keys are pretty much as specified in the spc.h file that defines the filerformat.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    # Open the file and read the main file header and unpack into a dict
    with SizedFileManager(filename, "rb") as (f, length):
        _filesize = length
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
        _xvars = [
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
        _yvars = [
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

        _header = dict(zip(keys, spchdr))
        n = _header["fnsub"]
        _pts = _header["fnpts"]
        if _header["ftflgs"] & 64 == 64 or not (
            75 <= _header["fversn"] <= 77
        ):  # This is the multiple XY curves in file flag.
            raise Core.StonerLoadError(
                "Filetype not implemented yet ! ftflgs={ftflgs}, fversn={fversn}".format(**_header) + f"{_header}"
            )
        # Read the xdata and add it to the file.
        xdata = _spc_read_xdata(instance, f, _header, _filesize)
        data = np.zeros((_pts, (n + 1)))  # initialise the data soace
        data[:, 0] = xdata  # Put in the X-Data
        column_headers = [_xvars[_header["fxtype"]]]  # And label the X column correctly

        # Now we're going to read the Y-data
        data = _spc_read_ydata(instance, f, data, column_headers, _header, _filesize, _yvars, _pts)
        if _header["flogoff"] != 0:  # Ok, we've got a log, so read the log header and merge into metadata
            _spc_read_loginfo(instance, f, _header, _filesize)
        # Ok now build the Stoner.Core.DataFile instance to return
        instance.data = data
        # The next bit generates the metadata. We don't just copy the metadata because we need to figure out
        # the typehints first - hence the loop
        # here to call Core.DataFile.__setitem()
        instance.update(_header)
        instance.column_headers = column_headers
        if len(instance.column_headers) == 2:
            instance.setas = "xy"
        return instance


@register_loader(patterns=["*.fld"], priority=16, description="Oxford Instruments VSM File")
def VSMFile(filename, **kargs):
    """VSM file loader routine.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    header_line = kargs.pop("header_line", 3)
    data_line = kargs.pop("data_line", 3)
    header_delim = kargs.pop("header_delim", ",")
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    try:
        with FileManager(instance.filename, errors="ignore", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    first = line.strip()
                    instance["Timestamp"] = first
                    check = datetime.strptime(first, "%a %b %d %H:%M:%S %Y")
                    if check is None:
                        raise Core.StonerLoadError("Not a VSM file ?")
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
        raise Core.StonerLoadError(f"Not a VSM File {err}") from err
    instance.data = np.genfromtxt(
        instance.filename,
        dtype="float",
        usemask=True,
        skip_header=data_line - 1,
        missing_values=["6:0", "---"],
        invalid_raise=False,
    )

    instance.data = np.ma.mask_rows(instance.data)
    cols = instance.data.shape[1]
    instance.data = instance.data.loc[~np.all(np.isnan(instance.data), axis=1)]
    instance.column_headers = column_headers
    instance.setas(x="H_vsm (T)", y="m (emu)")  # pylint: disable=not-callable
    return instance


@register_loader(
    patterns=["*.dql"],
    mime_types=["application/x-wine-extension-ini", "text/plain"],
    priority=16,
    description="Bruker XRD Files",
)
def XRDFile(filename, **kargs):
    """Read an XRD Core.DataFile as produced by the Brucker diffractometer.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.

    Notes:
        Format is ini file like but not enough to do standard inifile processing - in particular
        one can have multiple sections with the same name (!)
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    sh = re.compile(r"\[(.+)\]")  # Regexp to grab section name
    with FileManager(instance.filename, errors="ignore", encoding="utf-8") as f:  # Read filename linewise
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
                elif section == "Data":
                    data = pd.read_csv(f, sep=",")
                    for k in list(data.columns):
                        if np.all(np.isnan(data[k])):
                            data.drop(k, axis=1, inplace=True)
                for line in f:  # Now start reading lines in this section...
                    if line.strip() == "":
                        # A blank line marks the end of the section, so go back to the outer loop which will
                        # handle a new section
                        break
                    else:  # Other sections contain metadata
                        parts = line.split("=")
                        key = parts[0].strip()
                        dat = parts[1].strip()
                        # Keynames in main metadata are section:key - use theCore.DataFile magic to do type
                        # determination
                        instance[section + ":" + key] = string_to_type(dat)
        column_headers = ["Angle", "Counts"]  # Assume the columns were Angles and Counts

    instance.data = data
    instance.setas = "xy"
    instance["four_bounce"] = instance["HardwareConfiguration:Monochromator"] == 1
    instance.column_headers = column_headers
    wavelength = instance[
        "HardwareConfiguration:AlphaAverage" if not instance["four_bounce"] else "HardwareConfiguration:Alpha1"
    ]
    if kargs.pop("Q", False):
        instance.add_column(
            (4 * np.pi / wavelength) * np.sin(np.pi * instance.column(0) / 360),
            header="Momentum Transfer, Q ($\\AA^{-1}$)",
        )
    return instance


def _raw_read_1(instance, f):
    """"RAW v1 file reader."""
    raise StonerLoadError("Unable to handle version 1 RAW files")


def _raw_read_2(instance, f):
    """RAW v2 file reader."""
    f.seek(4)
    n_blocks = int(struct.unpack("<i", f.read(4))[0])
    f.seek(168)
    instance["Date/Time"] = f.read(20).decode("latin1")
    instance["Anode"] = f.read(2).decode("latin1")
    instance["Ka1"] = _r_float(f)
    instance["Ka2"] = _r_float(f)
    instance["Ka2/Ka1"] = _r_float(f)
    f.seek(206)
    instance["Kb"] = _r_float(f)
    pos = 256
    f.seek(pos)
    blockNum = instance["block"]
    if blockNum >= n_blocks:
        raise StonerLoadError("Tried reading an out of range block!")
    for iBlock in range(blockNum):
        headLen = _r_int(f)
        nSteps = _r_int(f)
        if iBlock < blockNum:
            pos += headLen + 4 * nSteps
            f.seek(pos)
            continue
        f.seek(pos + 12)
        step = _r_float(f)
        start2Th = _r_float(f)
        pos += headLen  # position at start of data block
        f.seek(pos)
        x = np.arange(start2Th, start2Th + step * (nSteps + 1), step)
        y = np.array([max(1.0, _r_float(f)) for i in range(nSteps)])
        instance.data = np.column_stack((x, y))
        instance.column_headers = ["Two Theta", "Counts"]
    instance["repeats"] = blockNum != n_blocks
    return instance


def _raw_read_3(instance, f):
    """RAW v3 file reader."""
    f.seek(12)
    n_blocks = _r_int(f)
    instance["Date/Time"] = f.read(20).decode("latin1")
    f.seek(326)
    instance["Sample="] = f.read(60).decode("latin1")
    f.seek(564)
    radius = _r_float(f)
    instance["Gonio. radius"] = radius
    f.seek(608)
    instance["Anode"] = f.read(2).decode("latin1")
    f.seek(616)
    instance["Ka_mean"] = _r_float(f, 8)
    instance["Ka1"] = _r_float(f, 8)
    instance["Ka2"] = _r_float(f, 8)
    instance["Kb"] = _r_float(f, 8)
    instance["Ka2/Ka1"] = _r_float(f, 8)
    pos = 712
    f.seek(pos)  # position at 1st block header
    blockNum = instance["block"]
    if blockNum >= n_blocks:
        raise StonerLoadError("Tried reading an out of range block!")
    for iBlock in range(blockNum):
        headLen = _r_int(f)
        nSteps = _r_int(f)
        if not nSteps:
            break
        if n_blocks > 1:
            f.seek(pos + 256)
            headLen += _r_float(f)
        else:
            headLen += 40
        if iBlock + 1 != blockNum:
            pos += headLen + 4 * nSteps
            f.seek(pos)
            continue
        f.seek(pos + 8)
        _r_float(f, 8)
        start2Th = _r_float(f, 8)
        f.seek(pos + 212)
        temp = _r_float(f)
        if temp > 0.0:
            instance["Temperature"] = temp
        f.seek(pos + 176)
        step = _r_float(f, 8)
        pos += headLen  # position at start of data block
        f.seek(pos)
        x = np.arange(start2Th, start2Th + step * (nSteps + 1), step)
        try:
            y = np.array([max(1.0, _r_float(f)) for i in range(nSteps)])
        except (ValueError, TypeError, IOError):  # this is absurd
            f.seek(pos - 40)
            y = np.array([max(1.0, _r_float(f)) for i in range(nSteps)])
        w = 1.0 / y
        instance.data = np.column_stack((x, y, w))
        break
    instance["repeats"] = blockNum != n_blocks
    return instance


def _raw_read_4(instance, f):
    """"RAW v1 file reader."""
    raise StonerLoadError("Unable to handle version 4 RAW files")


@register_loader(
    patterns=["*.raw"], mime_types=["application/octet-stream"], priority=16, description="Brucker RAW format XRD File"
)
def BrukerRawFile(filename, **kargs):
    """Read an XRD Core.DataFile as produced by the Brucker diffractometer.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.

    Notes:
        Format is ini file like but not enough to do standard inifile processing - in particular
        one can have multiple sections with the same name (!)
    """
    _header = {
        "RAW ": ("Bruker RAW ver. 1", _raw_read_1),
        "RAW2": ("Bruker RAW ver. 2", _raw_read_2),
        "RAW1.01": ("Bruker RAW ver. 3", _raw_read_3),
        "RAW4.00": ("Bruker RAW ver. 4", _raw_read_4),
    }
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    instance["block"] = kargs.get("block", 1)
    with FileManager(instance.filename, "rb") as f:  # Read filename linewise
        header = f.read(7).decode("latin1")
        val = ("", "_read_4")
        for pat, val in _header.items():
            if header.startswith(pat):
                instance["Version"] = val[0]
                break
        else:
            raise StonerLoadError("Doesn't match a known RAW file format version.")
        reader = val[1]
    return reader(instance, f)
