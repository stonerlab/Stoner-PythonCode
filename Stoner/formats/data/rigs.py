#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile like classes for Various experimental rigs."""

import re
import csv

import numpy as np

from ...compat import bytes2str
from ...core.base import string_to_type
from ...core.exceptions import StonerLoadError
from ..data.generic import load_csvfile
from ...tools.file import FileManager, get_filename

from ..decorators import register_loader


@register_loader(
    patterns=[(".dat", 64), (".iv", 64), (".rvt", 64)], mime_types=("text/plain", 64), name="BigBlueFile", what="Data"
)
def load_bigblue(new_data, *args, **kargs):
    """Just call the parent class but with the right parameters set.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename

    new_data = load_csvfile(new_data, filename, *args, header_line=3, data_line=7, data_delim=" ", header_delim=",")
    if np.all(np.isnan(new_data.data)):
        raise StonerLoadError("All data was NaN in Big Blue format")
    return new_data


@register_loader(patterns=(".dat", 32), mime_types=("text/plain", 32), name="BirgeIVFile", what="Data")
def load_birge(new_data, *args, **kargs):
    """File loader for PinkLib.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    ix = 0
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
        if not re.compile(r"\d{1,2}/\d{1,2}/\d{4}").match(f.readline()):
            raise StonerLoadError("Not a BirgeIVFile as no date on first line")
        data = f.readlines()
        expected = ["Vo(-))", "Vo(+))", "Ic(+)", "Ic(-)"]
        for length, m in zip(data[-4:], expected):
            if not length.startswith(m):
                raise StonerLoadError("Not a BirgeIVFile as wrong footer line")
            key = length[: len(m)]
            val = length[len(m) :]
            if "STDEV" in val:
                ix2 = val.index("STDEV")
                key2 = val[ix2 : ix2 + 4 + len(key)]
                val2 = val[ix2 + 4 + len(key) :]
                new_data.metadata[key2] = string_to_type(val2.strip())
                val = val[:ix2]
            new_data.metadata[key] = string_to_type(val.strip())
        for ix, line in enumerate(data):  # Scan the ough lines to get metadata
            if ":" in line:
                parts = line.split(":")
                new_data.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
            elif "," in line:
                for part in line.split(","):
                    parts = part.split(" ")
                    new_data.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
            elif line.startswith("H "):
                new_data.metadata["H"] = string_to_type(line.split(" ")[1].strip())
            else:
                headers = [x.strip() for x in line.split(" ")]
                break
        else:
            raise StonerLoadError("Oops ran off the end of the file!")
    new_data.data = np.genfromtxt(filename, skip_header=ix + 2, skip_footer=4)
    new_data.column_headers = headers

    new_data.setas = "xy"
    return new_data


@register_loader(patterns=[(".dat", 16), (".txt", 16)], mime_types=("text/plain", 16), name="MokeFile", what="Data")
def load_old_moke(new_data, *args, **kargs):
    """Leeds  MOKE file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    with FileManager(new_data.filename, mode="rb") as f:
        line = bytes2str(f.readline()).strip()
        if line != "#Leeds CM Physics MOKE":
            raise StonerLoadError("Not a Core.DataFile from the Leeds MOKE")
        while line.startswith("#") or line == "":
            parts = line.split(":")
            if len(parts) > 1:
                key = parts[0][1:]
                data = ":".join(parts[1:]).strip()
                new_data[key] = data
            line = bytes2str(f.readline()).strip()
        column_headers = [x.strip() for x in line.split(",")]
        new_data.data = np.genfromtxt(f, delimiter=",")
    new_data.setas = "xy.de"
    new_data.column_headers = column_headers
    return new_data


@register_loader(patterns=(".dat", 16), mime_types=("text/plain", 16), name="FmokeFile", what="Data")
def load_fmoke(new_data, *args, **kargs):
    """Sheffield Focussed MOKE file loader routine.

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
    with FileManager(new_data.filename, mode="rb") as f:
        try:
            value = [float(x.strip()) for x in bytes2str(f.readline()).split("\t")]
        except (TypeError, ValueError) as err:
            f.close()
            raise StonerLoadError("Not an FMOKE file?") from err
        label = [x.strip() for x in bytes2str(f.readline()).split("\t")]
        if label[0] != "Header:":
            f.close()
            raise StonerLoadError("Not a Focussed MOKE file !")
        del label[0]
        for k, v in zip(label, value):
            new_data.metadata[k] = v  # Create metadata from first 2 lines
        column_headers = [x.strip() for x in bytes2str(f.readline()).split("\t")]
        new_data.data = np.genfromtxt(f, dtype="float", delimiter="\t", invalid_raise=False)
        new_data.column_headers = column_headers
    return new_data


def _extend_columns(new_data, i):
    """Ensure the column headers are at least i long."""
    if len(new_data.column_headers) < i:
        length = len(new_data.column_headers)
        new_data.data = np.append(
            new_data.data, np.zeros((new_data.shape[0], i - length)), axis=1
        )  # Need to expand the array first
        new_data.column_headers.extend([f"Column {x}" for x in range(length, i)])


def _et_cmd(new_data, parts):
    """Handle axis labellling command."""
    if parts[0] == "x":
        _extend_columns(new_data, 1)
        new_data.column_headers[0] = parts[1]
    elif parts[0] == "y":
        _extend_columns(new_data, 2)
        new_data.column_headers[1] = parts[1]
    elif parts[0] == "g":
        new_data["title"] = parts[1]


def _td_cmd(new_data, parts):
    new_data.setas = parts[0]


def _sa_cmd(new_data, parts):
    """Implement the sa (set-axis?) command."""
    if parts[0] == "l":  # Legend
        col = int(parts[2])
        _extend_columns(new_data, col + 1)
        new_data.column_headers[col] = parts[1]


@register_loader(patterns=("*", 64), mime_types=("text/plain", 64), name="EasyPlotFile", what="Data")
def load_easyplot(new_data, *args, **kargs):
    """Private loader method."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    delimiter = kargs.pop("delimiter", None)

    datastart = -1
    dataend = -1

    i = 0
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as data:
        if "******** EasyPlot save file ********" not in data.read(1024):
            raise StonerLoadError("Not an EasyPlot Save file?")
        data.seek(0)
        for i, line in enumerate(data):
            line = line.strip()
            if line == "":
                continue
            if line[0] not in "-0123456789" and dataend < 0 <= datastart:
                dataend = i
            if line.startswith('"') and ":" in line:
                parts = [x.strip() for x in line.strip('"').split(":")]
                new_data[parts[0]] = string_to_type(":".join(parts[1:]))
            elif line.startswith("/"):  # command
                parts = [x.strip('"') for x in next(csv.reader([line], delimiter=" ")) if x != ""]
                cmd = parts[0].strip("/")
                if len(cmd) > 1:
                    cmdname = f"_{cmd}_cmd"
                    if cmdname in globals():
                        cmd = globals()[cmdname]
                        cmd(new_data, parts[1:])
                    else:
                        if len(parts[1:]) > 1:
                            cmd = cmd + "." + parts[1]
                            value = ",".join(parts[2:])
                        elif len(parts[1:]) == 1:
                            value = parts[1]
                        else:
                            value = True
                        new_data[cmd] = value
            elif line[0] in "-0123456789" and datastart < 0:  # start of data
                datastart = i
                if "," in line:
                    delimiter = ","
                else:
                    delimiter = None
    if dataend < 0:
        dataend = i
    new_data.data = np.genfromtxt(
        new_data.filename, skip_header=datastart, skip_footer=i - dataend, delimiter=delimiter
    )
    if new_data.data.shape[1] == 2:
        new_data.setas = "xy"
    return new_data


@register_loader(patterns=(".dat", 64), mime_types=("text/plain", 64), name="PinkLibFile", what="Data")
def load_pinklib(new_data, *args, **kargs):
    """File loader for PinkLib.

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
    header_line = 0
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
        if "PINKlibrary" not in f.readline():
            raise StonerLoadError("Not a PINK file")
        f = f.readlines()
        happened_before = False
        for i, line in enumerate(f):
            if line[0] != "#" and not happened_before:
                header_line = i - 2  # -2 because there's a commented out data line
                happened_before = True
                continue  # want to get the metadata at the bottom of the file too
            if any(s in line for s in ("Start time", "End time", "Title")):
                tmp = line.strip("#").split(":")
                new_data.metadata[tmp[0].strip()] = ":".join(tmp[1:]).strip()
        column_headers = f[header_line].strip("#\t ").split("\t")
        data = np.genfromtxt(f, dtype="float", delimiter="\t", invalid_raise=False, comments="#")
    new_data.data = data[:, 0:-2]  # Deal with an errant tab at the end of each line
    new_data.column_headers = column_headers
    if np.all([h in column_headers for h in ("T (C)", "R (Ohm)")]):
        new_data.setas(x="T (C)", y="R (Ohm)")  # pylint: disable=not-callable
    return new_data
