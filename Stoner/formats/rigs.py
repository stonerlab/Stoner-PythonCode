#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile like classes for Various experimental rigs."""

__all__ = ["BigBlueFile", "BirgeIVFile", "MokeFile", "FmokeFile", "EasyPlotFile", "PinkLibFile"]
import csv
import re
import warnings

import numpy as np
import pandas as pd

from .. import Core
from ..compat import bytes2str
from ..core.base import string_to_type
from ..tools.decorators import make_Data, register_loader
from ..tools.file import FileManager
from .generic import CSVFile


@register_loader(patterns=["*.dat", "*.iv", "*.rvt"], priority=64, description="Old VB6 Big Blue code files")
def BigBlueFile(filename, **kargs):
    """Just call the parent class but with the right parameters set.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """

    instance = CSVFile(filename, header_line=3, data_line=7, data_delim=" ", header_delim=",")
    if np.all(np.isnan(instance.data)):
        raise Core.StonerLoadError("All data was NaN in Big Blue format")
    return instance


@register_loader(patterns=["*.dat"], priority=64, description="Birge Group IV files")
def BirgeIVFile(filename, **kargs):
    """File loader for PinkLib.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    ix = 0
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
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
                instance.metadata[key2] = string_to_type(val2.strip())
                val = val[:ix2]
            instance.metadata[key] = string_to_type(val.strip())
        for ix, line in enumerate(data):  # Scan the ough lines to get metadata
            if ":" in line:
                parts = line.split(":")
                instance.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
            elif "," in line:
                for part in line.split(","):
                    parts = part.split(" ")
                    instance.metadata[parts[0].strip()] = string_to_type(parts[1].strip())
            elif line.startswith("H "):
                instance.metadata["H"] = string_to_type(line.split(" ")[1].strip())
            else:
                headers = [x.strip() for x in line.split(" ")]
                break
        else:
            raise Core.StonerLoadError("Oops ran off the end of the file!")
    instance.data = np.genfromtxt(filename, skip_header=ix + 2, skip_footer=4)
    instance.column_headers = headers

    instance.setas = "xy"
    return instance


@register_loader(patterns=["*.dat", "*.txt"], priority=16, description="Old VB6 Code MOKE files")
def MokeFile(filename, **kargs):
    """Leeds  MOKE file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data)
    instance.filename = filename
    with FileManager(instance.filename, mode="rb") as f:
        line = bytes2str(f.readline()).strip()
        if line != "#Leeds CM Physics MOKE":
            raise Core.StonerLoadError("Not a Core.DataFile from the Leeds MOKE")
        while line.startswith("#") or line == "":
            parts = line.split(":")
            if len(parts) > 1:
                key = parts[0][1:]
                data = ":".join(parts[1:]).strip()
                instance[key] = data
            line = bytes2str(f.readline()).strip()
        column_headers = [x.strip() for x in line.split(",")]
        instance.data = np.genfromtxt(f, delimiter=",")
    instance.setas = "xy.de"
    instance.column_headers = column_headers
    return instance


@register_loader(patterns=["*.dat"], priority=16, description="Sheffield focussed MOKE files")
def FmokeFile(filename, **kargs):
    """Sheffield Focussed MOKE file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    with FileManager(instance.filename, mode="rb") as f:
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
            instance.metadata[k] = v  # Create metatdata from first 2 lines
        column_headers = [x.strip() for x in bytes2str(f.readline()).split("\t")]
        instance.data = np.genfromtxt(f, dtype="float", delimiter="\t", invalid_raise=False)
        instance.column_headers = column_headers
    return instance


def _ep_extend_columns(instance, i):
    """Ensure the column headers are at least i long."""
    if len(instance.column_headers) < i:
        length = len(instance.column_headers)
        new_columns = pd.DataFrame(
            np.zeros((instance.shape[0], i - length)), columns=[f"Column {x}" for x in range(length, i)]
        )
        instance &= new_columns
    return instance


def _ep_et_cmd(instance, parts):
    """Handle axis labellling command."""
    if parts[0] == "x":
        instance = _ep_extend_columns(instance, 1)
        instance.column_headers[0] = parts[1]
    elif parts[0] == "y":
        instance = _ep_extend_columns(instance, 2)
        instance.column_headers[1] = parts[1]
    elif parts[0] == "g":
        instance["title"] = parts[1]
    return instance


def _ep_td_cmd(instance, parts):
    instance.setas = parts[0]
    return instance


def _ep_sa_cmd(instance, parts):
    """Implement the sa (set-axis?) command."""
    if parts[0] == "l":  # Legend
        col = int(parts[2])
        instance = _ep_extend_columns(instance, col + 1)
        instance.column_headers[col] = parts[1]
    return instance


@register_loader(patterns=["*.txt", "*.dat", "*.scn"], priority=32, description="Easy Plot file")
def EasyPlotFile(filename, **kargs):
    """Private loader method."""
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename

    datastart = -1
    dataend = -1

    i = 0
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as data:
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
                instance[parts[0]] = string_to_type(":".join(parts[1:]))
            elif line.startswith("/"):  # command
                parts = [x.strip('"') for x in next(csv.reader([line], delimiter=" ")) if x != ""]
                cmd = parts[0].strip("/")
                if len(cmd) > 1:
                    cmdfunc = globals().get(f"_ep_{cmd}_cmd", None)
                    if cmdfunc is None:
                        if len(parts[1:]) > 1:
                            cmd = cmd + "." + parts[1]
                            value = ",".join(parts[2:])
                        elif len(parts[1:]) == 1:
                            value = parts[1]
                        else:
                            value = True
                        instance[cmd] = value
                    else:
                        instance = cmdfunc(instance, parts[1:])

            elif line[0] in "-0123456789" and datastart < 0:  # start of data
                datastart = i
                if "," in line:
                    delimiter = ","
                else:
                    delimiter = None
    if dataend < 0:
        dataend = i
    instance.data = np.genfromtxt(
        instance.filename, skip_header=datastart, skip_footer=i - dataend, delimiter=delimiter
    )
    if instance.data.shape[1] == 2:
        instance.setas = "xy"
    return instance


@register_loader(patterns=["*.dat"], priority=32, description="Mark de Vries' pinklib daat file format")
def PinkLibFile(filename, **kargs):
    """File loader for PinkLib.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as f:  # Read filename linewise
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
                instance.metadata[tmp[0].strip()] = ":".join(tmp[1:]).strip()
        column_headers = f[header_line].strip("#\t ").split("\t")
        data = np.genfromtxt(f, dtype="float", delimiter="\t", invalid_raise=False, comments="#")
    instance.data = data[:, 0:-2]  # Deal with an errant tab at the end of each line
    instance.column_headers = column_headers
    if np.all([h in column_headers for h in ("T (C)", "R (Ohm)")]):
        instance.setas(x="T (C)", y="R (Ohm)")  # pylint: disable=not-callable
    return instance
