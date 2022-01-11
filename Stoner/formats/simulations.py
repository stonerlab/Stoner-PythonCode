#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements classes to load file formats from various simulation packages."""

__all__ = ["GenXFile", "OVFFile"]
import io
import re

import numpy as np

from ..Core import DataFile
from ..core.base import string_to_type
from ..core.exceptions import StonerLoadError, assertion
from ..tools.decorators import make_Data, register_loader
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


@register_loader(patterns=["*.dat"], priority=16, description="GenX Text export file")
def GenXFile(filename, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards."""
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    pattern = re.compile(r'# Dataset "([^\"]*)" exported from GenX on (.*)$')
    pattern2 = re.compile(r"#\sFile\sexported\sfrom\sGenX\'s\sReflectivity\splugin")
    i = 0
    ix = 0
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as datafile:
        line = datafile.readline()
        match = pattern.match(line)
        match2 = pattern2.match(line)
        if match is not None:
            dataset = match.groups()[0]
            date = match.groups()[1]
            instance["date"] = date
            i = 2
        elif match2 is not None:
            line = datafile.readline()
            instance["date"] = line.split(":")[1].strip()
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
    instance.data = np.real(np.genfromtxt(instance.filename, skip_header=skip, dtype=complex))
    instance["dataset"] = dataset
    if "sld" in dataset.lower():
        instance["type"] = "SLD"
    elif "asymmetry" in dataset.lower():
        instance["type"] = "Asymmetry"
    elif "dd" in dataset.lower():
        instance["type"] = "Down"
    elif "uu" in dataset.lower():
        instance["type"] = "Up"
    instance.column_headers = column_headers
    return instance


@register_loader(
    patterns=["*.ovf"],
    mime_types=["text/plain", "application/octet-stream"],
    priority=16,
    description="OOMMF output file format",
)
def OVFFile(filename, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards.

    Notes:
        This code can handle only the first segment in the data file.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    # Reading in binary, converting to utf-8 where we should be in the text header
    with FileManager(instance.filename, "rb") as data:
        line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
        if line not in ["OOMMF: rectangular mesh v1.0", "OOMMF OVF 2.0"]:
            raise StonerLoadError("Not an OVF 1.0 or 2.0 file.")
        while _read_line(data, instance.metadata):
            pass  # Read the file until we reach the start of the data block.
        if instance.metadata["representation"] == "Binary":
            size = (  # Work out the size of the data block to read
                instance.metadata["xnodes"]
                * instance.metadata["ynodes"]
                * instance.metadata["znodes"]
                * instance.metadata["valuedim"]
                + 1
            ) * instance.metadata["representation size"]
            bin_data = data.read(size)
            numbers = np.frombuffer(bin_data, dtype=f"<f{instance.metadata['representation size']}")
            chk = numbers[0]
            if (
                chk != [1234567.0, 123456789012345.0][instance.metadata["representation size"] // 4 - 1]
            ):  # If we have a good check number we can carry on, otherwise try the other endianess
                numbers = np.frombuffer(bin_data, dtype=f">f{instance.metadata['representation size']}")
                chk = numbers[0]
                if chk != [1234567.0, 123456789012345.0][instance.metadata["representation size"] // 4 - 1]:
                    raise StonerLoadError("Bad binary data for ovf gile.")

            data = np.reshape(
                numbers[1:],
                (instance.metadata["xnodes"] * instance.metadata["ynodes"] * instance.metadata["znodes"], 3),
            )
        else:
            data = np.genfromtxt(
                data, max_rows=instance.metadata["xnodes"] * instance.metadata["ynodes"] * instance.metadata["znodes"],
            )
    xmin, xmax, xstep = (
        instance.metadata["xmin"],
        instance.metadata["xmax"],
        instance.metadata["xstepsize"],
    )
    ymin, ymax, ystep = (
        instance.metadata["ymin"],
        instance.metadata["ymax"],
        instance.metadata["ystepsize"],
    )
    zmin, zmax, zstep = (
        instance.metadata["zmin"],
        instance.metadata["zmax"],
        instance.metadata["zstepsize"],
    )
    Z, Y, X = np.meshgrid(
        np.arange(zmin + zstep / 2, zmax, zstep) * 1e9,
        np.arange(ymin + ystep / 2, ymax, ystep) * 1e9,
        np.arange(xmin + xstep / 2, xmax, xstep) * 1e9,
        indexing="ij",
    )
    instance.data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), data))

    column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
    instance.setas = "xyzuvw"
    instance.column_headers = column_headers
    return instance
