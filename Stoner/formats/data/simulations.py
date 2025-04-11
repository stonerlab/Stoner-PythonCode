#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements classes to load file formats from various simulation packages."""

import re

import numpy as np

from ...core.exceptions import StonerLoadError
from ...tools.file import FileManager, get_filename

from ..decorators import register_loader


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


@register_loader(patterns=(".dat", 16), mime_types=("text/plain", 16), name="GenXFile", what="Data")
def load_genx(new_data, *args, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    pattern = re.compile(r'# Dataset "([^\"]*)" exported from GenX on (.*)$')
    pattern2 = re.compile(r"#\sFile\sexported\sfrom\sGenX\'s\sReflectivity\splugin")
    i = 0
    ix = 0
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as datafile:
        line = datafile.readline()
        match = pattern.match(line)
        match2 = pattern2.match(line)
        if match is not None:
            dataset = match.groups()[0]
            date = match.groups()[1]
            new_data["date"] = date
            i = 2
        elif match2 is not None:
            line = datafile.readline()
            new_data["date"] = line.split(":")[1].strip()
            dataset = datafile.readline()[1:].strip()
            i = 3
        else:
            raise StonerLoadError("Not a GenXFile")
        for ix, line in enumerate(datafile):
            line = line.strip()
            if line in ["# Headers:", "# Column labels:"]:
                line = next(datafile)[1:].strip()
                break
        else:
            raise StonerLoadError("Cannot find headers")
    skip = ix + i + 2
    column_headers = [f.strip() for f in line.strip().split("\t")]
    new_data.data = np.real(np.genfromtxt(new_data.filename, skip_header=skip, dtype=complex))
    new_data["dataset"] = dataset
    if "sld" in dataset.lower():
        new_data["type"] = "SLD"
    elif "asymmetry" in dataset.lower():
        new_data["type"] = "Asymmetry"
    elif "dd" in dataset.lower():
        new_data["type"] = "Down"
    elif "uu" in dataset.lower():
        new_data["type"] = "Up"
    new_data.column_headers = column_headers
    return new_data


@register_loader(
    patterns=(".ovf", 16),
    mime_types=[("text/plain", 16), ("application/octet-stream", 16)],
    name="OVFFile",
    what="Data",
)
def _load(new_data, *args, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards.

    Notes:
        This code can handle only the first segment in the data file.
    """
    filename, args, kargs = get_filename(args, kargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    # Reading in binary, converting to utf-8 where we should be in the text header
    with FileManager(new_data.filename, "rb") as data:
        line = data.readline().decode("utf-8", errors="ignore").strip("#\n \t\r")
        if line not in ["OOMMF: rectangular mesh v1.0", "OOMMF OVF 2.0"]:
            raise StonerLoadError("Not an OVF 1.0 or 2.0 file.")
        while _read_line(data, new_data.metadata):
            pass  # Read the file until we reach the start of the data block.
        if new_data.metadata["representation"] == "Binary":
            size = (  # Work out the size of the data block to read
                new_data.metadata["xnodes"]
                * new_data.metadata["ynodes"]
                * new_data.metadata["znodes"]
                * new_data.metadata["valuedim"]
                + 1
            ) * new_data.metadata["representation size"]
            bin_data = data.read(size)
            numbers = np.frombuffer(bin_data, dtype=f"<f{new_data.metadata['representation size']}")
            chk = numbers[0]
            if (
                chk != [1234567.0, 123456789012345.0][new_data.metadata["representation size"] // 4 - 1]
            ):  # If we have a good check number we can carry on, otherwise try the other endianness
                numbers = np.frombuffer(bin_data, dtype=f">f{new_data.metadata['representation size']}")
                chk = numbers[0]
                if chk != [1234567.0, 123456789012345.0][new_data.metadata["representation size"] // 4 - 1]:
                    raise StonerLoadError("Bad binary data for ovf gile.")

            data = np.reshape(
                numbers[1:],
                (new_data.metadata["xnodes"] * new_data.metadata["ynodes"] * new_data.metadata["znodes"], 3),
            )
        else:
            data = np.genfromtxt(
                data,
                max_rows=new_data.metadata["xnodes"] * new_data.metadata["ynodes"] * new_data.metadata["znodes"],
            )
    xmin, xmax, xstep = (
        new_data.metadata["xmin"],
        new_data.metadata["xmax"],
        new_data.metadata["xstepsize"],
    )
    ymin, ymax, ystep = (
        new_data.metadata["ymin"],
        new_data.metadata["ymax"],
        new_data.metadata["ystepsize"],
    )
    zmin, zmax, zstep = (
        new_data.metadata["zmin"],
        new_data.metadata["zmax"],
        new_data.metadata["zstepsize"],
    )
    Z, Y, X = np.meshgrid(
        np.arange(zmin + zstep / 2, zmax, zstep) * 1e9,
        np.arange(ymin + ystep / 2, ymax, ystep) * 1e9,
        np.arange(xmin + xstep / 2, xmax, xstep) * 1e9,
        indexing="ij",
    )
    new_data.data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), data))

    column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
    new_data.setas = "xyzuvw"
    new_data.column_headers = column_headers
    return new_data
