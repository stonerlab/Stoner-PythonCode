#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements DataFile like classes for various large scale facilities."""

# Standard Library imports
import linecache
import re

import numpy as np

from ...compat import str2bytes
from ...core.base import string_to_type
from ...tools.file import FileManager, get_filename
from ...core.exceptions import StonerLoadError
from ..decorators import register_loader

try:
    import fabio
except ImportError:
    fabio = None


def _bnl_find_lines(new_data):
    """Return an array of ints [header_line,data_line,scan_line,date_line,motor_line]."""
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as fp:
        new_data.line_numbers = [0, 0, 0, 0, 0]
        counter = 0
        for line in fp:
            counter += 1
            if counter == 1 and line[0] != "#":
                raise StonerLoadError("Not a BNL File ?")
            if len(line) < 2:
                continue  # if there's nothing written on the line go to the next
            if line[0:2] == "#L":
                new_data.line_numbers[0] = counter
            elif line[0:2] == "#S":
                new_data.line_numbers[2] = counter
            elif line[0:2] == "#D":
                new_data.line_numbers[3] = counter
            elif line[0:2] == "#P":
                new_data.line_numbers[4] = counter
            elif line[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                new_data.line_numbers[1] = counter
                break


def _bnl_get_metadata(new_data):
    """Load metadta from file.

    Metadata found is scan number 'Snumber', scan type and parameters 'Stype',
    scan date/time 'Sdatetime' and z motor position 'Smotor'.
    """
    scanLine = linecache.getline(new_data.filename, new_data.line_numbers[2])
    dateLine = linecache.getline(new_data.filename, new_data.line_numbers[3])
    motorLine = linecache.getline(new_data.filename, new_data.line_numbers[4])
    new_data.__setitem__("Snumber", scanLine.split()[1])
    tmp = "".join(scanLine.split()[2:])
    new_data.__setitem__("Stype", "".join(tmp.split(",")))  # get rid of commas
    new_data.__setitem__("Sdatetime", dateLine[3:-1])  # don't want \n at end of line so use -1
    new_data.__setitem__("Smotor", motorLine.split()[3])


def _parse_bnl_data(new_data):
    """Parse BNL data.

     The meta data is labelled by #L type tags
    so easy to find but #L must be excluded from the result.
    """
    _bnl_find_lines(new_data)
    # creates a list, line_numbers, formatted [header_line,data_line,scan_line,date_line,motor_line]
    header_string = linecache.getline(new_data.filename, new_data.line_numbers[0])
    header_string = re.sub(r'["\n]', "", header_string)  # get rid of new line character
    header_string = re.sub(r"#L", "", header_string)  # get rid of line indicator character
    column_headers = map(lambda x: x.strip(), header_string.split())
    _bnl_get_metadata(new_data)
    try:
        new_data.data = np.genfromtxt(new_data.filename, skip_header=new_data.line_numbers[1] - 1)
    except IOError:
        new_data.data = np.array([0])
        print(f"Did not import any data for {new_data.filename}")
    new_data.column_headers = column_headers


@register_loader(patterns=(".txt", 64), mime_types=("text/plain", 64), name="BNLFile", what="Data")
def load_bnl(new_data, *args, **kargs):  # pylint: disable=unused-argument
    """Load the file from disc.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.

    Notes:
        Overwrites load method in Core.DataFile class, no header positions and data
        positions are needed because of the hash title structure used in BNL files.

        Normally its good to use _parse_plain_data method from Core.DataFile class
        to load data but unfortunately Brookhaven data isn't very plain so there's
        a new method below.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    try:
        _parse_bnl_data(new_data)  # call an internal function rather than put it in load function
    except (IndexError, TypeError, ValueError, StonerLoadError) as err:
        raise StonerLoadError("Not parseable as a NFLS file!") from err
    linecache.clearcache()
    return new_data


def _read_mdaascii_header(data, new_data, i):
    """Read the header block."""
    for i[0], line in enumerate(data):
        line.strip()
        if "=" in line:
            parts = line[2:].split("=")
            new_data[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
        elif line.startswith("#  Extra PV:"):
            # Onto the next metadata bit
            break


def _read_mdaascii_metadata(data, new_data, i):
    """Read the metadata block."""
    pvpat = re.compile(r"^#\s+Extra\s+PV\s\d+\:(.*)")
    for i[1], line in enumerate(data):
        if line.strip() == "":
            continue
        if line.startswith("# Extra PV"):
            res = pvpat.match(line)
            bits = [b.strip().strip(r'"') for b in res.group(1).split(",")]
            if bits[1] == "":
                key = bits[0]
            else:
                key = bits[1]
            if len(bits) > 3:
                key = f"{key} ({bits[3]})"
            new_data[key] = string_to_type(bits[2])
        else:
            break  # End of Extra PV stuff
    else:
        raise StonerLoadError("Overran Extra PV Block")
    for i[2], line in enumerate(data):
        line.strip()
        if line.strip() == "":
            continue
        elif line.startswith("# Column Descriptions:"):
            break  # Start of column headers now
        elif "=" in line:
            parts = line[2:].split("=")
            new_data[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
    else:
        raise StonerLoadError("Overran end of scan header before column descriptions")


def _read_mdaascii_columns(data, new_data, i):
    """Reads the column header block."""
    colpat = re.compile(r"#\s+\d+\s+\[([^\]]*)\](.*)")
    column_headers = []
    for i[3], line in enumerate(data):
        res = colpat.match(line)
        line.strip()
        if line.strip() == "":
            continue
        elif line.startswith("# 1-D Scan Values"):
            break  # Start of data
        elif res is not None:
            if "," in res.group(2):
                bits = [b.strip() for b in res.group(2).split(",")]
                if bits[-2] == "":
                    colname = bits[0]
                else:
                    colname = bits[-2]
                if bits[-1] != "":
                    colname += f"({bits[-1]})"
                if colname in column_headers:
                    colname = f"{bits[0]}:{colname}"
            else:
                colname = res.group(1).strip()
            column_headers.append(colname)
    else:
        raise StonerLoadError("Overand the end of file without reading data")
    new_data.column_headers = column_headers


@register_loader(patterns=(".txt", 32), mime_types=("text/plain", 32), name="MDAASCIIFile", what="Data")
def load_mdaasci(new_data, *args, **kargs):  # pylint: disable=unused-argument
    """Load function. File format has space delimited columns from row 3 onwards."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    i = [0, 0, 0, 0]
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
        if next(data).strip() != "## mda2ascii 1.2 generated output":
            raise StonerLoadError("Not a file mda2ascii")
        _read_mdaascii_header(data, new_data, i)
        _read_mdaascii_metadata(data, new_data, i)
        _read_mdaascii_columns(data, new_data, i)
        new_data.data = np.genfromtxt(data)  # so that's ok then !
    return new_data


@register_loader(patterns=(".dat", 16), mime_types=("text/plain", 16), name="OpenGDAFile", what="Data")
def load_gda(new_data, *args, **kargs):
    """Load an OpenGDA file.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    i = 0
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line != "&SRS":
                raise StonerLoadError(f"Not a GDA File from Rasor ?\n{line}")
            if "&END" in line:
                break
            parts = line.split("=")
            if len(parts) != 2:
                continue
            key = parts[0]
            value = parts[1].strip()
            new_data.metadata[key] = string_to_type(value)
        column_headers = f.readline().strip().split("\t")
        new_data.data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", invalid_raise=False)
    new_data.column_headers = column_headers
    return new_data


@register_loader(patterns=(".dat", 16), mime_types=("text/plain", 16), name="SNSFile", what="Data")
def load_sns(new_data, *args, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards."""
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    with FileManager(new_data.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
        line = data.readline()
        if not line.strip().startswith(
            "# Datafile created by QuickNXS 0.9.39"
        ):  # bug out oif we don't like the header
            raise StonerLoadError("Not a file from the SNS BL4A line")
        for line in data:
            if line.startswith("# "):  # We're in the header
                line = line[2:].strip()  # strip the header and whitespace

            if line.startswith("["):  # Look for a section header
                section = line.strip().strip("[]")
                if section == "Data":  # The Data section has one line of column headers and then data
                    header = next(data)[2:].split("\t")
                    column_headers = [h.strip() for h in header]
                    new_data.data = np.genfromtxt(data)  # we end by reading the raw data
                elif section == "Global Options":  # This section can go into metadata
                    for line in data:
                        line = line[2:].strip()
                        if line.strip() == "":
                            break
                        else:
                            new_data[line[2:10].strip()] = line[11:].strip()
                elif (
                    section == "Direct Beam Runs" or section == "Data Runs"
                ):  # These are constructed into lists ofg dictionaries for each file
                    sec = list()
                    header = next(data)
                    header = header[2:].strip()
                    keys = [s.strip() for s in header.split("  ") if s.strip()]
                    for line in data:
                        line = line[2:].strip()
                        if line == "":
                            break
                        else:
                            values = [s.strip() for s in line.split("  ") if s.strip()]
                            sec.append(dict(zip(keys, values)))
                    new_data[section] = sec
            else:  # We must still be in the opening un-labelled section of meta data
                if ":" in line:
                    i = line.index(":")
                    key = line[:i].strip()
                    value = line[i + 1 :].strip()
                    new_data[key.strip()] = value.strip()
    new_data.column_headers = column_headers
    return new_data


if fabio:

    @register_loader(
        patterns=(".edf", 16),
        mime_types=[("application/octet-stream", 16), ("text/plain", 15)],
        name="ESRF_DataFile",
        what="Data",
    )
    def load_esrf(self, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        try:
            img = fabio.edfimage.edfimage().read(self.filename)
            self.data = img.data
            self.metadata.update(img.header)
            return self
        except (OSError, ValueError, TypeError, IndexError) as err:
            raise StonerLoadError("Not an ESRF data file !") from err
