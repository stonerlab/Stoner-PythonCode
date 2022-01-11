#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements DataFile like classes for various large scale facilities."""

__all__ = ["BNLFile", "MDAASCIIFile", "OpenGDAFile", "RasorFile", "SNSFile", "ESRF_DataFile", "ESRF_ImageFile"]
# Standard Library imports
import linecache
import re

import numpy as np

from .. import Core, Image
from ..compat import str2bytes
from ..core.base import string_to_type
from ..core.exceptions import StonerLoadError
from ..tools.decorators import make_Data, register_loader
from ..tools.file import FileManager

try:
    import fabio
except ImportError:
    fabio = None


@register_loader(
    patterns=["*.txt"], mime_types=["text/plain"], priority=64, description="Brookhaven National Lab SPEC format"
)
def BNLFile(filename, **kargs):
    """Load the file from disc.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.

    Notes:
        Overwrites load method in Core.DataFile class, no header positions and data
        positions are needed because of the hash title structure used in BNL files.

        Normally its good to use _parse_plain_data method from Core.DataFile class
        to load data but unfortunately Brookhaven data isn't very plain so there's
        a new method below.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    try:
        with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as fp:
            line_numbers = [0, 0, 0, 0, 0]
            counter = 0
            for line in fp:
                counter += 1
                if counter == 1 and line[0] != "#":
                    raise Core.StonerLoadError("Not a BNL File ?")
                if len(line) < 2:
                    continue  # if there's nothing written on the line go to the next
                if line[0:2] == "#L":
                    line_numbers[0] = counter
                elif line[0:2] == "#S":
                    line_numbers[2] = counter
                elif line[0:2] == "#D":
                    line_numbers[3] = counter
                elif line[0:2] == "#P":
                    line_numbers[4] = counter
                elif line[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    line_numbers[1] = counter
                    break
        # creates a list, line_numbers, formatted [header_line,data_line,scan_line,date_line,motor_line]
        header_string = linecache.getline(instance.filename, line_numbers[0])
        header_string = re.sub(r'["\n]', "", header_string)  # get rid of new line character
        header_string = re.sub(r"#L", "", header_string)  # get rid of line indicator character
        column_headers = map(lambda x: x.strip(), header_string.split())
        scanLine = linecache.getline(instance.filename, line_numbers[2])
        dateLine = linecache.getline(instance.filename, line_numbers[3])
        motorLine = linecache.getline(instance.filename, line_numbers[4])
        instance["Snumber"] = scanLine.split()[1]
        tmp = "".join(scanLine.split()[2:])
        instance["Stype"] = "".join(tmp.split(","))  # get rid of commas
        instance["Sdatetime"] = dateLine[3:-1]  # don't want \n at end of line so use -1
        instance["Smotor"] = motorLine.split()[3]
        try:
            instance.data = np.genfromtxt(instance.filename, skip_header=line_numbers[1] - 1)
        except IOError:
            instance.data = np.array([0])
            print(f"Did not import any data for {instance.filename}")
        instance.column_headers = column_headers
    except (IndexError, TypeError, ValueError, StonerLoadError) as err:
        raise StonerLoadError("Not parseable as a NFLS file!") from err
    finally:
        linecache.clearcache()
    return instance


@register_loader(patterns=["*.txt"], mime_types=["text/plain"], priority=16, description="APS MDA ASCII format")
def MDAASCIIFile(filename, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards."""
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    i = [0, 0, 0, 0]
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
        for i[0], line in enumerate(data):
            if (
                i[0] == 0 and line.strip() != "## mda2ascii 1.2 generated output"
            ):  # bug out oif we don't like the header
                raise Core.StonerLoadError("Not a file mda2ascii")
            line.strip()
            if "=" in line:
                parts = line[2:].split("=")
                instance[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
            elif line.startswith("#  Extra PV:"):
                # Onto the next metadata bit
                break
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
                instance[key] = string_to_type(bits[2])
            else:
                break  # End of Extra PV stuff
        else:
            raise Core.StonerLoadError("Overran Extra PV Block")
        for i[2], line in enumerate(data):
            line.strip()
            if line.strip() == "":
                continue
            elif line.startswith("# Column Descriptions:"):
                break  # Start of column headers now
            elif "=" in line:
                parts = line[2:].split("=")
                instance[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
        else:
            raise Core.StonerLoadError("Overran end of scan header before column descriptions")
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
            raise Core.StonerLoadError("Overand the end of file without reading data")
    instance.data = np.genfromtxt(instance.filename, skip_header=sum(i))  # so that's ok then !
    instance.column_headers = column_headers
    return instance


@register_loader(
    patterns=["*.dat"], mime_types=["text/plain"], priority=16, description="Text export format from OpenGDA"
)
def OpenGDAFile(filename, **kargs):
    """Load an OpenGDA file.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itinstance after loading the data.
    """
    instance = kargs.pop("instance", make_Data())
    instance.filename = filename
    i = 0
    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line != "&SRS":
                raise Core.StonerLoadError(f"Not a GDA File from Rasor ?\n{line}")
            if "&END" in line:
                break
            parts = line.split("=")
            if len(parts) != 2:
                continue
            key = parts[0]
            value = parts[1].strip()
            instance.metadata[key] = string_to_type(value)
        if i == 0:
            raise StonerLoadError("Empty fiule processed by OpenGDAFile!")
        column_headers = f.readline().strip().split("\t")
        instance.data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", invalid_raise=False)
    instance.column_headers = column_headers
    return instance


@register_loader(
    patterns=["*.dat"], mime_types=["text/plain"], priority=16, description="Text file from Oak Ridge National SNS"
)
def SNSFile(filename, **kargs):
    """Load function. File format has space delimited columns from row 3 onwards."""

    instance = kargs.pop("instance", make_Data())
    instance.filename = filename

    with FileManager(instance.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
        line = data.readline()
        if not line.strip().startswith(
            "# Datafile created by QuickNXS 0.9.39"
        ):  # bug out oif we don't like the header
            raise Core.StonerLoadError("Not a file from the SNS BL4A line")
        for line in data:
            if line.startswith("# "):  # We're in the header
                line = line[2:].strip()  # strip the header and whitespace

            if line.startswith("["):  # Look for a section header
                section = line.strip().strip("[]")
                if section == "Data":  # The Data section has one line of colum headers and then data
                    header = next(data)[2:].split("\t")
                    column_headers = [h.strip() for h in header]
                    instance.data = np.genfromtxt(data)  # we end by reading the raw data
                elif section == "Global Options":  # This section can go into metadata
                    for line in data:
                        line = line[2:].strip()
                        if line.strip() == "":
                            break
                        else:
                            instance[line[2:10].strip()] = line[11:].strip()
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
                    instance[section] = sec
            else:  # We must still be in the opening un-labelled section of meta data
                if ":" in line:
                    i = line.index(":")
                    key = line[:i].strip()
                    value = line[i + 1 :].strip()
                    instance[key.strip()] = value.strip()
    instance.column_headers = column_headers
    return instance


if fabio:

    @register_loader(
        patterns=["*.edf"],
        mime_types=["application/octet-stream", "text/plain"],
        priority=16,
        description="EDF text file from the XRSF",
    )
    def ESRF_DataFile(filename, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        instance = kargs.pop("isntance", make_Data())
        instance.filename = filename
        try:
            img = fabio.edfimage.edfimage().read(instance.filename)
            instance.data = img.data
            instance.metadata.update(img.header)
            return instance
        except (OSError, ValueError, TypeError, IndexError) as err:
            raise StonerLoadError("Not an ESRF data file !") from err

    class FabioImageFile(Image.ImageFile):

        """Utilise the fabIO library to read an edf file has a DataFile."""

        priority = 32
        patterns = ["*.edf"]
        mime_type = ["text/plain", "application/octet-stream"]

        def _load(instance, filename=None, *args, **kargs):
            """Load function. File format has space delimited columns from row 3 onwards."""
            if filename is None or not filename:
                instance.get_filename("r")
            else:
                instance.filename = filename
            try:
                img = fabio.open(instance.filename)
                instance.image = img.data
                instance.metadata.update(img.header)
                return instance
            except (OSError, ValueError, TypeError, IndexError) as err:
                try:
                    filename.seek(0)
                except AttributeError:
                    pass
                raise StonerLoadError("Not a Fabio Image file !") from err


else:
    ESRF_DataFile = None
    ESRF_ImageFile = None
