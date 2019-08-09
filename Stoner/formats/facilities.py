#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements DataFile like classes for various large scale facilities
"""
__all__ = ["BNLFile", "MDAASCIIFile", "OpenGDAFile", "RasorFile", "SNSFile"]
# Standard Library imports
import io
import linecache
import re

import numpy as np

import Stoner.Core as Core
from Stoner.compat import python_v3, str2bytes
from Stoner.core.base import string_to_type


class BNLFile(Core.DataFile):

    """
    Creates BNLFile a subclass of Core.DataFile that caters for files in the SPEC format given by BNL (specifically u4b beamline but hopefully generalisable).

    Author Rowan 12/2011

    The file from BNL must be split into seperate scan files before Stoner can use
    them, a separate python script has been written for this and should be found
    in data/Python/PythonCode/scripts.
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 64
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.txt"]  # Recognised filename patterns

    def __init__(self, *params):
        """Constructor modification.

        Do a normal initiation using the parent class 'self' followed by adding an extra attribute line_numbers,
        line_numbers is a list of important line numbers in the file.
        I've left it open for someone to add options for more args if they wish.
        """
        super(BNLFile, self).__init__(*params)
        self.line_numbers = []

    def __find_lines(self):
        """Returns an array of ints [header_line,data_line,scan_line,date_line,motor_line]."""
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as fp:
            self.line_numbers = [0, 0, 0, 0, 0]
            counter = 0
            for line in fp:
                counter += 1
                if counter == 1 and line[0] != "#":
                    raise Core.StonerLoadError("Not a BNL File ?")
                if len(line) < 2:
                    continue  # if there's nothing written on the line go to the next
                elif line[0:2] == "#L":
                    self.line_numbers[0] = counter
                elif line[0:2] == "#S":
                    self.line_numbers[2] = counter
                elif line[0:2] == "#D":
                    self.line_numbers[3] = counter
                elif line[0:2] == "#P":
                    self.line_numbers[4] = counter
                elif line[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    self.line_numbers[1] = counter
                    break

    def __get_metadata(self):
        """Load metadta from file.

        Metadata found is scan number 'Snumber', scan type and parameters 'Stype',
        scan date/time 'Sdatetime' and z motor position 'Smotor'.
        """
        scanLine = linecache.getline(self.filename, self.line_numbers[2])
        dateLine = linecache.getline(self.filename, self.line_numbers[3])
        motorLine = linecache.getline(self.filename, self.line_numbers[4])
        self.__setitem__("Snumber", scanLine.split()[1])
        tmp = "".join(scanLine.split()[2:])
        self.__setitem__("Stype", "".join(tmp.split(",")))  # get rid of commas
        self.__setitem__("Sdatetime", dateLine[3:-1])  # don't want \n at end of line so use -1
        self.__setitem__("Smotor", motorLine.split()[3])

    def __parse_BNL_data(self):
        """Internal function for parsing BNL data.

         The meta data is labelled by #L type tags
        so easy to find but #L must be excluded from the result.
        """
        self.__find_lines()
        # creates a list, line_numbers, formatted [header_line,data_line,scan_line,date_line,motor_line]
        header_string = linecache.getline(self.filename, self.line_numbers[0])
        header_string = re.sub(r'["\n]', "", header_string)  # get rid of new line character
        header_string = re.sub(r"#L", "", header_string)  # get rid of line indicator character
        column_headers = map(lambda x: x.strip(), header_string.split())
        self.__get_metadata()
        try:
            self.data = np.genfromtxt(self.filename, skip_header=self.line_numbers[1] - 1)
        except IOError:
            self.data = np.array([0])
            print("Did not import any data for {}".format(self.filename))
        self.column_headers = column_headers

    def _load(self, filename, *args, **kargs):  # fileType omitted, implicit in class call
        """BNLFile.load(filename)

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.

        Notes:
            Overwrites load method in Core.DataFile class, no header positions and data
            positions are needed because of the hash title structure used in BNL files.

            Normally its good to use _parse_plain_data method from Core.DataFile class
            to load data but unfortunately Brookhaven data isn't very plain so there's
            a new method below.
        """
        self.filename = filename
        self.__parse_BNL_data()  # call an internal function rather than put it in load function
        linecache.clearcache()
        return self


class MDAASCIIFile(Core.DataFile):

    """Reads files generated from the APS."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.txt"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        i = [0, 0, 0, 0]
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
            for i[0], line in enumerate(data):
                if (
                    i[0] == 0 and line.strip() != "## mda2ascii 1.2 generated output"
                ):  # bug out oif we don't like the header
                    raise Core.StonerLoadError("Not a file mda2ascii")
                line.strip()
                if "=" in line:
                    parts = line[2:].split("=")
                    self[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
                elif line.startswith("#  Extra PV:"):
                    # Onto the next metadata bit
                    break
            pvpat = re.compile(r"^#\s+Extra\s+PV\s\d+\:(.*)")
            for i[1], line in enumerate(data):
                if line.strip() == "":
                    continue
                elif line.startswith("# Extra PV"):
                    res = pvpat.match(line)
                    bits = [b.strip().strip(r'"') for b in res.group(1).split(",")]
                    if bits[1] == "":
                        key = bits[0]
                    else:
                        key = bits[1]
                    if len(bits) > 3:
                        key = key + " ({})".format(bits[3])
                    self[key] = string_to_type(bits[2])
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
                    self[parts[0].strip()] = string_to_type("".join(parts[1:]).strip())
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
                            colname += " ({})".format(bits[-1])
                        if colname in column_headers:
                            colname = "{}:{}".format(bits[0], colname)
                    else:
                        colname = res.group(1).strip()
                    column_headers.append(colname)
            else:
                raise Core.StonerLoadError("Overand the end of file without reading data")
        self.data = np.genfromtxt(self.filename, skip_header=sum(i))  # so that's ok then !
        self.column_headers = column_headers
        return self


class OpenGDAFile(Core.DataFile):

    """Extends Core.DataFile to load files from RASOR"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16  # Makes a positive ID of it's file type so give priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Load an OpenGDA file.

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
        i = 0
        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line != "&SRS":
                    raise Core.StonerLoadError("Not a GDA File from Rasor ?" + str(line))
                if "&END" in line:
                    break
                parts = line.split("=")
                if len(parts) != 2:
                    continue
                key = parts[0]
                value = parts[1].strip()
                self.metadata[key] = string_to_type(value)
            if python_v3:
                column_headers = f.readline().strip().split("\t")
            else:
                column_headers = f.next().strip().split("\t")
            self.data = np.genfromtxt([str2bytes(l) for l in f], dtype="float", invalid_raise=False)
        self.column_headers = column_headers
        return self


class RasorFile(OpenGDAFile):

    """Just an alias for OpenGDAFile"""

    pass


class SNSFile(Core.DataFile):

    """Reads the ASCII exported Poalrised Neutron Rfeflectivity reduced files from BL-4A line at the Spallation Neutron Source at Oak Ridge National Lab.

    File has a large header marked up with # prefixes which include several section is []
    Each section seems to have a slightly different format
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns = ["*.dat"]  # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename

        with io.open(self.filename, "r", errors="ignore", encoding="utf-8") as data:  # Slightly ugly text handling
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
                        if not python_v3:
                            column_headers = [h.strip().encode("ascii", errors="replace") for h in header]
                        else:
                            column_headers = [h.strip() for h in header]
                        self.data = np.genfromtxt(data)  # we end by reading the raw data
                    elif section == "Global Options":  # This section can go into metadata
                        for line in data:
                            line = line[2:].strip()
                            if line.strip() == "":
                                break
                            else:
                                self[line[2:10].strip()] = line[11:].strip()
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
                        self[section] = sec
                else:  # We must still be in the opening un-labelled section of meta data
                    if ":" in line:
                        i = line.index(":")
                        key = line[:i].strip()
                        value = line[i + 1 :].strip()
                        self[key.strip()] = value.strip()
        self.column_headers = column_headers
        return self
