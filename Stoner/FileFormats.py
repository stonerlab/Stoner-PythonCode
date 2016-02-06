"""Stoner.FileFormats is a module within the Stoner package that provides extra classes
that can load data from various instruments into DataFile type objects.

Eacg class has a priority attribute that is used to determine the order in which
they are tried by DataFile and friends where trying to load data. High priority
is run last.

Eacg class should implement a load() method and optionally a save() method.
"""
from __future__ import print_function
from Stoner.compat import *
import linecache
import re
import numpy as _np_
import fileinput
import csv
import os
import struct
from re import split
from datetime import datetime
import numpy.ma as ma

from .Core import DataFile, StonerLoadError
from .pyTDMS import read as tdms_read


class CSVFile(DataFile):
    """A subclass of DataFiule for loading generic deliminated text fiules without metadata."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=128 # Rather generic file format so make it a low priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.csv","*.txt"] # Recognised filename patterns

    def _load(self, filename=None, header_line=0, data_line=1, data_delim=',', header_delim=',', **kargs):
        """Generic deliminated file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Keyword Arguments:
            header_line (int): The line in the file that contains the column headers.
                If None, then column headers are auotmatically generated.
            data_line (int): The line on which the data starts
            data_delim (string): Thge delimiter used for separating data values
            header_delim (strong): The delimiter used for separating header values

        Returns:
            A copy of the current object after loading the data.
                """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        if header_line is not None:
            header_string = linecache.getline(self.filename, header_line+1)
            header_string = re.sub(r'["\n]', '', header_string)
            try:
                tmp = header_string.index(header_delim)
            except ValueError:
                raise StonerLoadError("No Delimiters in header line")
            column_headers = [x.strip() for x in header_string.split(header_delim)]
        else:
            column_headers = ["Column" + str(x) for x in range(_np_.shape(self.data)[1])]
            data_line = linecache.getline(self.filename, data_line)
            try:
                data_line.index(data_delim)
            except ValueError:
                raise StonerLoadError("No delimiters in data lines")

        self.data = _np_.genfromtxt(self.filename, dtype='float', delimiter=data_delim, skip_header=data_line)
        self.column_headers=column_headers
        return self

    def save(self, filename, deliminator=','):
        """Overrides the save method to allow CSVFiles to be written out to disc (as a mininmalist output)

        Args:
            filename (string): Fielname to save as (using the same rules as for the load routines)

        Keyword Arguments:
            deliminator (string): Record deliniminator (defaults to a comma)

        Returns:
            A copy of itself."""
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog('w')
        spamWriter = csv.writer(open(filename, 'wb'), delimiter=deliminator, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        spamWriter.writerow(self.column_headers)
        while i < self.data.shape[0]:
            spamWriter.writerow(self.data[i,:])
            i += 1
        return self


class VSMFile(DataFile):
    """Extends DataFile to open VSM Files"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Now makes a positive ID of its contents
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.fld"] # Recognised filename patterns


    def __parse_VSM(self, header_line=3, data_line=7, data_delim=' ', header_delim=','):
        """An intrernal function for parsing deliminated data without a leading column of metadata.copy

        Keyword Arguments:
            header_line (int): The line in the file that contains the column headers.
                If None, then column headers are auotmatically generated.
            data_line (int): The line on which the data starts
            data_delim (string): Thge delimiter used for separating data values
            header_delim (strong): The delimiter used for separating header values

        Returns:
            Nothing, but modifies the current object.

        Note:
            The default values are configured fir read VSM data files
        """
        try:
            with open(self.filename) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        self["Timestamp"] = line.strip()
                        check = datetime.strptime(self["Timestamp"], "%a %b %d %H:%M:%S %Y")
                        if check is None:
                            raise StonerLoadError("Not a VSM file ?")
                    elif i == 1:
                        assert line.strip() == ""
                    elif i == 2:
                        header_string = line.strip()
                    elif i == 3:
                        unit_string = line.strip()
                        column_headers = [
                            "{} ({})".format(h.strip(), u.strip())
                            for h, u in zip(header_string.split(header_delim), unit_string.split(header_delim))
                        ]
                    elif i > 3:
                        break
        except (ValueError, AssertionError, TypeError) as e:
            raise StonerLoadError('Not a VSM File' + str(e.args))
        self.data = _np_.genfromtxt(self.filename,
                                    dtype='float',
                                    usemask=True,
                                    skip_header=data_line - 1,
                                    missing_values=['6:0', '---'],
                                    invalid_raise=False)

        self.data = ma.mask_rows(self.data)
        cols = self.data.shape[1]
        self.data = _np_.reshape(self.data.compressed(), (-1, cols))
        self.column_headers=column_headers
        self.setas(x="H_vsm (T)",y="m (emu)")

    def _load(self, filename=None, *args, **kargs):
        """VSM file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        self.__parse_VSM()
        return self


class BigBlueFile(CSVFile):
    """Extends CSVFile to load files from BigBlue"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=64 # Also rather generic file format so make a lower priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat","*.iv","*.rvt"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """Just call the parent class but with the right parameters set

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """

        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        super(BigBlueFile, self)._load(self.filename, header_line=3, data_line=7, data_delim=' ', header_delim=',')
        if _np_.all(_np_.isnan(self.data)):
            raise StonerLoadError("All data was NaN in Big Blue format")
        return self


class QDSquidVSMFile(DataFile):
    """Extends DataFile to load files from The SQUID VSM"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Is able to make a positive ID of its file content, so get priority to check
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """QDSquidVSM file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(self.filename, "r") as f:  # Read filename linewise
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line != "[Header]":
                    raise StonerLoadError("Not a Quantum Design File !")
                if i == 2 and "Quantum Design" not in line:
                    raise StonerLoadError("Not a Quantum Design File !")
                elif "[Data]" in line:
                    break
                elif i < 2:
                    continue
                if line[0] == ";":
                    continue
                parts = line.split(',')
                if parts[0] == "INFO":
                    key = parts[0] + parts[2]
                    key = key.title()
                    value = parts[1]
                elif parts[0] in ['BYAPP', 'FILEOPENTIME']:
                    key = parts[0].title()
                    value = ' '.join(parts[1:])
                else:
                    key = parts[0] + "." + parts[1]
                    key = key.title()
                    value = ' '.join(parts[2:])
                self.metadata[key] = self.metadata.string_to_type(value)
            if python_v3:
                column_headers = f.readline().strip().split(',')
            else:
                column_headers = f.next().strip().split(',')
        self.data = _np_.genfromtxt(self.filename, dtype='float', delimiter=',', invalid_raise=False, skip_header=i + 2)
        self.column_headers=column_headers
        self.setas(x="Magnetic Field", y="Moment")
        return self


class OpenGDAFile(DataFile):
    """Extends DataFile to load files from RASOR"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Makes a positive ID of it's file type so give priority
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """OpenGDA file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(self.filename, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line != "&SRS":
                    raise StonerLoadError("Not a GDA File from Rasor ?" + str(line))
                if "&END" in line:
                    break
                parts = line.split('=')
                if len(parts) != 2:
                    continue
                key = parts[0]
                value = parts[1].strip()
                self.metadata[key] = self.metadata.string_to_type(value)
            if python_v3:
                column_headers = f.readline().strip().split("\t")
            else:
                column_headers = f.next().strip().split("\t")
        self.data = _np_.genfromtxt(self.filename, dtype='float', invalid_raise=False, skip_header=i + 2)
        self.column_headers=column_headers
        return self


class RasorFile(OpenGDAFile):
    """Just an alias for OpenGDAFile"""
    pass


class SPCFile(DataFile):
    """Extends DataFile to load SPC files from Raman"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Can't make a positive ID of itself
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.spc"] # Recognised filename patterns

    mime_type=["application/octet-stream"]

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
            Metadata keys are pretty much as specified in the spc.h file that defines the filerformat."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        filesize = os.stat(self.filename).st_size
        with open(filename, 'rb') as f:
            spchdr = struct.unpack(b'BBBciddiBBBBi9s9sH8f30s130siiBBHf48sfifB187s', f.read(512))
            keys = ("ftflgs", "fversn", "fexper", "fexp", "fnpts", "ffirst", "flast", "fnsub", "fxtype", "fytype",
                    "fztype", "fpost", "fres", "fsource", "fpeakpt", "fspare1", "fspare2", "fspare3", "fspare4",
                    "fspare5", "fspare6", "fspare7", "fspare8", "fcm", "nt", "fcatx", "flogoff", "fmods", "fprocs",
                    "flevel", "fsampin", "ffactor", "fmethod", "fzinc", "fwplanes", "fwinc", "fwtype", "fwtype",
                    "fresv")
            header = dict(zip(keys, spchdr))

            if header['ftflgs'] &64 == 64 or not (75 <= header['fversn'] <=
                                             77):  # This is the multiple XY curves in file flag.
                raise StonerLoadError("Filetype not implemented yet ! ftflgs={ftflgs}, fversn={fversn}".format(**header))
            else:  # A single XY curve in the file.
                n = header['fnsub']
                pts = header['fnpts']
                if header['ftflgs'] & 128:  # We need to read some X Data
                    if 4 * pts > filesize - f.tell():
                        raise StonerLoadError("Trying to read too much data!")
                    xvals = f.read(4 * pts)  # I think storing X vals directly implies that each one is 4 bytes....
                    xdata = _np_.array(struct.unpack(str2bytes(str(pts) + "f"), xvals))
                else:  # Generate the X Data ourselves
                    first = header['ffirst']
                    last = header['flast']
                    if pts > 1E6:  # Something not right here !
                        raise StonerLoadError("More than 1 million points requested. Bugging out now!")
                    xdata = _np_.linspace(first, last, pts)
                data = _np_.zeros((pts, (n + 1)))  # initialise the data soace
                data[:, 0] = xdata  # Put in the X-Data
                xvars = ["Arbitrary", "Wavenumber (cm-1)", "Micrometers (um)", "Nanometers (nm)", "Seconds", "Minutes",
                         "Hertz (Hz)", "Kilohertz (KHz)", "Megahertz (MHz)", "Mass (M/z)", "Parts per million (PPM)",
                         "Days", "Years", "Raman Shift (cm-1)", "Raman Shift (cm-1)", "eV",
                         "XYZ text labels in fcatxt (old 0x4D version only)", "Diode Number", "Channel", "Degrees",
                         "Temperature (F)", "Temperature (C)", "Temperature (K)", "Data Points", "Milliseconds (mSec)",
                         "Microseconds (uSec)", "Nanoseconds (nSec)", "Gigahertz (GHz)", "Centimeters (cm)",
                         "Meters (m)", "Millimeters (mm)", "Hours", "Hours"]
                yvars = ["Arbitrary Intensity", "Interferogram", "Absorbance", "Kubelka-Monk", "Counts", "Volts",
                         "Degrees", "Milliamps", "Millimeters", "Millivolts", "Log(1/R)", "Percent", "Percent",
                         "Intensity", "Relative Intensity", "Energy", "Decibel", "Temperature (F)", "Temperature (C)",
                         "Temperature (K)", "Index of Refraction [N]", "Extinction Coeff. [K]", "Real", "Imaginary",
                         "Complex", "Complex", "Transmission (ALL HIGHER MUST HAVE VALLEYS!)", "Reflectance",
                         "Arbitrary or Single Beam with Valley Peaks", "Emission", "Emission"]
                column_headers = [xvars[header['fxtype']]]  # And label the X column correctly

                #Now we're going to read the Y-data
                # Start by preping some vars for use

                subhdr_keys = ("subflgs", "subexp", "subindx", "subtime", "subnext", "subnois", "subnpts", "subscan",
                               "subwlevel", "subresv")
                if header['ftflgs'] & 1:
                    y_width = 2
                    y_fmt = 'h'
                    divisor = 2 ** 16
                else:
                    y_width = 4
                    y_fmt = 'i'
                    divisor = 2 ** 32
                if n * (y_width * pts + 32) > filesize - f.tell():
                    raise StonerLoadError("No good, going to read too much data!")
                for j in range(n):  # We have n sub-scans
                    # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
                    subhdr = struct.unpack(b'BBHfffIIf4s', f.read(32))
                    subheader = dict(zip(["scan" + str(j) + ":" + x for x in subhdr_keys], subhdr))

                    # Now read the y-data
                    exponent = subheader["scan" + str(j) + ':subexp']
                    if int(exponent) & -128:  # Data is unscaled direct floats
                        ydata = _np_.array(struct.unpack(str2bytes(str(pts) + "f"), f.read(pts * y_width)))
                    else:  # Data is scaled by exponent
                        yvals = struct.unpack(str2bytes(str(pts) + y_fmt), f.read(pts * y_width))
                        ydata = _np_.array(yvals, dtype='float64') * (2 ** exponent) / divisor

    # Pop the y-data into the array and merge the matadata in too.
                    data[:, j + 1] = ydata
                    header = dict(header, **subheader)
                    column_headers.append("Scan" + str(j) + ":" + yvars[header['fytype']])

    # Now we're going to read any log information
                if header['flogoff'] != 0:  # Ok, we've got a log, so read the log header and merge into metadata
                    logstc = struct.unpack(b'IIIII44s', f.read(64))
                    logstc_keys = ("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
                    logheader = dict(zip(logstc_keys, logstc))
                    header = dict(header, **logheader)

                    # Can't handle either binary log information or ion disk log information (wtf is this anyway !)
                    if header['logbins'] + header['logdsks'] > filesize - f.tell():
                        raise StonerLoadError("Too much logfile data to read")
                    f.read(header['logbins'] + header['logdsks'])

                    # The renishaw seems to put a 16 character timestamp next - it's not in the spec but never mind that.
                    header['Date-Time'] = f.read(16)
                    # Now read the rest of the file as log text
                    logtext = f.read()
                    # We expect things to be single lines terminated with a CR-LF of the format key=value
                    for line in split(b"[\r\n]+", logtext):
                        if b"=" in line:
                            parts = line.split(b'=')
                            key = parts[0].decode()
                            value = parts[1].decode()
                            header[key] = value
            # Ok now build the Stoner.DataFile instance to return
            self.data = data
            # The next bit generates the metadata. We don't just copy the metadata because we need to figure out the typehints first - hence the loop here to call DataFile.__setitem()
            for x in header:
                self[x] = header[x]
            self.column_headers = column_headers
            if len(self.column_headers) == 2:
                self.setas = "xy"
            return self


class TDMSFile(DataFile):
    """A first stab at writing a file that will import TDMS files"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Makes a positive ID of its file contents
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.tdms"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """TDMS file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        f = open(self.filename, "rb")  # Read filename linewise
        try:
            assert f.read(4) == b"TDSm"
        except AssertionError:
            f.close()
            raise StonerLoadError('Not a TDMS File')
        f.close()
        (metadata, data) = tdms_read(self.filename)
        for key in metadata:
            self.metadata[key] = metadata[key]
        column_headers = list()
        for column in data:
            nd = data[column]
            self.add_column(nd, header=column)
        return self


class RigakuFile(DataFile):
    """Loads a .ras file as produced by Rigaku X-ray diffractormeters"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 #Can make a positive id of file from first line
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.ras"] # Recognised filename patterns


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
            self.get_filename('rb')
        else:
            self.filename = filename
        sh = re.compile(r'^\*([^\s]+)\s+(.*)$')  # Regexp to grab the keys
        ka = re.compile(r'(.*)\-(\d+)$')
        header = dict()
        with open(self.filename, "rb") as f:
            for i, line in enumerate(f):
                line = bytes2str(line).strip()
                if i == 0 and line != "*RAS_DATA_START":
                    raise StonerLoadError("Not a Rigaku file!")
                if line == "*RAS_HEADER_START":
                    break
            for i2, line in enumerate(f):
                line = bytes2str(line).strip()
                m = sh.match(line)
                if m:
                    key = m.groups()[0].lower().replace('_', '.')
                    try:
                        value = m.groups()[1].decode('utf-8', 'ignore')
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
                    if key in self.metadata and not (isinstance(self[key], _np_.ndarray) or isinstance(self[key], list)):
                        if isinstance(self[key], str):
                            self[key] = list([self[key]])
                        else:
                            self[key] = _np_.array(self[key])
                    if key not in self.metadata:
                        if isinstance(newvalue, str):
                            self[key] = list([newvalue])
                        else:
                            self[key] = _np_.array([newvalue])
                    else:
                        if isinstance(self[key][0], str):
                            self[key].append(newvalue)
                        else:
                            self[key] = _np_.append(self[key], newvalue)
                else:
                    self.metadata[key] = newvalue

        self.data = _np_.genfromtxt(self.filename,
                                    dtype='float',
                                    delimiter=' ',
                                    invalid_raise=False,
                                    comments="*",
                                    skip_header=i + i2 + 1)
        column_headers = ['Column' + str(i) for i in range(self.data.shape[1])]
        column_headers[0:2] = [self.metadata['meas.scan.unit.x'], self.metadata['meas.scan.unit.y']]
        for key in self.metadata:
            if isinstance(self[key], list):
                self[key] = _np_.array(self[key])
        self.setas = "xy"
        self.column_headers=column_headers
        return self

    def to_Q(self, l=1.540593):
        """Adds an additional function to covert an angualr scale to momentum transfer

        returns a copy of itself."""

        self.add_column((4 * _np_.pi / l) * _np_.sin(_np_.pi * self.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)")


class XRDFile(DataFile):
    """Loads Files from a Brucker D8 Discovery X-Ray Diffractometer"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Makes a positive id of its file contents
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dql"] # Recognised filename patterns

    def __getattr__(self,name):
        ret=super(XRDFile,self).__getattr__(name)
        if name=="_public_attrs":
            ret.update({"four_bounce":bool})
        return ret

    def _load(self,filename=None,*args, **kargs):
        """Reads an XRD datafile as produced by the Brucker diffractometer

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
            self.get_filename('r')
        else:
            self.filename = filename
        sh = re.compile(r'\[(.+)\]')  # Regexp to grab section name
        f = fileinput.FileInput(self.filename)  # Read filename linewise
        if f.readline().strip() != ";RAW4.00":  # Check we have the corrrect fileformat
            raise StonerLoadError("File Format Not Recognized !")
        drive = 0
        for line in f:  #for each line
            m = sh.search(line)
            if m:  # This is a new section
                section = m.group(1)
                if section == "Drive":  #If this is a Drive section we need to know which Drive Section it is
                    section = section + str(drive)
                    drive = drive + 1
                elif section == "Data":  # Data section contains the business but has a redundant first line
                    if python_v3:
                        f.readline()
                    else:
                        f.next()
                for line in f:  #Now start reading lines in this section...
                    if line.strip(
                    ) == "":  # A blank line marks the end of the section, so go back to the outer loop which will handle a new section
                        break
                    elif section=="Data": # In the Data section read lines of data value,vale
                        parts=line.split(',')
                        angle=parts[0].strip()
                        counts=parts[1].strip()
                        dataline=_np_.array([float(angle), float(counts)])
                        self.data=_np_.append(self.data, dataline)
                    else: # Other sections contain metadata
                        parts=line.split('=')
                        key=parts[0].strip()
                        data=parts[1].strip()
                        self[section+":"+key]=self.metadata.string_to_type(data) # Keynames in main metadata are section:key - use theDataFile magic to do type determination
        column_headers=['Angle', 'Counts'] # Assume the columns were Angles and Counts

        f.close()# Cleanup
        self.data=_np_.reshape(self.data, (-1, 2))
        self.setas="xy"
        self.four_bounce=self["HardwareConfiguration:Monochromator"]==1
        self.column_headers=column_headers
        return self

    def to_Q(self, l=1.540593):
        """Adds an additional function to covert an angualr scale to momentum transfer

        returns a copy of itself."""

        self.add_column((4 * _np_.pi / l) * _np_.sin(_np_.pi * self.column(0) / 360), header="Momentum Transfer, Q ($\\AA$)")


class BNLFile(DataFile):
    """
    Creates BNLFile a subclass of DataFile that caters for files in the SPEC format given
    by BNL (specifically u4b beamline but hopefully generalisable).

    Author Rowan 12/2011

    The file from BNL must be split into seperate scan files before Stoner can use
    them, a separate python script has been written for this and should be found
    in data/Python/PythonCode/scripts.
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=64
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.txt"] # Recognised filename patterns

    def __init__(self, *params):
        """Constructor modification
        BNLFile('filename')
        Do a normal initiation using the parent class 'self' followed by adding an extra attribute line_numbers,
        line_numbers is a list of important line numbers in the file.
        I've left it open for someone to add options for more args if they wish."""
        super(BNLFile, self).__init__(*params)
        self.line_numbers = []

    def __find_lines(self):
        """returns an array of ints [header_line,data_line,scan_line,date_line,motor_line]"""
        fp = open(self.filename, 'r')
        self.line_numbers = [0, 0, 0, 0, 0]
        counter = 0
        for line in fp:
            counter += 1
            if counter == 1 and line[0] != '#':
                raise StonerLoadError("Not a BNL File ?")
            if len(line) < 2: continue  #if there's nothing written on the line go to the next
            elif line[0:2] == '#L': self.line_numbers[0] = counter
            elif line[0:2] == '#S': self.line_numbers[2] = counter
            elif line[0:2] == '#D': self.line_numbers[3] = counter
            elif line[0:2] == '#P': self.line_numbers[4] = counter
            elif line[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                self.line_numbers[1] = counter
                break

    def __get_metadata(self):
        """Metadata found is scan number 'Snumber', scan type and parameters 'Stype',
        scan date/time 'Sdatetime' and z motor position 'Smotor'."""
        scanLine = linecache.getline(self.filename, self.line_numbers[2])
        dateLine = linecache.getline(self.filename, self.line_numbers[3])
        motorLine = linecache.getline(self.filename, self.line_numbers[4])
        self.__setitem__('Snumber', scanLine.split()[1])
        tmp = "".join(scanLine.split()[2:])
        self.__setitem__('Stype', "".join(tmp.split(',')))  #get rid of commas
        self.__setitem__('Sdatetime', dateLine[3:-1])  #don't want \n at end of line so use -1
        self.__setitem__('Smotor', motorLine.split()[3])

    def __parse_BNL_data(self):
        """
        Internal function for parsing BNL data. The meta data is labelled by #L type tags
        so easy to find but #L must be excluded from the result.
        """
        self.__find_lines()
        """creates a list, line_numbers, formatted [header_line,data_line,scan_line,date_line,motor_line]"""
        header_string = linecache.getline(self.filename, self.line_numbers[0])
        header_string = re.sub(r'["\n]', '', header_string)  #get rid of new line character
        header_string = re.sub(r'#L', '', header_string)  #get rid of line indicator character
        column_headers = map(lambda x: x.strip(), header_string.split())
        self.__get_metadata()
        try:
            self.data = _np_.genfromtxt(self.filename, skip_header=self.line_numbers[1] - 1)
        except IOError:
            self.data = _np_.array([0])
            print('Did not import any data for {}'.format(self.filename))
        self.column_headers=column_headers

    def _load(self, filename, *args, **kargs):  #fileType omitted, implicit in class call
        """BNLFile.load(filename)

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.

        Notes:
            Overwrites load method in DataFile class, no header positions and data
            positions are needed because of the hash title structure used in BNL files.

            Normally its good to use _parse_plain_data method from DataFile class
            to load data but unfortunately Brookhaven data isn't very plain so there's
            a new method below.
        """
        self.filename = filename
        self.__parse_BNL_data()  #call an internal function rather than put it in load function
        return self


class MokeFile(DataFile):
    """Class that extgends DataFile to load files from the Leeds MOKE system."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priotity=16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat","*.txt"]


    def _load(self, filename=None, *args, **kargs):
        """Leeds  MOKE file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(self.filename, mode="rb") as f:
            line = bytes2str(f.readline()).strip()
            if line != "#Leeds CM Physics MOKE":
                raise StonerLoadError("Not a datafile from the Leeds MOKE")
            while line.startswith("#") or line == "":
                parts = line.split(":")
                if len(parts) > 1:
                    key = parts[0][1:]
                    data = ":".join(parts[1:]).strip()
                    self[key] = data
                line = bytes2str(f.readline()).strip()
            column_headers = [x.strip() for x in line.split(",")]
            self.data = _np_.genfromtxt(f, delimiter=",")
        self.setas = "xy.de"
        self.column_headers=column_headers
        return self


class FmokeFile(DataFile):
    """Extends DataFile to open Fmoke Files"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16 # Makes a positive ID check of its contents so give it priority in autoloading
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Sheffield Focussed MOKE file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        f = fileinput.FileInput(self.filename, mode="rb")  # Read filename linewise
        try:
            value = [float(x.strip()) for x in bytes2str(f.readline()).split('\t')]
        except:
            raise StonerLoadError("Not an FMOKE file?")
        label = [x.strip() for x in bytes2str(f.readline()).split('\t')]
        if label[0] != "Header:":
            raise StonerLoadError("Not a Focussed MOKE file !")
        del (label[0])
        for k, v in zip(label, value):
            self.metadata[k] = v  # Create metatdata from first 2 lines
        column_headers = [x.strip() for x in bytes2str(f.readline()).split('\t')]
        self.data = _np_.genfromtxt(f, dtype='float', delimiter='\t', invalid_raise=False)
        self.column_headers=column_headers
        return self


class GenXFile(DataFile):
    """Extends DataFile for GenX Exported data."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=64
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        pattern = re.compile(r'# Dataset "([^\"]*)" exported from GenX on (.*)$')
        pattern2 = re.compile(r"#\sFile\sexported\sfrom\sGenX\'s\sReflectivity\splugin")
        with open(self.filename, "r") as datafile:
            line = datafile.readline()
            match = pattern.match(line)
            match2 = pattern2.match(line)
            if match is not None:
                dataset = match.groups()[0]
                date = match.groups()[1]
                line = datafile.readline()
                line = datafile.readline()
                line = line[1:]
                self["date"] = date
            elif match2 is not None:
                line = datafile.readline()
                self["date"] = line.split(':')[1].strip()
                datafile.readline()
                line = datafile.readline()
                line = line[1:]
                dataset = "asymmetry"
            else:
                raise StonerLoadError("Not a GenXFile")
            column_headers = [f.strip() for f in line.strip().split('\t')]
            self.data = _np_.genfromtxt(datafile)
            self["dataset"] = dataset
            self.setas = "xye"
            self.column_headers=column_headers
        return self


class SNSFile(DataFile):
    """This reads the ASCII exported Poalrised Neutron Rfeflectivity reduced files from
    BL-4A line at the Spallation Neutron Source at Oak Ridge National Lab.

    File has a large header marked up with # prefixes which include several section is []
    Each section seems to have a slightly different format

    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns

    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        with open(self.filename, "r") as data:  # Slightly ugly text handling
            line = data.readline()
            if line.strip() != "# Datafile created by QuickNXS 0.9.39":  # bug out oif we don't like the header
                raise StonerLoadError("Not a file from the SNS BL4A line")
            for line in data:
                if line.startswith("# "):  # We're in the header
                    line = line[2:].strip()  # strip the header and whitespace

                if line.startswith("["):  # Look for a section header
                    section = line.strip().strip("[]")
                    if section == "Data":  # The Data section has one line of colum headers and then data
                        header = next(data)[2:].split("\t")
                        column_headers = [h.strip().decode('ascii', 'ignore') for h in header]
                        self.data = _np_.genfromtxt(data)  # we end by reading the raw data
                    elif section == "Global Options":  # This section can go into metadata
                        for line in data:
                            line = line[2:].strip()
                            if line.strip() == "":
                                break
                            else:
                                self[line[2:10].strip()] = line[11:].strip()
                    elif section == "Direct Beam Runs" or section == "Data Runs":  # These are constructed into lists ofg dictionaries for each file
                        sec = list()
                        header = next(data)
                        header = header[2:].strip()
                        keys = [s.strip() for s in header.split('  ') if s.strip()]
                        for line in data:
                            line = line[2:].strip()
                            if line == "":
                                break
                            else:
                                values = [s.strip() for s in line.split('  ') if s.strip()]
                                sec.append(dict(zip(keys, values)))
                        self[section] = sec
                else:  # We must still be in the opening un-labelled section of meta data
                    if ":" in line:
                        i = line.index(":")
                        key = line[:i].strip()
                        value = line[i + 1:].strip()
                        self[key.strip()] = value.strip()
        self.column_headers=column_headers
        return self


class OVFFile(DataFile):
    """A class that reads OOMMF vector format files and constructs x,y,z,u,v,w data.

    OVF 1 and OVF 2 files with text or binary data and only files with a meshtype rectangular are supported"""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.ovf"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        ptr = 0
        with open(self.filename, "r") as data:  # Slightly ugly text handling
            line = next(data)
            ptr += len(line)
            line = line.strip()
            if "OOMMF: rectangular mesh" in line:
                if "v1.0" in line:
                    self["version"] = 1
                elif "v2.0" in line:
                    self["version"] = 2
                else:
                    raise StonerLoadError("Cannot determine version of OOMMFF file")
            else:  # bug out oif we don't like the header
                raise StonerLoadError("Not n OOMMF OVF File: opening line eas {}".format(line))
            pattern = re.compile(r"#\s*([^\:]+)\:\s+(.*)$")
            for i, line in enumerate(data):
                ptr += len(line)
                line.strip()
                if line.startswith("# Begin: Data"):  # marks the start of the trext
                    break
                elif line.startswith("# Begin:") or line.startswith("# End:"):
                    continue
                else:
                    res = pattern.match(line)
                    if res is not None:
                        key = res.group(1)
                        val = res.group(2)
                        self[key] = self.metadata.string_to_type(val)
                    else:
                        raise StonerLoadError("Failed to understand metadata")
            fmt = re.match(r".*Data\s+(.*)", line).group(1).strip()
            assert self["meshtype"] == "rectangular", "Sorry only OVF files with rectnagular meshes are currently supported."
            if self["version"] == 1:
                if self["meshtype"] == "rectangular":
                    self["valuedim"] = 3
                else:
                    self["valuedim"] = 6
        if fmt == "Text":
            uvwdata = _np_.genfromtxt(self.filename, skip_header=i + 2)
        elif fmt == "Binary 4":
            if self["version"] == 1:
                dt = _np_.dtype('>f4')
            else:
                dt = _np_.dtype('<f4')
            with open(filename, "rb") as bindata:
                bindata.seek(ptr)
                uvwdata = _np_.fromfile(bindata,
                                        dtype=dt,
                                        count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"])
                assert uvwdata[0] == 1234567.0, "Binary 4 format check value incorrect ! Actual Value was {}".format(
                    uvwdata[0])
            uvwdata = uvwdata[1:]
            uvwdata = _np_.reshape(uvwdata, (-1, self["valuedim"]))
        elif fmt == "Binary 8":
            if self["version"] == 1:
                dt = _np_.dtype('>f8')
            else:
                dt = _np_.dtype('<f8')
            with open(filename, "rb") as bindata:
                bindata.seek(ptr)
                uvwdata = _np_.fromfile(bindata,
                                        dtype=dt,
                                        count=1 + self["xnodes"] * self["ynodes"] * self["znodes"] * self["valuedim"])
                assert uvwdata[
                    0
                ] == 123456789012345.0, "Binary 4 format check value incorrect ! Actual Value was {}".format(uvwdata[0])
            uvwdata = _np_.reshape(uvwdata, (-1, self["valuedim"]))
        else:
            raise StonerLoadError("Unknow OVF Format {}".format(fmt))

        x = (_np_.linspace(self["xmin"], self["xmax"], self["xnode"] + 1)[:-1] + self["xbase"]) * 1E9
        y = (_np_.linspace(self["ymin"], self["ymax"], self["ynode"] + 1)[:-1] + self["ybase"]) * 1E9
        z = (_np_.linspace(self["zmin"], self["zmax"], self["znode"] + 1)[:-1] + self["zbase"]) * 1E9
        (y, z, x) = (_np_.ravel(i) for i in _np_.meshgrid(y, z, x))
        self.data = _np_.column_stack((x, y, z, uvwdata))
        column_headers = ["X (nm)", "Y (nm)", "Z (nm)", "U", "V", "W"]
        self.setas = "xyzuvw"
        self.column_headers=column_headers
        return self


class MDAASCIIFile(DataFile):
    """Reads files generated from the APS."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.txt"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        with open(self.filename, "r") as data:  # Slightly ugly text handling
            for i1, line in enumerate(data):
                if i1 == 0 and line.strip() != "## mda2ascii 1.2 generated output":  # bug out oif we don't like the header
                    raise StonerLoadError("Not a file mda2ascii")
                line.strip()
                if "=" in line:
                    parts = line[2:].split("=")
                    self[parts[0].strip()] = self.metadata.string_to_type("".join(parts[1:]).strip())
                elif line.startswith("#  Extra PV:"):
                    # Onto the next metadata bit
                    break
            pvpat = re.compile(r'^#\s+Extra\s+PV\s\d+\:(.*)')
            for i2, line in enumerate(data):
                if line.strip() == "":
                    continue
                elif line.startswith("# Extra PV"):
                    res = pvpat.match(line)
                    bits = [b.strip().strip(r'"') for b in res.group(1).split(',')]
                    if bits[1] == "":
                        key = bits[0]
                    else:
                        key = bits[1]
                    if len(bits) > 3:
                        key = key + " ({})".format(bits[3])
                    self[key] = self.metadata.string_to_type(bits[2])
                else:
                    break  # End of Extra PV stuff
            else:
                raise StonerLoadError("Overran Extra PV Block")
            for i3, line in enumerate(data):
                line.strip()
                if line.strip() == "":
                    continue
                elif line.startswith("# Column Descriptions:"):
                    break  # Start of column headers now
                elif "=" in line:
                    parts = line[2:].split("=")
                    self[parts[0].strip()] = self.metadata.string_to_type("".join(parts[1:]).strip())
            else:
                raise StonerLoadError("Overran end of scan header before column descriptions")
            colpat = re.compile(r"#\s+\d+\s+\[([^\]]*)\](.*)")
            column_headers = []
            for i4, line in enumerate(data):
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
                raise StonerLoadError("Overand the end of file without reading data")
        self.data = _np_.genfromtxt(self.filename, skip_header=i1 + i2 + i3 + i4)  # so that's ok then !
        self.column_headers=column_headers
        return self


class LSTemperatureFile(DataFile):
    """A class that reads and writes Lakeshore Temperature Calibration Curves.

.. warning::
    THis class works for cernox curves in Log Ohms/Kelvin and Log Ohms/Log Kelvin. It may or may not work with any
    other temperature calibration data !

    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=16
     #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.340"]

    def _load(self, filename=None, *args, **kargs):
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        with open(self.filename, "rb") as data:
            keys = []
            vals = []
            for line in data:
                line = bytes2str(line)
                if line.strip() == "":
                    break
                parts = [p.strip() for p in line.split(":")]
                if len(parts) != 2:
                    raise StonerLoadError("Header doesn't contain two parts at {}".format(line.strip()))
                else:
                    keys.append(parts[0])
                    vals.append(parts[1])
            else:
                raise StonerLoadError("Overan the end of the file")
            if keys != ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit", "Temperature coefficient",
                        "Number of Breakpoints"]:
                raise StonerLoadError("Header did not contain recognised keys.")
            for (k, v) in zip(keys, vals):
                v = v.split()[0]
                self.metadata[k] = self.metadata.string_to_type(v)
            headers = bytes2str(next(data)).strip().split()
            column_headers = headers[1:]
            dat = _np_.genfromtxt(data)
            self.data = dat[:, 1:]
        self.column_headers=column_headers
        return self

    def save(self, filename=None):
        """Overrides the save method to allow CSVFiles to be written out to disc (as a mininmalist output)

        Args:
            filename (string): Filename to save as (using the same rules as for the load routines)

        Keyword Arguments:
            deliminator (string): Record deliniminator (defaults to a comma)

        Returns:
            A copy of itself."""
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog('w')
        with open(filename, "w") as f:
            for k in ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit", "Temperature coefficient",
                      "Number of Breakpoints"]:
                if k in ["Sensor Model", "Serial Number", "Data Format", "SetPoint Limit"]:
                    kstr = "{:16s}".format(k + ":")
                else:
                    kstr = "{}:   ".format(k)
                v = self[k]
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
                f.write("{}{}\r\n".format(kstr, vstr))
            f.write("\r\n")
            f.write("No.   ")
            for h in self.column_headers:
                f.write("{:11s}".format(h))
            f.write("\r\n\r\n")
            for i in range(
                len(self.data)):  # This is a slow way to write the data, but there should only ever be 200 lines
                line = "\t".join(["{:<10.8f}".format(n) for n in self.data[i]])
                f.write("{}\t".format(i))
                f.write("{}\r\n".format(line))
        return self


class EasyPlotFile(DataFile):
    """A class that will extract as much as it can from an EasyPlot save File."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=32 # Fairly generic, but can do some explicit testing

    def _load(self,filename, *args, **kargs):
        """Private loader method."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        datastart = -1
        dataend = -1

        with open(self.filename, "r") as data:
            if "******** EasyPlot save file ********" not in data.read(1024):
                raise StonerLoadError("Not an EasyPlot Save file?")
            else:
                data.seek(0)
            for i, line in enumerate(data):
                line = line.strip()
                if line == "":
                    continue
                if line[0] not in "-0123456789" and datastart > 0 and dataend < 0:
                    dataend = i
                if line.startswith('"') and ":" in line:
                    parts = [x.strip() for x in line.strip('"').split(':')]
                    self[parts[0]] = self.metadata.string_to_type(":".join(parts[1:]))
                elif line.startswith("/"):  # command
                    parts = [x.strip('"') for x in next(csv.reader([line], delimiter=" ")) if x != ""]
                    cmd = parts[0].strip("/")
                    if len(cmd) > 1:
                        cmdname = "_{}_cmd".format(cmd)
                        if cmdname in dir(self):  #If this command is implemented as a function run it
                            self.__getattr__("_{}_cmd".format(cmd))(parts[1:])
                        else:
                            if len(parts[1:]) > 1:
                                cmd = cmd + "." + parts[1]
                                value = ",".join(parts[2:])
                            elif len(parts[1:]) == 1:
                                value = parts[1]
                            else:
                                value = True
                            self[cmd] = value
                elif line[0] in "-0123456789" and datastart < 0:  #start of data
                    datastart = i
                    if "," in line:
                        delimiter = ","
                    else:
                        delimiter = None
        if dataend < 0:
            dataend = i
        self.data = _np_.genfromtxt(self.filename, skip_header=datastart, skip_footer=i - dataend, delimiter=delimiter)
        if self.data.shape[1] == 2:
            self.setas = "xy"
        return self

    def _extend_columns(self, i):
        """Ensure the column headers are at least i long."""
        if len(self.column_headers) < i:
            l = len(self.column_headers)
            self.data=_np_.append(self.data,_np_.zeros((self.shape[0],i-l)),axis=1) # Need to expand the array first
            self.column_headers.extend(["Column {}".format(x) for x in range(l, i)])

    def _et_cmd(self, parts):
        """Handle axis labellling command."""
        if parts[0] == "x":
            self._extend_columns(1)
            self.column_headers[0] = parts[1]
        elif parts[0] == "y":
            self._extend_columns(2)
            self.column_headers[1] = parts[1]
        elif parts[0] == "g":
            self["title"] = parts[1]

    def _td_cmd(self, parts):
        self.setas = parts[0]

    def _sa_cmd(self, parts):
        """The sa (set-axis?) command."""
        if parts[0] == "l":  #Legend
            col = int(parts[2])
            self._extend_columns(col + 1)
            self.column_headers[col] = parts[1]

class PinkLibFile(DataFile):
    """Extends DataFile to load files from MdV's PINK library - as used by the GMR anneal rig."""

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority=32 # reasonably generic format
    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    patterns=["*.dat"] # Recognised filename patterns


    def _load(self, filename=None, *args, **kargs):
        """PinkLib file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(self.filename, "r") as f:  # Read filename linewise
            if "PINKlibrary" not in f.readline():
                raise StonerLoadError("Not a PINK file")
            f=f.readlines()
            happened_before=False
            for i, line in enumerate(f):
                if line[0]!='#' and not happened_before:
                    header_line=i-2 #-2 because there's a commented out data line
                    happened_before=True
                    continue #want to get the metadata at the bottom of the file too
                elif any(s in line for s in ('Start time', 'End time', 'Title')):
                    tmp=line.strip('#').split(':')
                    self.metadata[tmp[0].strip()] = ':'.join(tmp[1:]).strip()
            column_headers = f[header_line].strip('#\t ').split('\t')
            data = _np_.genfromtxt(self.filename, dtype='float', delimiter='\t', invalid_raise=False, comments='#')
            self.data=data[:,0:-2] #Deal with an errant tab at the end of each line
            if _np_.all([h in column_headers for h in ('T (C)', 'R (Ohm)')]):
                self.setas(x='T (C)', y='R (Ohm)')
        self.column_headers=column_headers
        return self
