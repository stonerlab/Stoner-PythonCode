#############################################
#
# Core object of the Stoner Package
#
# $Id: Core.py,v 1.39 2012/03/11 01:41:56 cvs Exp $
#
# $Log: Core.py,v $
# Revision 1.39  2012/03/11 01:41:56  cvs
# Recompile API help
#
# Revision 1.38  2012/03/10 20:17:15  cvs
# Minor changes
#
# Revision 1.37  2012/01/03 21:51:04  cvs
# Fix a bug with add_column
# Upload new TDMS data
#
# Revision 1.36  2012/01/03 12:41:50  cvs
# Made Core pep8 compliant
# Added TDMS code and TDMSFile
#
# Revision 1.35  2011 / 12 / 09 12:10:41  cvs
# Remove cvs writing code from DataFile (use CSVFile.save()). Fixed BNLFile to
# always call the DataFile constructor
#
# Revision 1.34  2011 / 12 / 05 21:56:25  cvs
# Add in DataFile methods swap_column and reorder_columns and update API
# documentation.
# Fix some Doxygen problems.
#
# Revision 1.33  2011 / 12 / 04 23:09:16  cvs
# Fixes to Keissig and plotting code
#
# Revision 1.32  2011 / 12 / 03 13:58:48  cvs
# Replace the various format load routines in DataFile with subclasses of
# DataFile with their own overloaded load methods
# Improve the VSM load routine
# Add some new sample data sets to play with
# Updatedocumentation
#
# Revision 1.31  2011 / 11 / 28 14:26:52  cvs
# Merge latest versions
#
# Revision 1.30  2011 / 11 / 28 09:26:33  cvs
# Update documentation
#
# Revision 1.29  2011 / 10 / 24 12:17:55  cvs
# Update PCAR lab script to save data and fix a bug with save as mode in
# Stoner.Core
#
# Revision 1.28  2011 / 10 / 10 13:22:58  cvs
# Removed a print from the sort function.
#
# Revision 1.27  2011 / 08 / 17 12:46:29  cvs
# Hack to fix the following issue:
#
# Sort of - source is string value and the code should have exported it as
# '' rather than a blank string. I'll take a look. In the meantime you
# could put the eval statement in a try except block something like:
#
# try:
#     ret = eval(....)
# except SyntaxError:
#    ret=""
#
# which should at least stop the crash and burn...
#
# Revision 1.26  2011 / 08 / 09 14:17:28  cvs
# Added option to load Horiba Raman plaintext file
#
# Revision 1.25  2011 / 07 / 12 15:53:19  cvs
# Teach typeHintedDict to handle NaN as a value, introduce an export function
# to help with the string representation of DataFile and fix a weird regression
# in DataFile.__repr__. Update doxygen docs
#
# Revision 1.24  2011 / 06 / 24 16:23:58  cvs
# Update API documentation. Minor improvement to save method to force a dialog
# box.
#
# Revision 1.23  2011 / 06 / 15 11:38:20  cvs
# Matt - Removed printing of args datatype on DataFile initialization.
#
# Revision 1.22  2011 / 06 / 14 21:50:55  cvs
# Produce a clone attribute in Core that does a deep copy and update
# the documention some more.
#
# Revision 1.21  2011 / 06 / 13 20:38:10  cvs
# Merged in fixes to typeHintedDict with fixes for deepcopy
#
# Revision 1.20  2011 / 06 / 13 20:15:13  cvs
# Make copy and deepcopy work properly
#
# Revision 1.19  2011 / 06 / 13 14:40:51  cvs
# Make theroutine handle a blank value in metadata
#
# Revision 1.18  2011 / 05 / 17 21:04:29  cvs
# Finish implementing the DataFile metadata as a new typeHintDict()
# dictionary that keeps track of the type hinting strings internally.
# This ensures that we always have a type hint string available.
#
# Revision 1.17  2011 / 05 / 16 22:43:19  cvs
# Start work on a dict child class that keeps track of type hints to use as
# the core class for the metadata. GB
#
# Revision 1.16  2011 / 05 / 10 22:10:31  cvs
# Workaround new behaviou of deepcopy() in Python 2.7 and improve
# handling when a typehint for the metadata
# doesn't exist (printing the DataFile will fix the typehinting).
#
# Revision 1.15  2011 / 05 / 06 22:21:42  cvs
# Add code to read Renishaw spc files and some sample Raman data. GB
#
# Revision 1.14  2011 / 04 / 23 18:23:33  cvs
# What happened here ?
#
# Revision 1.13  2011 / 04 / 22 14:44:04  cvs
# Add code to return data as a structured record and to to provide a
# DataFile.sort() method
#
# Revision 1.12  2011 / 03 / 02 14:56:20  cvs
# Colon missing from else command in search function (Line 591)
#
# Revision 1.11  2011 / 03 / 02 13:16:52  cvs
# Fix buglet in DataFile.search
#
# Revision 1.10  2011 / 02 / 23 21:42:16  cvs
# Experimental code for displaying grid included
#
# Revision 1.9  2011 / 02 / 17 23:36:51  cvs
# Updated doxygen comment strings
#
# Revision 1.8  2011 / 02 / 14 17:00:03  cvs
# Updated documentation. More doxygen comments
#
# Revision 1.7  2011 / 02 / 13 15:51:08  cvs
# Merge in ma gui branch back to HEAD
#
# Revision 1.6  2011 / 02 / 12 22:12:43  cvs
# Added some doxygen compatible doc strings
#
# Revision 1.5  2011 / 02 / 11 00:00:58  cvs
# Add a DataFile.unique method
#
# Revision 1.4  2011 / 01 / 17 10:12:08  cvs
# Added code for mac implementation of wx.FileDialog()
#
# Revision 1.3  2011 / 01 / 13 22:30:56  cvs
# Enable chi^2 analysi where the parameters are varied and choi^2 calculated.
# Extra comments in the ini file
# Give DataFile some file dialog boxes
#
# Revision 1.2  2011 / 01 / 12 22:56:33  cvs
# Update documentation, add support for slices in some of the DataFile methods
#
# Revision 1.1  2011 / 01 / 08 20:30:02  cvs
# Complete splitting Stoner into a package with sub - packages - Core, Analysis
# and Plot.
# Setup some imports in __init__ so that import Stoner still gets all
# the subclasses - Gavin
#
#
#############################################

# Imports

import csv
import re
import scipy
#import pdb # for debugging
import os
import sys
import numpy
import math
import copy
import linecache
import wx


class evaluatable:
    """A very simple class that is just a placeholder"""


class typeHintedDict(dict):
    """Extends a regular dict to include type hints of what
    each key contains."""
    _typehints = dict()

    __regexGetType = re.compile(r'([^\{]*)\{([^\}]*)\}')
                                    # Match the contents of the inner most{}
    __regexSignedInt = re.compile(r'^I\d+')
                                    # Matches all signed integers
    __regexUnsignedInt = re.compile(r'^U / d+')
                                    # Match unsigned integers
    __regexFloat = re.compile(r'^(Extended|Double|Single)\sFloat')
                                    # Match floating point types
    __regexBoolean = re.compile(r'^Boolean')
    __regexString = re.compile(r'^(String|Path|Enum)')
    __regexEvaluatable = re.compile(r'^(Cluster|\dD Array)')

    __types = {'Boolean': bool, 'I32': int, 'Double Float': float,
        'Cluster': dict, 'Array': numpy.ndarray, 'String': str}
    # This is the inverse of the __tests below - this gives
    # the string type for standard Python classes

    __tests = [(__regexSignedInt, int), (__regexUnsignedInt, int),
             (__regexFloat, float), (__regexBoolean, bool),
             (__regexString, str), (__regexEvaluatable, evaluatable())]
        # This is used to work out the correct python class for
        # some string types

    def __init__(self, *args, **kargs):
        """Calls the dict() constructor, then runs through the keys of the
        created dictionary and either uses the string type embedded in
        the keyname to generate the type hint (and remove the
        embedded string type from the keyname) or determines the likely
        type hint from the value of the dict element.

        @param *args Pass through all parameters to the dict() constructor.
        @param **kargs Pass through all keyword parameters to dict()
                                constructor
        @return A dictionary like object that understands the type of data
                    stored in each key."""

        parent = super(typeHintedDict, self)
        parent.__init__(*args, **kargs)
        for key in self:  # Chekc through all the keys and see if they contain
                                    # type hints. If they do, move them to the
                                    # _typehint dict
            m = self.__regexGetType.search(key)
            if m is not None:
                k = m.group(1)
                t = m.group(2)
                self._typehints[k] = t
                super(typeHintedDict, self).__setitem__(k, self[key])
                super(typeHintedDict, self).__delitem__(key)
            else:
                self._typehints[key] = self.__findtype(parent.__getitem__(key))

    def __getattr__(self, name):
        """Handles attribute access"""
        if name == "types":
            return self._typehints
        else:
            raise AttributeError

    def __findtype(self,  value):
        """Determines the correct string type to return for common python
        classes. Understands booleans, strings, integers, floats and numpy
        arrays(as arrays), and dictionaries (as clusters).

                @param value The data value to determine the type hint for.
                @return A type hint string"""
        typ = "String"
        for t in self.__types:
            if isinstance(value, self.__types[t]):
                if t == "Cluster":
                    elements = []
                    for k in  value:
                        elements.append(self.__findtype(value[k]))
                    tt = ','
                    tt = tt.join(elements)
                    typ = 'Cluster (' + tt + ')'
                elif t == 'Array':
                    z = numpy.zeros(1, dtype=value.dtype)
                    typ = (str(len(numpy.shape(value))) + "D Array (" +
                                                self.__findtype(z[0]) + ")")
                else:
                    typ = t
                break
        return typ

    def __mungevalue(self, t, value):
        """Based on a string type t, return value cast to an
        appropriate python class

        @param t is a string representing the type
        @param value is the data value to be munged into the
                    correct class
        @return Returns the munged data value

        Detail: The class has a series of precompiled regular
        expressions that will match type strings, a list of these has been
        constructed with instances of the matching Python classes. These
        are tested in turn and if the type string matches the constructor of
        the associated python class is called with value as its argument."""
        for (regexp, valuetype) in self.__tests:
            m = regexp.search(t)
            if m is not None:
                if isinstance(valuetype, evaluatable):
                    try:
                        ret = eval(str(value), globals(), locals())
                    except NameError:
                        ret = str(value)
                    except SyntaxError:
                        ret = ""
                    return ret
                    break
                else:
                    return valuetype(value)
                    break
        return str(value)

    def __setitem__(self, name, value):
        """Provides a method to set an item in the dict, checking the key for
        an embedded type hint or inspecting the value as necessary.

        NB If you provide an embedded type string it is your responsibility
        to make sure that it correctly describes the actual data
        typehintDict does not verify that your data and type string are
        compatible."""
        m = self.__regexGetType.search(name)
        if m is not None:
            k = m.group(1)
            t = m.group(2)
            self._typehints[k] = t
            if len(value) == 0:  # Empty data so reset to string and set empty
                super(typeHintedDict, self).__setitem__(k, "")
                self._typehints[k] = "String"
            else:
                super(typeHintedDict, self).__setitem__(k,
                                                self.__mungevalue(t, value))
        else:
            self._typehints[name] = self.__findtype(value)
            super(typeHintedDict, self).__setitem__(name,
                self.__mungevalue(self._typehints[name], value))

    def __delitem__(self, name):
        """Deletes the specified key"""
        del(self._typehints[name])
        super(typeHintedDict, self).__delitem__(name)

    def copy(self):
        """Provides a copy method that is aware of the type hinting strings"""
        return typeHintedDict([(x + '{' + self.type(x) + '}', self[x])
                                                             for x in self])

    def type(self, key):
        """Returns the typehint for the given k(s)

        @param key Either a single string key or a iterable type containing
                keys
        @return the string type hint (or a list of string type hints)"""
        if isinstance(key, str):
            return self._typehints[key]
        else:
            try:
                return [self._typehints[x] for x in key]
            except TypeError:
                return self._typehints[key]

    def export(self, key):
        """Exports a single metadata value to a string representation with type
        hint
        @param key The metadata key to export
        @return A string of the format : key{type hint} = value"""
        return key + "{" + self.type(key) + "}=" + str(self[key])


class MyForm(wx.Frame):
    """Provides an editable grid for the DataFile class to use display data"""

    #----------------------------------------------------------------------
    def __init__(self, dfile, **kwargs):
        """Constructor
        @param dfile An instance of the Stoner.DataFile object
        @ param **kwargs Keyword arguments - recognised values include"""
        import wx.grid as gridlib
        if not isinstance(dfile, DataFile):
            raise TypeError('First argument must be a Stoner.DataFile')
        cols = max(len(dfile.column_headers), 4)
        rows = max(len(dfile), 20)
        wx.Frame.__init__(self, parent=None, title="Untitled")
        self.Bind(wx.EVT_SIZE, self._OnSize)
        self.panel = wx.Panel(self)

        myGrid = gridlib.Grid(self.panel)
        myGrid.CreateGrid(rows, cols)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(myGrid, 1, wx.EXPAND)
        self.panel.SetSizer(self.sizer)

        for i in range(len(dfile.column_headers)):
            myGrid.SetColLabelValue(i, dfile.column_headers[i])
            for j in range(len(dfile)):
                myGrid.SetCellValue(j, i, str(dfile.data[j, i]))

    def _OnSize(self, evt):
        evt.Skip()


class DataFile(object):
    """@b Stoner.Core.DataFile is the base class object that represents
    a matrix of data, associated metadata and column headers.

    @b DataFile provides the mthods to load, save, add and delete data, index
    and slice data, manipulate metadata and column headings.

    Authors: Matt Newman, Chris Allen and Gavin Burnell"""
    #   CONSTANTS
    defaultDumpLocation = 'C:\\dump.csv'

    data = numpy.array([])
    metadata = typeHintedDict()
    filename = None
    column_headers = list()


    #   INITIALISATION

    def __init__(self, *args):
        """Constructor method

        various forms are recognised:
        @li DataFile('filename',<optional filetype>,<args>)
        Creates the new DataFile object and then executes the \b DataFile.load
        method to load data from the given \a filename
        @li DataFile(array)
        Creates a new DataFile object and assigns the \a array to the
        \b DataFile.data  attribute.
        @li DataFile(dictionary)
        Creates the new DataFile object, but initialises the metadata with
        \a dictionary
        @li  DataFile(array,dictionary),
        Creates the new DataFile object and does the combination of the
        previous two forms.
        @li DataFile(DataFile)
        Creates the new DataFile object and initialises all data from the
        existing \b DataFile instance. This on the face of it does the same as
        the assignment operator, but is more useful when one or other of the
        DataFile objects is an instance of a sub - class of DataFile

        @param *args Variable number of arguments that match one of the
        definitions above
        @return A new instance of the DataFile class.
        """
         # Now check for arguments t the constructor
        self.metadata=typeHintedDict()
        if len(args) == 1:
            #print type(args[0])
            if (isinstance(args[0], str) or (
                isinstance(args[0], bool) and not args[0])):
                                        # Filename- load datafile
                t = self.load(args[0])
                self.data = t.data
                self.metadata = t.metadata
                self.column_headers = t.column_headers
            elif isinstance(args[0], numpy.ndarray):
                                                    # numpy.array - set data
                self.data = args[0]
                self.column_headers = ['Column' + str(x)
                                    for x in range(numpy.shape(args[0])[1])]
            elif isinstance(args[0], dict):  # Dictionary - use as metadata
                self.metadata = args[0].copy()
            elif isinstance(args[0], DataFile):
                self.metadata = args[0].metadata.copy()
                self.data = args[0].data
                self.column_headers = args[0].column_headers
            else:
                raise SyntaxError("No constructor")
        elif len(args) == 2:
                            # 2 argument forms either array,dict or dict,array
            if isinstance(args[0], numpy.ndarray):
                self.data = args[0]
            elif isinstance(args[0], dict):
                self.metadata = args[0].copy()
            elif isinstance(args[0], str) and isinstance(args[1], str):
                self.load(args[0], args[1])
            if isinstance(args[1], numpy.ndarray):
                self.data = args[1]
            elif isinstance(args[1], dict):
                self.metadata = args[1].copy()
        elif len(args) > 2:
            apply(self.load, args)

    # Special Methods

    def __getattr__(self, name):
        """
        Called for \b DataFile.x to handle some special pseudo attributes
        and otherwise to act as a shortcut for @b DataFile.column

        @param name The name of the attribute to be returned.
        These include: records
        @return the DataFile object in various forms

        Supported attributes:
        @a records - return the DataFile data as a numpy structured
        array - i.e. rows of elements whose keys are column headings
        @a clone - returns a deep copy of the current DataFile instance

        Otherwise the @a name parameter is tried as an argument to
        ~b DataFile.column and the resultant column isreturned. If
        @b DataFile.column raises a KeyError this is remapped as an
        AttributeError.
        """
        if name == "records":
            dtype = [(x, numpy.float64) for x in self.column_headers]
            return self.data.view(dtype=dtype).reshape(len(self))
        elif name == "clone":
            return copy.deepcopy(self)
        else:
            try:
                return self.column(name)
            except KeyError:
                raise AttributeError(name +
                " is neither an attribute of DataFile, nor a column \
                heading of this DataFile instance")

    def __getitem__(self, name):
            # called for DataFile[x] returns row x if x is integer, or
            # metadata[x] if x is string
        """Called for \b DataFile[x] to return either a row or iterm of
        metadata

        @param name The name, slice or number of the part of the
        \b DataFile to be returned.
        @return an item of metadata or row(s) of data.
        \li If \a name is an integer then the corresponding single row will be
        rturned
        \li if \a name is a slice, then the corresponding rows of data will be
        returned. \li If \a name is a string then the metadata dictionary item
        with the correspondoing key will be returned.

        If a tuple is supplied as the arguement then there are a number of
        possible behaviours. If the first element of the tuple is a string,
        then it is assumed that it is the nth element of the named metadata is
        required. Otherwise itis assumed that it is a particular element
        within a column determined by the second part of the tuple that is
        required. e.g. DataFile['Temp',5] would return the 6th element of the
        list of elements in the metadata called 'Temp', while
        DataFile[5,'Temp'] would return the 6th row of the data column
        called 'Temp' and DataFile[5,3] would return the 6th element of the
        4th column.

        """
        if isinstance(name, slice):
            indices = name.indices(len(self))
            name = range(*indices)
            d = self.data[name[0], :]
            d = numpy.atleast_2d(d)
            for x in range(1, len(name)):
                d = numpy.append(d, numpy.atleast_2d(self.data[x, :]), 0)
            return d
        elif isinstance(name, int):
            return self.data[name, :]
        elif isinstance(name, str):
            return self.meta(name)
        elif isinstance(name, tuple) and len(name) == 2:
            x, y = name
            if isinstance(x, str):
                return self[x][y]
            else:
                d = numpy.atleast_2d(self[x])
                y = self.find_col(y)
                r = d[:, y]
                if len(r) == 1:
                    r = r[0]
                return r
        else:
            raise TypeError("Key must be either numeric of string")

    def __setitem__(self, name, value):
        """Called for \b DataFile[\em name ] = \em value to write mewtadata
        entries.
            @param name The string key used to access the metadata
            @param value The value to be written into the metadata.
            Currently bool, int, float and string values are correctly
            handled. Everythign else is treated as a string.
            @return Nothing."""
        self.metadata[name] = value

    def __add__(self, other):
        """ Implements a + operator to concatenate rows of data
                @param other Either a numpy array object or an instance
                of a \b DataFile object.
                @return A Datafile object with the rows of \a other appended
                to the rows of the current object.

                If \a other is a 1D numopy array with the same number of
                elements as their are columns in \a self.data then the numpy
               array is treated as a new row of data If \a ither is a 2D numpy
               array then it is appended if it has the same number of
               columns and \a self.data."""
        if isinstance(other, numpy.ndarray):
            if len(self.data) == 0:
                t = numpy.atleast_2d(other)
                c = numpy.shape(t)[1]
                if len(self.column_headers) < c:
                    self.column_headers.extend(map(lambda x: "Column_" +
                                str(x), range(c - len(self.column_headers))))
                newdata = self.__class__(self)
                newdata.data = t
                return newdata
            elif len(numpy.shape(other)) == 1:
                                    # 1D array, so assume a single row of data
                if numpy.shape(other)[0] == numpy.shape(self.data)[1]:
                    newdata = self.__class__(self)
                    newdata.data = numpy.append(self.data,
                                                numpy.atleast_2d(other), 0)
                    return newdata
                else:
                    return NotImplemented
            elif len(numpy.shape(other)) == 2 and numpy.shape(
                    other)[1] == numpy.shape(self.data)[1]:
                            # DataFile + array with correct number of columns
                newdata = self.__class__(self)
                newdata.data = numpy.append(self.data, other, 0)
                return newdata
            else:
                return NotImplemented
        elif isinstance(other, DataFile):  # Appending another DataFile
            if self.column_headers == other.column_headers:
                newdata = self.__class__(other)
                for x in self.metadata:
                    newdata[x] = self.__class__(self[x])
                newdata.data = numpy.append(self.data, other.data, 0)
                return newdata
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __and__(self, other):
        """Implements the & operator to concatenate columns of data in a
        \b Stoner.DataFile object.

        @param other Either a numpy array or \b Stoner.DataFile object
        @return A \b Stoner.DataFile object with the columns of other con
        catenated as new columns at the end of the self object.

        Whether \a other is a numopy array of \b Stoner.DataFile, it must
        have the same or fewer rows than the self object.
        The size of \a other is increased with zeros for the extra rows.
        If \a other is a 1D numpy array it is treated as a column vector.
        The new columns are given blank column headers, but the
        length of the \b Stoner.DataFile.column_headers is
        increased to match the actual number of columns.
        """
        if isinstance(other, numpy.ndarray):
            if len(other.shape) != 2:  # 1D array, make it 2D column
                other = numpy.atleast_2d(other)
                other = other.T
            if other.shape[0] <= self.data.shape[0]:
                    # DataFile + array with correct number of rows
                if other.shape[0] < self.data.shape[0]:
                    # too few rows we can extend with zeros
                    other = numpy.append(other, numpy.zeros((self.data.shape[0]
                                    - other.shape[0], other.shape[1])), 0)
                newdata = self.__class__(self)
                newdata.column_headers.extend([""
                                               for x in range(other.shape[1])])
                newdata.data = numpy.append(self.data, other, 1)
                return newdata
            else:
                return NotImplemented
        elif isinstance(other, DataFile):  # Appending another datafile
            if self.data.shape[0] == other.data.shape[0]:
                newdata = self.__class__(self)
                newdata.column_headers.extend(other.column_headers)
                for x in other.metadata:
                    newdata[x] = other[x]
                newdata.data = numpy.append(self.data, other.data, 1)
                return newdata
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __repr__(self):
        """Outputs the \b Stoner.DataFile object in TDI format.
        This allows one to print any \b Stoner.DataFile to a stream based
                object andgenerate a reasonable textual representation of
                the data.shape

                @return \a self in a textual format. """
        outp = "TDI Format 1.5" + "\t" + reduce(lambda x, y: str(x) +
        "\t" + str(y), self.column_headers) + "\n"
        m = len(self.metadata)
        (r, c) = numpy.shape(self.data)
        md = [self.metadata.export(x) for x in sorted(self.metadata)]
        for x in range(min(r, m)):
            outp = outp + md[x] + "\t" + reduce(lambda z, y:
                            str(z) + "\t" + str(y), self.data[x]) + "\n"
        if m > r:  # More metadata
            for x in range(r, m):
                    outp = outp + md[x] + "\n"
        elif r > m:  # More data than metadata
            for x in range(m, r):
                    outp = outp + "\t" + reduce(lambda z, y:
                        str(z) + "\t" + str(y), self.data[x]) + "\n"
        return outp

    def __len__(self):
        """Return the length of the data.shape
                @return Returns the number of rows of data
                """
        return numpy.shape(self.data)[0]

    def __setstate__(self, state):
        self.data = state["data"]
        self.column_headers = state["column_headers"]
        self.metadata = state["metadata"]

    def __getstate__(self):
        return {"data": self.data,  "column_headers":
            self.column_headers,  "metadata": self.metadata}

    def __reduce_ex__(self, p):
        return (DataFile, (), self.__getstate__())

    #   PRIVATE FUNCTIONS

    def __file_dialog(self, mode):
        """Creates a file dialog box for loading or saving ~b DataFile objects

        @param mode The mode of the file operation  'r' or 'w'
        @return A filename to be used for the file operation."""
        from enthought.pyface.api import FileDialog, OK
        # Wildcard pattern to be used in file dialogs.
        file_wildcard = "Text file (*.txt)|*.txt|Data file (*.dat)|\
        *.dat|All files|*"

        if mode == "r":
            mode = "open"
        elif mode == "w":
            mode = "save as"

        if self.filename is not None:
            filename = os.path.basename(self.filename)
            dirname = os.path.dirname(self.filename)
        else:
            filename = ""
            dirname = ""
        dlg = FileDialog(action=mode, wildcard=file_wildcard)
        dlg.open()
        if dlg.return_code == OK:
            self.filename = dlg.path
            return self.filename
        else:
            return None

    def __parse_metadata(self, key, value):
        """Parse the metadata string, removing the type hints into a separate
        dictionary from the metadata

        @param key The name of the metadata parameter to be written,
        possibly including a type hinting string.
        @value The value of the item of metadata.
        @return Nothing, but the current instance's metadata is changed.

        Uses the typehint to set the type correctly in the dictionary

        NB All the clever work of managing the typehinting is done in the
        metadata dictionary object now.
        """
        self.metadata[key] = value

    def __parse_data(self):
        """Internal function to parse the tab deliminated text file
        """
        reader = csv.reader(open(self.filename, "rb"), delimiter='\t',
                            quoting=csv.QUOTE_NONE)
        row = reader.next()
        assert row[0] == "TDI Format 1.5"
                            # Bail out if not the correct format
        self.data = numpy.array([])
        headers = row[1:len(row)]
        maxcol = 1
        for row in reader:
            if maxcol < len(row):
                    maxcol = len(row)
            if row[0].find('=') > -1:
                md = row[0].split('=')
                self.__parse_metadata(md[0], md[1])
            if (len(row[1:len(row)]) > 1) or len(row[1]) > 0:
                self.data = numpy.append(self.data, map(lambda x:
                                                        float(x), row[1:]))
        else:
            shp = (-1, maxcol - 1)
            self.data = numpy.reshape(self.data,  shp)
            self.column_headers = ["" for x in range(self.data.shape[1])]
            self.column_headers[0:len(headers)] = headers

    def __parse_plain_data(self, header_line=3, data_line=7,
                           data_delim=' ', header_delim=','):
        """An intrernal function for parsing deliminated data without a leading
        column of metadata.copy
        @param header_line is the line on which the column headers are
        recorded (default 3)
        @param data_line is the first line of tabulated data (default 7)
        @param data_delim is the deliminator for the data rows (default = space)
        @param header_delim is the deliminator for the header values
        (default = tab)
        @return Nothing, but updates the current instances data

        NBThe default values are configured fir read VSM data files
        """
        header_string = linecache.getline(self.filename, header_line)
        header_string = re.sub(r'["\n]', '', header_string)
        self.column_headers = map(lambda x:
            x.strip(),  header_string.split(header_delim))
        self.data = numpy.genfromtxt(self.filename, dtype='float',
                            delimiter=data_delim, skip_header=data_line - 1)

    def __loadHariboPlain(self):

        self.__parse_plain_data(0, 0, data_delim='\t', header_delim='\t')

    #   PUBLIC METHODS

    def get_filename(self, mode):
        self.filename = self.__file_dialog(mode)
        return self.filename

    def load(self, filename=None, fileType="TDI", *args):
        """DataFile.load(filename,type,*args)

            Loads data from file filename using routines dependent on the f
            ileType parameter
            fileType is one on TDI,VSM,BigBlue,csv Default is TDI.

            Example: To load Big Blue file

                d.load(file,"BigBlue",8,10)

            Where "BigBlue" is filetype and 8 / 10 are the line numbers of the
            headers / start of data respectively

            TODO: Implement a filename extension check to more intelligently
            guess the datafile type
            """

        if filename is None:
            filename = self.__file_dialog('r')
        else:
            self.filename = filename

        if fileType == "TDI":
            self.__parse_data()
        elif fileType == "csv":
            self.__parse_plain_data(args[0], args[1], args[2], args[3])
        elif fileType == "HariboPlain":
            self.__loadHariboPlain()
            self.column_headers = ['wavenumbers', 'intensity']
        else:
            raise SyntaxError()
        return self

    def save(self, filename=None):
        """Saves a string representation of the current DataFile object into
        the file 'filename'


                @param filename = None  filename to save data as, if this is \
                b None then the current filename for the object is used
                    If this is not set, then then a file dialog is used. If f
                    ilename is \b False then a file dialog is force.
                @return The current object
                """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):
            # now go and ask for one
            filename = self.__file_dialog('w')
        f = open(filename, 'w')
        f.write(repr(self))
        f.close()
        self.filename = filename
        return self

    def find_col(self, col):
        """Indexes the column headers in order to locate a column of data.shape
        @param col Which column(s) to retuirn indices for.

        @return  If @a col is an integer then simply returns the matching
        integer assuming that the corresponding column exists. If @a col
        is a string, then first attemps an exact string comparison, and then
        falls back to a regular expression search on the column headers. If
        either of these searches returns more than onbe match, then the
        first is used. If @a col  is a slice then a list of the matching
        indices is returned. If @a col is a list
        then a list of the results of apply @b DataFile.find_col on each
        element of @a col is returned.
        """
        if isinstance(col, int):  # col is an int so pass on
            if col < 0 or col >= len(self.column_headers):
                raise IndexError('Attempting to index a non - existant column')
            pass
        elif isinstance(col, str):  # Ok we have a string
            if col in self.column_headers:  # and it is an exact string match
                col = self.column_headers.index(col)
            else:  # ok we'll try for a regular expression
                test = re.compile(col)
                possible = filter(test.search, self.column_headers)
                if len(possible) == 0:
                    raise KeyError('Unable to find any possible column \
                    matches')
                col = self.column_headers.index(possible[0])
        elif isinstance(col, slice):
            indices = col.indices(numpy.shape(self.data)[1])
            col = range(*indices)
            col = self.find_col(col)
        elif isinstance(col, list):
            col = map(self.find_col, col)
        else:
            raise TypeError('Column index must be an integer, string, \
            list or slice')
        return col

    def column(self, col):
        """Extracts a column of data by index or name

        @param col is the column index as defined for @b DataFile.find_col
        @returns one or more columns of data"""
        if isinstance(col, slice):
            # convert a slice into a list and then continue
            indices = col.indices(numpy.shape(self.data)[1])
            col = range(*indices)
        if isinstance(col, list):
            d = self.column(col[0])
            d = numpy.reshape(d, (len(d), 1))
            for x in range(1, len(col)):
                t = self.column(col[x])
                t = numpy.reshape(t, (len(t), 1))
                d = numpy.append(d, t, 1)
            return d
        else:
            return self.data[:, self.find_col(col)]

    def meta(self, ky):
        """Returns some metadata

        @param ky The name of the metadata item to be returned.
        If @a key is not an exact match for an item of metadata,
        then a regular expression match is carried out.
        @return Returns the item of metadata."""
        if isinstance(ky, str):  # Ok we go at it with a string
            if ky in self.metadata:
                return self.metadata[ky]
            else:
                test = re.compile(ky)
                possible = filter(test.search, self.metadata)
                if len(possible) == 0:
                    raise KeyError("No metadata with keyname: " + ky)
                elif len(possible) == 1:
                    return self.metadata[possible[0]]
                else:
                    d = dict()
                    for p in possible:
                        d[p] = self.metadata[p]
                    return d
        else:
            raise TypeError("Only string are supported as search \
            keys currently")
            # Should implement using a list of strings as well

    def dir(self, pattern=None):
        """ Return a list of keys in the metadata, filtering wiht a regular
        expression if necessary

                @param pattern is a regular expression or None to
                list all keys
                @return Returns a list of metadata keys."""
        if pattern == None:
            return self.metadata.keys()
        else:
            test = re.compile(pattern)
            possible = filter(test.search, self.metadata.keys())
            return possible

    def keys(self):
        """An alias for @b DataFile.dir(None)

        @return a list of all the keys in the metadata dictionary"""
        return self.dir(None)

    def search(self, *args):
        """Searches in the numerica data part of the file for lines
        that match and returns  the corresponding rows

        Find row(s) that match the specified value in column:

        search(Column,value,columns= [list])

        Find rows that where the column is >= lower_limit and < upper_limit:

        search(Column,function ,columns= [list])

        Find rows where the function evaluates to true. Function should take
        two parameters x (float) and y(numpy array of floats).
        e.g. AnalysisFile.search('x',lambda x,y:
        x < 10 and y[0] == 2, ['y1','y2'])
        """

        if len(args) == 2:
            col = args[0]
            targets = []
            val = args[1]
        elif len(args) == 3:
            col = args[0]
            if not isinstance(args[2], list):
                c = [args[2]]
            else:
                c = args[2]
            targets = map(self.find_col, c)
            val = args[1]
        if len(targets) == 0:
            targets = range(self.data.shape[1])
        d = numpy.transpose(numpy.atleast_2d(self.column(col)))
        d = numpy.append(d, self.data[:, targets], 1)
        if callable(val):
            rows = numpy.nonzero([val(x[0], x[1:]) for x in d])[0]
        elif isinstance(val, float):
            rows = numpy.nonzero([x[0] == val for x in d])[0]
        return self.data[rows][:, targets]

    def unique(self, col, return_index=False, return_inverse=False):
        """Return the unique values from the specified column - pass through
        for numpy.unique"""
        return numpy.unique(self.column(col), return_index, return_inverse)

    def del_rows(self, col, val=None):
        """Searchs in the numerica data for the lines that match and deletes
        the corresponding rows
        @param col Column containg values to search for. Maybe a list or slice
        @param val Specifies rows to delete. Maybe None - in which case
        whole columns are deleted, a float in which case rows whose column
        \b col = \b val are deleted or a function - in which case rows where
        the function evaluates to be true are deleted.
        @return The current object

        If \b val is a function it should take two arguments - a float and a
        list. The float is the value of the
        current row that corresponds to column \b col abd the second
        argument is the current row.
            """
        if isinstance(col, slice) and val is None:
            indices = col.indices(len(self))
            col -= range(*indices)
        if isinstance(col, list) and val is None:
            col.sort(reverse=True)
            for c in col:
                self.del_rows(c)
        elif isinstance(col,  int) and val is None:
            self.data = numpy.delete(self.data, col, 0)
        else:
            col = self.find_col(col)
            d = self.column(col)
            if callable(val):
                rows = numpy.nonzero([val(x[col], x) for x in self])[0]
            elif isinstance(val, float):
                rows = numpy.nonzero([x == val for x in d])[0]
            self.data = numpy.delete(self.data, rows, 0)
        return self

    def add_column(self, column_data, column_header=None, index=None,
                   func_args=None, replace=False):
        """Appends a column of data or inserts a column to a datafile

        @param column_data An array or list of data to append or insert or
        a callable function that will generate new data
        @param column_header The text to set the column header to, if
        not supplied then defaults to 'col#'
        @param index The  index (numeric or string) to insert (or
        replace) the data
        @param func_args If @a column_data is a callable object, then this
        argument can be used to supply a dictionary of function arguments
        to the callable object.
        @param replace Replace the data or insert the data (default)
        @return The @b DataFile instance with the additonal column inserted.
        NB also modifies the original DataFile Instance."""
        if index is None:
            index = len(self.column_headers)
            replace = False
            if column_header is None:
                column_header = "Col" + str(index)
        else:
            index = self.find_col(index)
            if column_header is None:
                column_header = self.column_headers[index]
        if not replace:
            if len(self.column_headers) == 0:
                self.column_headers=[column_header]
            else:
                self.column_headers.insert(index, column_header)
        else:
            self.column_headers[index] = column_header

        # The following 2 lines make the array we are adding a
        # [1, x] array, i.e. a column by first making it 2d and
        # then transposing it.
        if isinstance(column_data, numpy.ndarray):
            if len(numpy.shape(column_data)) != 1:
                raise ValueError('Column data must be 1 dimensional')
            else:
                numpy_data = column_data
        elif callable(column_data):
            if isinstance(func_args, dict):
                new_data = [column_data(x, **func_args) for x in self]
            else:
                new_data = [column_data(x) for x in self]
            numpy_data = numpy.array(new_data)
        elif isinstance(column_data,  list):
            numpy_data=numpy.array(column_data)
        else:
            return NotImplemented
        if replace:
            self.data[:, index] = numpy_data
        else:
            if len(self.data) == 0:
                self.data=numpy.transpose(numpy.atleast_2d(numpy_data))
            else:
                self.data = numpy.insert(self.data, index, numpy_data, 1)
        return self

    def del_column(self, col):
        """Deletes a column from the current @b DataFile object
                @param col A column index (passed via @B DataFile.find_col)
                to the column to be deleted
                @return The @b DataFile object with the column deleted."""
        c = self.find_col(col)
        self.data = numpy.delete(self.data, c, 1)
        if isinstance(c, list):
            c.sort(reverse=True)
        else:
            c = [c]
        for col in c:
            del self.column_headers[col]
        return self

    def swap_column(self, swp, headers_too=True):
        """Swaps pairs of columns in the data. Useful for reordering
        data for idiot programs that expect columns in a fixed order.

            @param swp  A tuple of list of tuples of two elements. Each
            element will be iused as a column index (using the normal rules
            for matching columns).  The two elements represent the two
            columns that are to be swapped.
            @param headers_too A boolean that indicates the column headers
            are swapped as well
            @return A copy of the modified DataFile objects

            If @swp is a list, then the function is called recursively on each
            element of the list. Thus in principle the @swp could contain
            lists of lists of tuples
        """
        if isinstance(swp, list):
            for item in swp:
                self.swap_column(item, headers_too)
        elif isinstance(swp, tuple):
            col1 = self.find_col(swp[0])
            col2 = self.find_col(swp[1])
            self.data[:,  [col1, col2]] = self.data[:, [col2, col1]]
            if headers_too:
                self.column_headers[col1], self.column_headers[col2] =\
                self.column_headers[col2], self.column_headers[col1]
        else:
            raise TypeError("Swap parameter must be either a tuple or a \
            list of tuples")
        return self

    def reorder_columns(self, cols, headers_too=True):
        """Construct a new data array from the original data by assembling
        the columns in the order given
                @param cols A list of column indices (referred to the oriignal
                data set) from which to assemble the new data set
                @param headers_too Reorder the column headers in the same
                way as the data (defaults to True)
                @return A copy of the modified DataFile object"""
        if headers_too:
            self.column_headers = [self.column_headers[self.find_col(x)]
                                                            for x in cols]

        newdata = numpy.atleast_2d(self.data[:, self.find_col(cols.pop(0))])
        for col in cols:
            newdata = numpy.append(newdata, numpy.atleast_2d(self.data[:,
                                                self.find_col(col)]), axis=0)
        self.data = numpy.transpose(newdata)
        return self

    def rows(self):
        """Generator method that will iterate over rows of data
        @return Returns the next row of data"""
        (r, c) = numpy.shape(self.data)
        for row in range(r):
            yield self.data[row]

    def columns(self):
        """Generator method that will iterate over columns of data

        @return Returns the next column of data."""
        (r, c) = numpy.shape(self.data)
        for col in range(c):
            yield self.data[col]

    def sort(self, order):
        """Sorts the data by column name. Sorts in place and returns a
        copy of the sorted data object for chaining methods
        @param order Either a scalar integer or string or a list of integer
        or strings that represent the sort order
        @return A copy of the sorted object
        """
        if isinstance(order, list) or isinstance(order, tuple):
            order = [self.column_headers[self.find_col(x)] for x in order]
        else:
            order = [self.column_headers[self.find_col(order)]]
        d = numpy.sort(self.records, order=order)
        #print d
        self.data = d.view(dtype='f8').reshape(len(self), len(self.
                                                              column_headers))
        return self

    def edit(self):
        """Produce an editor window with a grid"""
        app = wx.PySimpleApp()
        frame = MyForm(self).Show()
        app.MainLoop()
        while app.IsMainLoopRunning:
            pass
