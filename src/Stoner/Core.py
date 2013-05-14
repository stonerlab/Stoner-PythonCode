import pdb############################################
#
# Core object of the Stoner Package
#


#
#############################################

# Imports

import fileinput
import re
import scipy
#import pdb # for debugging
import os
import sys
import numpy
import numpy.ma as ma
import math
import copy
import linecache
import wx
import inspect
import itertools
import collections


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
    __regexEvaluatable = re.compile(r'^(Cluster|\dD Array|List)')

    __types = {'Boolean': bool, 'I32': int, 'Double Float': float,
        'Cluster': dict, 'Array': numpy.ndarray,'List':list,  'String': str}
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

    def string_to_type(self, value):
        """Given a string value try to work out if there is a better python type dor the value
        @param value String representation of he value
        @return A python object of the natural type for value"""
        if not isinstance(value, str):
            raise TypeError("Value must be a string not a "+str(type(value)))
        value=value.strip()
        if len(value)==0:
            return None
        if value[0]=="[":
            try:
                return eval('list('+value+')') #List
            except SyntaxError:
                pass
        if value[0]=="{":
            try:
                return eval('dict('+value+')') #Dict
            except SyntaxError:
                pass
        if value in ['True', 'TRUE','true', 'Yes', 'YES', 'yes'] or value in ['False', 'FALSE', 'false', 'No','NO', 'no']:
            return value in ['True', 'TRUE','true', 'Yes', 'YES', 'yes'] #Booleab
        try:
            return int(value) # try as an int
        except ValueError:
            pass
        try:
            return float(value) # Ok try as a float
        except ValueError:
            return value.strip('"\'')

    def __setitem__(self, name, value):
        """Provides a method to set an item in the dict, checking the key for
        an embedded type hint or inspecting the value as necessary.

        NB If you provide an embedded type string it is your responsibility
        to make sure that it correctly describes the actual data
        typehintDict does not verify that your data and type string are
        compatible."""
        name=str(name)
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
        name=str(name)
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




class DataFile(object):
    """@b Stoner.Core.DataFile is the base class object that represents
    a matrix of data, associated metadata and column headers.

    @b DataFile provides the mthods to load, save, add and delete data, index
    and slice data, manipulate metadata and column headings.

    Authors: Matt Newman, Chris Allen and Gavin Burnell"""
    #   CONSTANTS
    defaultDumpLocation = 'C:\\dump.csv'

    data = ma.masked_array([])
    metadata = typeHintedDict()
    filename = None
    column_headers = list()
    priority=32
    debug=False
    _masks=[False]
    _conv_string=numpy.vectorize(lambda x:str(x))
    _conv_float=numpy.vectorize(lambda x:float(x))


    #   INITIALISATION

    def __init__(self, *args, **kargs):
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
            if (isinstance(args[0], str) or isinstance(args[0], unicode) or (
                isinstance(args[0], bool) and not args[0])):
                                        # Filename- load datafile
                t = self.load(*args, **kargs)
                self.data = ma.masked_array(t.data)
                self.metadata = t.metadata
                self.column_headers = t.column_headers
            elif isinstance(args[0], numpy.ndarray):
                                                    # numpy.array - set data
                self.data = ma.masked_array(args[0])
                self.column_headers = ['Column' + str(x)
                                    for x in range(numpy.shape(args[0])[1])]
            elif isinstance(args[0], dict):  # Dictionary - use as metadata
                self.metadata = args[0].copy()
            elif isinstance(args[0], DataFile):
                self.metadata = args[0].metadata.copy()
                self.data = ma.masked_array(args[0].data)
                self.column_headers = args[0].column_headers
            else:
                raise SyntaxError("No constructor")
        elif len(args) == 2:
                            # 2 argument forms either array,dict or dict,array
            if isinstance(args[0], numpy.ndarray):
                self.data = ma.masked_array(args[0])
            elif isinstance(args[0], dict):
                self.metadata = args[0].copy()
            elif isinstance(args[0], str) and isinstance(args[1], str):
                self.load(args[0], args[1])
            if isinstance(args[1], numpy.ndarray):
                self.data = ma.masked_array(args[1])
            elif isinstance(args[1], dict):
                self.metadata = args[1].copy()
        elif len(args) > 2:
            self.load(*args, **kargs)
        self.metadata["Stoner.class"]=self.__class__.__name__

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
        elif name=="subclasses":
            return {x.__name__:x for x in itersubclasses(DataFile)}
        elif name=="mask":
            return ma.getmaskarray(self.data)
        else:
            try:
                return self.column(name)
            except KeyError:
                raise AttributeError(name +
                " is neither an attribute of DataFile, nor a column \
                heading of this DataFile instance")

    def __setattr__(self, name, value):
        """Handles attempts to set attributes not covered with class attribute variables.
        @param name Name of attribute to set. Details of possible attributes below:
        
        \b mask Passes through to the mask attribute of self.data (which is a numpy masked array). Also handles
        the case where you pass a callable object to nask where we pass each row to the function and use the return reult as the mask"""
        if name=="mask":
            if callable(value):
                self._set_mask(value, invert=False)
            else:
                self.data.mask=value
        else:
            self.__dict__[name] = value 
    
    def __contains__(self, item):
        """Operator function for membertship tests - used to check metadata contents
        @param item String of metadata key
        @return iem in self.metadata"""
        return item in self.metadata

    def __getitem__(self, name):
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
        elif isinstance(name, str) or isinstance(name, unicode):
            name=str(name)
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

    def __delitem__(self,item):
        """Implements row or metadata deletion.
        @param item either an ingteger to delete a row or a string to delete metadata"""
        if isinstance(item,str):
            del(self.metadata[item])
        else:
            self.del_rows[item]
            
    
    
    
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
            new_data=numpy.zeros((len(other), len(self.column_headers)))*numpy.nan
            for i in range(len(self.column_headers)):
                column=self.column_headers[i]
                try:
                    new_data[:, i]=other.column(column)
                except KeyError:
                    pass
            newdata = self.__class__(other)
            for x in self.metadata:
                newdata[x] = self[x]
            newdata.data = numpy.append(self.data, new_data, 0)
            return newdata
        else:
            return NotImplemented('Failed in DataFile')

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
        newdata=self.__class__(self.clone)
        if isinstance(other, numpy.ndarray):
            if len(other.shape) != 2:  # 1D array, make it 2D column
                other = numpy.atleast_2d(other)
                other = other.T
            if numpy.product(self.data.shape)==0: #Special case no data yet                
                newdata.data=other
                newdata.column_headers=["Coumn "+str(i) for i in range(other.shape[1])]
            elif self.data.shape[0]==other.shape[0]:
                newdata.data=numpy.append(newdata.data,other,1)
                newdata.column_headers.extend(["Column "+str(i+len(newdata.column_headers)) for i in range(other.shape[1])])    
            elif self.data.shape[0]<other.shape[0]: #Need to extend self.data
                newdata.data=numpy.append(self.data,numpy.zeros((other.shape[0]-self.data.shape[0],self.data.shape[1])),0)
                newdata.data=numpy.append(self.data,other,1)
                newdata.column_headers.extend(["Column "+str(i+len(newdata.column_headers)) for i in range(other.shape[1])])            
            else:                    # DataFile + array with correct number of rows
                if other.shape[0] < self.data.shape[0]:
                    # too few rows we can extend with zeros
                    other = numpy.append(other, numpy.zeros((self.data.shape[0]
                                    - other.shape[0], other.shape[1])), 0)
                newdata.column_headers.extend([""
                                               for x in range(other.shape[1])])
                newdata.data = numpy.append(self.data, other, 1)
        elif isinstance(other, DataFile):  # Appending another datafile
            if self.data.shape[0] == other.data.shape[0]:
                newdata.column_headers.extend(other.column_headers)
                for x in other.metadata:
                    newdata[x] = other[x]
                newdata.data = numpy.append(self.data, other.data, 1)
            else:
                return NotImplemented
        else:
            return NotImplemented
        return newdata
            
    def __lshift__(self, other):
        """Overird the left shift << operator for a string or an iterable object to import using the __read_iterable() function
        @param other Either a string or iterable object used to source the DataFile object
        @return A new DataFile object
        """
        newdata=self.__class__()
        if isinstance(other, str):
            lines=itertools.imap(lambda x:x,  other.splitlines())
            newdata._read_iterable(lines)
        elif isinstance(other, collections.Iterable):
            newdata._read_iterable(other)
        return newdata
            

    def __repr__(self):
        """Outputs the \b Stoner.DataFile object in TDI format.
        This allows one to print any \b Stoner.DataFile to a stream based
                object andgenerate a reasonable textual representation of
                the data.shape

                @return \a self in a textual format. """
        outp = "TDI Format 1.5\t" + "\t".join(self.column_headers)+"\n"
        m = len(self.metadata)
        self.data=ma.masked_array(numpy.atleast_2d(self.data))
        (r, c) = numpy.shape(self.data)
        md = [self.metadata.export(x) for x in sorted(self.metadata)]
        for x in range(min(r, m)):
            outp = outp + md[x] + "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
        if m > r:  # More metadata
            for x in range(r, m):
                    outp = outp + md[x] + "\n"
        elif r > m:  # More data than metadata
            for x in range(m, r):
                    outp = outp + "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
        return outp

    def __len__(self):
        """Return the length of the data.shape
                @return Returns the number of rows of data
                """
        return numpy.shape(self.data)[0]

    def __setstate__(self, state):
        self.data = ma.masked_array(state["data"])
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

    def _set_mask(self, func, invert=False,  cumulative=False, col=0):
        """Applies func to each row in self.data and uses the result to set the mask for the row
        @param func A Callable object of the form lambda x:True where x is a row of data (numpy
        @pram invert Optionally invert te reult of the func test so that it unmasks data instead
        @param cumulative if tru, then an unmask value doesn't unmask the data, it just leaves it as it is."""
        
        i=-1
        args=len(inspect.getargs(func.__code__)[0])
        for r in self.rows():
            i+=1
            if args==2:
                t=func(r[col], r)
            else:
                t=func(r)
            if isinstance(t, bool) or isinstance(t, numpy.bool_):
                if t^invert:
                    self.data[i]=ma.masked
                elif not cumulative:
                    self.data[i]=self.data.data[i]
            else:
                for j in range(min(len(t), numpy.shape(self.data)[1])):
                    if t[j]^invert:
                        self.data[i, j]=ma.masked
                    elif not cumulative:
                        self.data[i, j]=self.data.data[i, j]
                    
    def _push_mask(self, mask=None):
        """Copy the current data mask to a temporary store and replace it with a new mask if supplied
        @param mask The new data mask to apply (defaults to None = unmask the data
        @return None"""
        self._masks.append(self.mask)
        if mask is None:
            self.data.mask=False
        else:
            self.mask=mask
            
    def _pop_mask(self):
        """Replaces the mask on the data with the last one stored by _push_mask()
        @return None"""
        self.mask=False
        self.mask=self._masks.pop()
        if len(self._masks)==0:
            self.__masks=[False]


    
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
        
        self._read_iterable(fileinput.FileInput(self.filename))
        
    def _read_iterable(self, reader):
        row=reader.next().split('\t')
        if row[0].strip()!="TDI Format 1.5":
            raise RuntimeError("Not a TDI File")
        self.column_headers=row[1:]
        cols=len(self.column_headers)
        self.data=ma.masked_array([])
        for r in reader:
            if r.strip()=="": # Blank line
                continue
            row=r.rstrip().split('\t')
            cols=max(cols, len(row)-1)
            if row[0].strip()!='':
                md=row[0].split('=')
                if len(md)==2:
                    self.metadata[md[0].strip()]=md[1].strip()
            if len(row)<2:
                continue
            self.data=numpy.append(self.data, self._conv_float(row[1:]))
        self.data=numpy.reshape(self.data, (-1, cols))

    #   PUBLIC METHODS

    def rename(self, old_col, new_col):
        """Renames columns without changing the underlying data
        @param old_col Old column index or name (using standard rules)
        @param new_col New name of column
        @return A copy of self
        """
        
        old_col=self.find_col(old_col)
        self.column_headers[old_col]=new_col
        return self
    
    def get(self, item):
        """A wrapper around __get_item__ that handles missing keys by returning None. This is useful for the DataFolder class
        @param item A string representing the metadata keyname
        @return self.metadata[item] or None if item not in self.metadata"""
        try:
            return self[item]
        except KeyError:
            return None


    def get_filename(self, mode):
        self.filename = self.__file_dialog(mode)
        return self.filename

    def load(self, filename=None, auto_load=True,  filetype=None,  *args, **kargs):
        """DataFile.load(filename,type,*args)
        @param filename path to file to load
        @param auto_load If True (default) then the load routine tries all the subclasses of DataFile in turn to load the file
        @param filetype If not none then tries using filetype as the loader
        @return A copy of the loaded instance
            """

        if filename is None or (isinstance(filename, bool) and not filename):
            filename = self.__file_dialog('r')
        else:
            self.filename = filename

        failed=True
        try:
            if filetype is None:
                self.__parse_data()
                self["Loaded as"]="DataFile"
            else:
                self.__class__(filetype(filename))
                self["Loaded as"]=filetype.__name__
            failed=False
            return self
        except RuntimeError: # We failed to parse assuming this was a TDI
            if auto_load: # We're going to try every subclass we can
                subclasses={x:x.priority for x in itersubclasses(DataFile)}
                for cls, priority in sorted(subclasses.iteritems(), key=lambda (k,v): (v,k)):
                    if self.debug:
                        print cls.__class__.__name__
                    try:
                        test=cls()
                        test.load(self.filename, auto_load=False)
                        failed=False
                        break
                    except Exception as inst:
                        continue

        if failed:
            raise SyntaxError("Failed to load file")
        else:
            self.data=ma.masked_array(test.data)
            self.metadata=test.metadata
            self.column_headers=test.column_headers
            self["Loaded as"]=cls.__name__

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
        first is used. If neither string tests works, the tries casting the string to an integer and checking that
        the result is in range for the size of the  dataset.
        If @a col  is a slice then a list of the matching
        indices is returned. If @a col is a list
        then a list of the results of apply @b DataFile.find_col on each
        element of @a col is returned.
        """
        if isinstance(col, int):  # col is an int so pass on
            if col < 0 or col >= len(self.column_headers):
                raise IndexError('Attempting to index a non - existant column')
            pass
        elif isinstance(col, str) or isinstance(col, unicode):  # Ok we have a string
            col=str(col)
            if col in self.column_headers:  # and it is an exact string match
                col = self.column_headers.index(col)
            else:  # ok we'll try for a regular expression
                test = re.compile(col)
                possible = filter(test.search, self.column_headers)
                if len(possible) == 0:
                    try:
                        col=int(col)
                    except ValueError:
                        raise KeyError('Unable to find any possible column \
                    matches')
                    if col<0 or col>=self.data.shape[1]:
                        raise KeyError('Column index out of range')
                else:
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

    def search(self, xcol,value,columns=None):
        """Searches in the numerica data part of the file for lines
        that match and returns  the corresponding rows
        
        @param xcol is a Search Column Index
        @param value is a numerical value, a tuple, a list of numbers or tuples, or a callable function
        @param columns is either a index or array of indices or None (default) for all columns.
		@return numpy array of matching rows or column values depending on the arguements.

        """
        rows=[]
        for (r,x) in zip(range(len(self)),self.column(xcol)):
            if isinstance(value,tuple) and value[0]<=x<=value[1]:
                rows.append(r)
            elif isinstance(value,list):
                for v in value:
                    if isinstance(v,tuple) and v[0]<=x<=v[1]:
                        rows.append(r)
                        break
                    elif x==v:
                        rows.append(r)
                        break
            elif callable(value) and value(x,self[r]):
                rows.append(r)
            elif x==value:
                rows.append(r)
        if columns is None: #Get the whole slice
            return self.data[rows,:]
        elif isinstance(columns,str) or isinstance(columns,int):
            columns=self.find_col(columns)
            if len(rows)==1:
                return self.data[rows,columns]
            else:
                return self.data[rows][:, columns]
        else:
            targets=[self.find_col(t) for t in columns]
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
                rows = numpy.nonzero([bool(val(x[col], x) and not x.mask) for x in self])[0]
            elif isinstance(val, float):
                rows = numpy.nonzero([x == val for x in d])[0]
            self.data = ma.masked_array(numpy.delete(self.data, rows, 0), mask=numpy.delete(self.data.mask, rows, 0))
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
                self.data=ma.masked_array(numpy.transpose(numpy.atleast_2d(numpy_data)))
            else:
                self.data = ma.masked_array(numpy.insert(self.data, index, numpy_data, 1))
        return self

    def del_column(self, col):
        """Deletes a column from the current @b DataFile object
                @param col A column index (passed via @B DataFile.find_col)
                to the column to be deleted
                @return The @b DataFile object with the column deleted."""
        c = self.find_col(col)
        self.data = numpy.ma.masked_array(numpy.delete(self.data, c, 1), mask=numpy.delete(self.data.mask, c, 1))
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
        self.data = ma.masked_array(numpy.transpose(newdata))
        return self
        
    def insert_rows(self, row, new_data):
        """Insert new_data into the data array at position row. This is a wrapper for numpy.insert
        @param row Data row to insert into
        @param new_data a numpy array with an equal number of columns as the main data array containing the new row(s) of data to insert
        @return A copy of the modified DataFile object"""
        self.data=numpy.insert(self.data, row,  new_data, 0)
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
        self.data = ma.masked_array(d.view(dtype='f8').reshape(len(self), len(self.
                                                              column_headers)))
        return self

    def edit(self):
        """Produce an editor window with a grid"""
        app = wx.PySimpleApp()
        frame = MyForm(self).Show()
        app.MainLoop()
        while app.IsMainLoopRunning:
            pass

def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """

    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub
