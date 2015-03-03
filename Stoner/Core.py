"""Stoner.Core provides the core classes for the Stoner package.

Classes:
    `DataFile` :the base class for representing a single data set of experimental data.
    `typeHintedDict`: a dictionary subclass that tries to keep track of the underlying type of data
        stored in each element. This class facilitates the export to strongly typed
        languages such as LabVIEW.

"""
from Stoner.compat import *
import re
#import pdb # for debugging
import os
import csv
import numpy as _np_
import numpy.ma as _ma_
import copy
import os.path as path
import inspect as _inspect_
import itertools
from collections import Iterable,OrderedDict

class _attribute_store(dict):
    """A class that provides attributes that refer to columns in a DataFile instance."""

    def __init__(self,*args,**kargs):
        if len(args)==1 and isinstance(args[0],dict):
            self.update(args[0])
        else:
            super(_attribute_store,self).__init__(*args,**kargs)

    def __setattr__(self,name,value):
        self[name]=value

    def __getattr__(self,name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError

class _tab_delimited(csv.Dialect):
    """A customised csv dialect class for reading tab delimited text files."""
    delimiter="\t"
    quoting=csv.QUOTE_NONE
    doublequote=False
    lineterminator="\r\n"

class _setas(object):
    """A Class that provides a mechanism for managing the column assignments in a DataFile like object."""

    def __init__(self,initial_val=None,**kargs):
        """Constructs the setas instance and sets an initial value.

        Args:
            ref (DataFile): Contains a reference to the owning DataFile instance

        Keyword Arguments:
            initial_val (string or list or dict): Initial values to set
        """
        self.setas=list()
        self.column_headers=[]
        self.cols=_attribute_store()

        if initial_val is not None:
            self(initial_val)
        elif len(kargs)>0:
            self(**kargs)

    def __call__(self,*args,**kargs):
        """Treat the current instance as a callable object and assign columns accordingly.

         Variois forms of this method are accepted::

            setas("xyzuvw")
            setas(["x"],["y"],["z"],["u"],["v"],["w"])
            setas(x="column_1",y=3,column4="z")
        """
        try:
            assert len(args)==0 or len(args)==1
            if len(args)==1:
                assert isinstance(args[0],string_types) or isinstance(args[0],Iterable)
            elif len(args)==1:
                assert len(kargs)>0
        except AssertionError:
            raise SyntaxError("setas must be called with a single argument - string or other iterable")

        if len(self.setas)<len(self.column_headers):
            self.setas.extend(list("."*(len(self.column_headers)-len(self.setas))))

        if len(args)>0:
            value=args[0]
            if isinstance(value,string_types): # expand the number-code combos in value
                pattern=re.compile("[^0-9]*(([0-9]+?)(x|y|z|d|e|f|u|v|w|\.|\-))")
                while True:
                    res=pattern.match(value)
                    if res is None:
                        break
                    (total,count,code)=res.groups()
                    if count=="":
                        count=1
                    else:
                        count=int(count)
                    value=value.replace(total,code*count,1)
        else:
            value=kargs
        if isinstance(value,dict):
            alt_vals=dict()
            for k,v in value.items():
                if v not in alt_vals:
                    alt_vals[v]=[k]
                else:
                    alt_vals[v].append(k)

            for typ in "xyzdefuvw.-":
                if typ in value:
                    try:
                        for c in self.find_col(value[typ],True): #x="Col1" type
                            self.setas[c]=typ
                    except KeyError:
                        pass
                if typ in alt_vals:
                    try:
                        for c in self.find_col(alt_vals[typ],True): #col1="x" type
                            self.setas[c]=typ
                    except KeyError:
                        pass
        elif isinstance(value,Iterable):
            if len(value)> len(self.column_headers):
                value=value[:len(self.column_headers)]
            elif len(value)<len(self.column_headers):
                value=[v for v in value] # Ensure value is now a list
                value.extend(list("."*(len(self.column_headers)-len(value))))
            for i,v in enumerate(list(value)):
                if v.lower() not in "xyzedfuvw.-":
                    raise ValueError("Set as column element is invalid: {}".format(v))
                if v!="-":
                    self.setas[i]=v.lower()
        else:
            raise ValueError("Set as column string ended with a number")
        self.cols.update(self._get_cols())



    def __getitem__(self,name):
        """Permit the setas attribute to be treated like either a list or a dictionary.

        Args:
            name (int, slice or string): if *name* is an integer or a slice, return the column type
                of the corresponding column(s). If a string, should be a single letter
                from the set x,y,z,u,v,w,d,e,f - if so returns the corresponding
                column(s)

        Returns:
            Either a single letter x,y,z,u,v,w,d,e or f, or a list of letters if used in
            list mode, or a single coliumn name or list of names if used in dictionary mode.
        """
        if isinstance(name,string_types) and len(name)==1 and name in "xyzuvwdef.-":
            ret=list()
            for i,v in enumerate(self.setas):
                if v==name:
                    ret.append(self.column_headers[i])
        elif isinstance(name,slice):
            indices = name.indices(len(self.setas))
            name = range(*indices)
            ret=[self[x] for x in name]
        else:
            try:
                name=int(name)
                ret= [self.setas[name]]
            except ValueError:
                raise TypeError("Index should be a number, slice or x,y,z,u,v,w,e,d of f")
        if len(ret)==1:
            ret=ret[0]
        return ret

    def __setitem__(self,name,value):
        """Allow setting of the setas variable like a dictionary or a list.

        Args:
            name (string or int): If name is a string, it should be in the set x,y,z,u,v,w,d,e or f
                and value should be a column index type. If name is an integer, then value should be
                a single letter string in the set above.
            value (integer or column index): See above.
        """
        if isinstance(name,string_types) and len(name)==1 and name in "xyzuvwdef.-":
            self({name:value})
        else:
            try:
                name=int(name)
                if len(value)==1 and value in "xyzuvwdef.":
                    self.setas[name]=value
                elif value=="-":
                    pass
                else:
                    raise ValueError("Column types can only be set to x,y,z,u,v,w,d,e, or f, not to {}".format(value))
            except ValueError:
                kargs={name:value}
                self(**kargs)

    def __len__(self):
        return len(self.setas)

    def __repr__(self):
        if len(self.setas)> len(self.column_headers):
            self.setas=self.setas[:len(self.column_headers)]
        elif len(self.setas)<len(self.column_headers):
            self.setas.extend(list("."*(len(self.column_headers)-len(self.setas))))
        return self.setas.__repr__()

    def find_col(self, col,force_list=False):
        """Indexes the column headers in order to locate a column of data.shape.

        Indexing can be by supplying an integer, a string, a regular experssion, a slice or a list of any of the above.

        -   Integer indices are simply checked to ensure that they are in range
        -   String indices are first checked for an exact match against a column header
            if that fails they are then compiled to a regular expression and the first
            match to a column header is taken.
        -   A regular expression index is simply matched against the column headers and the
            first match found is taken. This allows additional regular expression options
            such as case insensitivity.
        -   A slice index is converted to a list of integers and processed as below
        -   A list index returns the results of feading each item in the list at :py:meth:`find_col`
            in turn.

        Args:
            col (int, a string, a re, a slice or a list):  Which column(s) to retuirn indices for.

        Keyword Arguments:
            force_list (bool): Force the output always to be a list. Mainly for internal use only

        Returns:
            The matching column index as an integer or a KeyError
        """
        if isinstance(col, int):  # col is an int so pass on
            if not 0<=col<len(self.column_headers):
                raise IndexError('Attempting to index a non - existant column '+str(col))
        elif isinstance(col, string_types):  # Ok we have a string
            col=str(col)
            if col in self.column_headers:  # and it is an exact string match
                col = self.column_headers.index(col)
            else:  # ok we'll try for a regular expression
                test = re.compile(col)
                possible =[x for x in self.column_headers if test.search(x)]
                if len(possible) == 0:
                    try:
                        col=int(col)
                    except ValueError:
                        raise KeyError('Unable to find any possible column matches for '+str(col))
                    if col<0 or col>=self.data.shape[1]:
                        raise KeyError('Column index out of range')
                else:
                    col = self.column_headers.index(possible[0])
        elif isinstance(col,re._pattern_type):
            test = col
            possible = [x for x in self.column_headers if test.search(x)]
            if len(possible) == 0:
                raise KeyError('Unable to find any possible column matches for '+str(col.pattern))
            else:
                col = self.find_col(possible)
        elif isinstance(col, slice):
            indices = col.indices(_np_.shape(self.data)[1])
            col = range(*indices)
            col = self.find_col(col)
        elif isinstance(col, list):
            col = [self.find_col(x) for x in col]
        else:
            raise TypeError('Column index must be an integer, string, \
            list or slice')
        if force_list and not isinstance(col,list):
            col=[col]
        return col

    def _get_cols(self,what=None,startx=0):
        """Uses the setas attribute to work out which columns to use for x,y,z etc.

        Keyword Arguments:
            what (string): Returns either xcol, ycol, zcol, ycols,xcols rather than the full dictionary
            starts (int): Start looking for x columns at this column.

        Returns:
            A single integer, a list of integers or a dictionary of all columns.
        """

        if len(self.setas)<len(self.column_headers):
            self.setas.extend(list("."*(len(self.column_headers)-len(self.setas))))
        try:
            xcol=self.setas[startx:].index("x")+startx
            maxcol=self.setas[xcol+1:].index("x")+xcol+1
        except ValueError:
            if "x" not in self.setas:
                xcol=None
                maxcol=startx
            else:
                maxcol=len(self.column_headers)+1
        try:
            xerr=self.setas[startx:maxcol].index("d")+startx
        except ValueError:
            xerr=None
        ycol=list()
        yerr=list()
        starty=xcol
        has_yerr=False
        while "y" in self.setas:
            try:
                ycol.append(self.setas[starty:maxcol].index("y")+starty)
            except ValueError:
                break
            try:
                yerr.append(self.setas[starty:maxcol].index("e")+starty)
                has_yerr=True
            except ValueError:
                yerr.append(None)
            starty=ycol[-1]+1
        zcol=list()
        zerr=list()
        startz=xcol
        has_zerr=False
        while "z" in self.setas:
            try:
                zcol.append(self.setas[startz:maxcol].index("z")+startz)
            except ValueError:
                break
            startz=zcol[-1]+1
            try:
                zerr.append(self.setas[startz:maxcol].index("g")+startz)
                has_zerr=True
            except ValueError:
                zerr.append(None)

        ucol=list()
        startu=xcol
        while "u" in self.setas:
            try:
                ucol.append(self.setas[startu:maxcol].index("u")+startu)
            except ValueError:
                break
            startu=ucol[-1]+1
        vcol=list()
        startv=xcol
        while "v" in self.setas:
            try:
                vcol.append(self.setas[startv:maxcol].index("v")+startv)
            except ValueError:
                break
            startv=vcol[-1]+1
        wcol=list()
        startw=xcol
        while "w" in self.setas:
            try:
                wcol.append(self.setas[startw:maxcol].index("w")+startw)
            except ValueError:
                break
            startw=wcol[-1]+1

        if xcol is None:
            axes=0
        elif len(ycol)==0:
            axes=1
        elif len(zcol)==0:
            axes=2
        else:
            axes=3
        if axes==2 and len(ucol)*len(vcol)>0:
            axes=4
        elif axes==3:
            if len(ucol)*len(vcol)*len(wcol)>0:
                axes=6
            elif len(ucol)*len(vcol)>0:
                axes=5
        ret=_attribute_store()
        ret.update({"xcol":xcol,"xerr":xerr,
             "ycol":ycol,"yerr":yerr,
                 "zcol":zcol,"zerr":zerr,
                 "ucol":ucol,"vcol":vcol,"wcol":wcol,
                 "axes":axes})
        ret["has_xerr"]=xerr is not None
        ret["has_yerr"]=has_yerr
        ret["has_zerr"]=has_zerr
        ret["has_uvw"]=len(ucol)!=0
        if what=="xcol":
            ret=ret["xcol"]
        elif what in ("ycol","zcol","ucol","vcol","wcol","yerr","zerr"):
            ret=ret[what][0]
        elif what in ("ycols","zcols","ucols","vcols","wcols","yerrs","zerrs"):
            ret=ret[what[0:-1]]
        return ret

class _evaluatable(object):
    """A very simple class that is just a placeholder."""
    pass


class typeHintedDict(dict):
    """Extends a regular dict to include type hints of what each key contains.

    The CM Physics Group at Leeds makes use of a standard file format that closely matches
    the :py:class:`DataFile` data structure. However, it is convenient for this file format
    to be ASCII text for ease of use with other programs. In order to represent metadata which
    can have arbitary types, the LabVIEW code that generates the data file from our measurements
    adds a type hint string. The Stoner Python code can then make use of this type hinting to
    choose the correct representation for the metadata. The type hinting information is retained
    so that files output from Python will retain type hints to permit them to be loaded into
    strongly typed languages (sch as LabVIEW).

    Attributes:
        _typehints (dict): The backing store for the type hint information
        __regexGetType (re): Used to extract the type hint from a string
        __regexSignedInt (re): matches type hint strings for signed intergers
        __regexUnsignedInt (re): matches the type hint string for unsigned integers
        __regexFloat (re): matches the type hint strings for floats
        __regexBoolean (re): matches the type hint string for a boolean
        __regexStrng (re): matches the type hint string for a string variable
        __regexEvaluatable (re): matches the type hint string for a compoind data type
        __types (dict): mapping of type hinted types to actual Python types
        __tests (dict): mapping of the regex patterns to actual python types
    """
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
        'Cluster': dict, 'Array': _np_.ndarray,'List':list,  'String': str}
    # This is the inverse of the __tests below - this gives
    # the string type for standard Python classes

    __tests = [(__regexSignedInt, int), (__regexUnsignedInt, int),
             (__regexFloat, float), (__regexBoolean, bool),
             (__regexString, str), (__regexEvaluatable, _evaluatable())]
        # This is used to work out the correct python class for
        # some string types

    def __init__(self, *args, **kargs):
        """typeHintedDict constructor method.

        Args:
            *args, **kargs: Pass any parameters through to the dict() constructor.


        Calls the dict() constructor, then runs through the keys of the
        created dictionary and either uses the string type embedded in
        the keyname to generate the type hint (and remove the
        embedded string type from the keyname) or determines the likely
        type hint from the value of the dict element.
        """

        super(typeHintedDict, self).__init__(*args, **kargs)
        for key in self:  # Chekc through all the keys and see if they contain
                                    # type hints. If they do, move them to the
                                    # _typehint dict
            value=super(typeHintedDict,self).__getitem__(key)
            super(typeHintedDict,self).__delitem__(key)
            self[key]=value #__Setitem__ has the logic to handle embedded type hints correctly

    def __getattr__(self, name):
        """Handles attribute access"""
        if name == "types":
            return self._typehints
        else:
            raise AttributeError

    def findtype(self,  value):
        """Determines the correct string type to return for common python
        classes.

        Args:
            value (any): The data value to determine the type hint for.

        Returns:
            A type hint string

        Note:
            Understands booleans, strings, integers, floats and _np_
            arrays(as arrays), and dictionaries (as clusters).
        """
        typ = "String"
        for t in self.__types:
            if isinstance(value, self.__types[t]):
                if t == "Cluster":
                    elements = []
                    for k in  value:
                        elements.append(self.findtype(value[k]))
                    tt = ','
                    tt = tt.join(elements)
                    typ = 'Cluster (' + tt + ')'
                elif t == 'Array':
                    z = _np_.zeros(1, dtype=value.dtype)
                    typ = (str(len(_np_.shape(value))) + "D Array (" +
                                                self.findtype(z[0]) + ")")
                else:
                    typ = t
                break
        return typ

    def __mungevalue(self, t, value):
        """Based on a string type t, return value cast to an
        appropriate python class.

        Args:
            t (string): is a string representing the type
            value (any): is the data value to be munged into the
                        correct class
        Returns:
            Returns the munged data value

        Detail:
            The class has a series of precompiled regular
            expressions that will match type strings, a list of these has been
            constructed with instances of the matching Python classes. These
            are tested in turn and if the type string matches the constructor of
            the associated python class is called with value as its argument."""
        ret=None
        for (regexp, valuetype) in self.__tests:
            m = regexp.search(t)
            if m is not None:
                if isinstance(valuetype, _evaluatable):
                    try:
                        ret = eval(str(value), globals(), locals())
                    except NameError:
                        ret = str(value)
                    except SyntaxError:
                        ret = ""
                    break
                else:
                    ret=valuetype(value)
                    break
        else:
            ret=str(value)
        return ret

    def string_to_type(self, value):
        """Given a string value try to work out if there is a better python type dor the value.

        First of all the first character is checked to see if it is a [ or { which would
        suggest this is a list of dictionary. If the value looks like a common boolean
        value (i.e. Yes, No, True, Fale, On, Off) then it is assumed to be a boolean value.
        Fianlly it interpretation as an int, float or string is tried.

        Args:
            value (string): string representation of he value
        Returns:
            A python object of the natural type for value"""
        ret=None
        if not isinstance(value, string_types):
            raise TypeError("Value must be a string not a "+str(type(value)))
        value=value.strip()
        if len(value)!=0:
            tests=['list('+value+')','dict('+value+')']
            try:
                i="[{".index(value[0])
                ret=eval(tests[i])
            except (SyntaxError,ValueError):
                if value.lower() in ['true', 'ues','on','false','no','off']:
                    ret=value.lower() in ['true','yes','on'] #Booleab
                else:
                    for trial in [int,float,str]:
                        try:
                            ret=trial(value)
                            break
                        except ValueError:
                            continue
                    else:
                        ret=None
        return ret

    def _get_name_(self,name):
        """Checks a string name for an embedded type hint and strips it out.

        Args:
            name(string): String containing the name with possible type hint embedeed
        Returns:
            (name,typehint) (tuple): A tuple containing just the name of the mateadata and (if found
                the type hint string),
        """
        name=str(name)
        m = self.__regexGetType.search(name)
        if m is not None:
            k = m.group(1)
            t = m.group(2)
            return k,t
        else:
            k=name
            t=None
            return k,None

    def __getitem__(self,name):
        """Provides a get item method that checks whether its been given a typehint in the
        item name and deals with it appropriately.

        Args:
            name (string): metadata key to retrieve

        Returns:
            metadata value
        """
        (name,typehint)=self._get_name_(name)
        value=super(typeHintedDict,self).__getitem__(name)
        if typehint is not None:
            value=self.__mungevalue(typehint,value)
        return value

    def __setitem__(self, name, value):
        """Provides a method to set an item in the dict, checking the key for
        an embedded type hint or inspecting the value as necessary.

        Args:
            name (string): The metadata keyname
            value (any): The value to store in the metadata string

        Note:
            If you provide an embedded type string it is your responsibility
            to make sure that it correctly describes the actual data
            typehintDict does not verify that your data and type string are
            compatible."""
        name,typehint=self._get_name_(name)
        if typehint is not None:
            self._typehints[name] = typehint
            if len(str(value)) == 0:  # Empty data so reset to string and set empty
                super(typeHintedDict, self).__setitem__(name, "")
                self._typehints[name] = "String"
            else:
                super(typeHintedDict, self).__setitem__(name,
                                                self.__mungevalue(typehint, value))
        else:
            self._typehints[name] = self.findtype(value)
            super(typeHintedDict, self).__setitem__(name,
                self.__mungevalue(self._typehints[name], value))

    def __delitem__(self, name):
        """Deletes the specified key.

        Args:
            name (string): The keyname to be deleted"""
        name=self._get_name_(name)[0]
        del(self._typehints[name])
        super(typeHintedDict, self).__delitem__(name)

    def copy(self):
        """Provides a copy method that is aware of the type hinting strings.

        This produces a flat dictionary with the type hint embedded in the key name.

        Returns:
            A copy of the current typeHintedDict
        """
        ret=typeHintedDict()
        for k in self.keys():
            t=self._typehints[k]
            nk=k+"{"+t+"}"
            ret[nk]=copy.deepcopy(self[k])
        return ret

    def type(self, key):
        """Returns the typehint for the given k(s).

        This simply looks up the type hinting dictionary for each key it is given.

        Args:
            key (string or sequence of strings): Either a single string key or a iterable type containing
                keys
        Returns:
            The string type hint (or a list of string type hints)"""
        if isinstance(key, str):
            return self._typehints[key]
        else:
            try:
                return [self._typehints[x] for x in key]
            except TypeError:
                return self._typehints[key]

    def export(self, key):
        """Exports a single metadata value to a string representation with type
        hint.

        In the ASCII based file format, the type hinted metadata is represented
        in the first column of a tab delimited text file as a series of lines
        with format keyname{typhint}=string_value.

        Args:
            key (string): The metadata key to export
        Returns:
            A string of the format : key{type hint} = value"""
        return "{}{{{}}}={}".format(key,self.type(key),self[key])




class DataFile(object):
    """:py:class:`Stoner.Core.DataFile` is the base class object that represents
    a matrix of data, associated metadata and column headers.

    Attributes:
        data (2D Array) A numpy masked array of data (usually floats)
        metadata (typeHintedDict): of key-value metadata pairs. The dictionary
            tries to retain information about the type of data so as to aid import and
            export from CM group LabVIEw code.
        column_headers (list): of strings of the column names of the data
        title (string): The title of the measurement
        filename (string): The current filename of the data if loaded from or
            already saved to disc. This is the default filename used by the :py:meth:`load` and  :py:meth:`save`
        mask (array of booleans): Returns the current mask applied to the numerical data equivalent to self.data.mask
        setas (list or string): Defines certain columns to contain X, Y, Z or errors in X,Y,Z data.
        shape (tuple of integers): Returns the shape of the data (rows,columns) - equivalent to self.data.shape
        records (numpoy record array): Returns the data in the form of a list of dictionaries
        clone (DataFile): Creates a deep copy of the :py:class`DataFile` object
        subclasses (list): Returns a list of all the subclasses of DataFile currently in memory, sorted by
            their py:attr:`DataFile.priority`. Each entry in the list consists of the string name of the subclass
            and the class object.
        patterns (list): A list of strings containing file glob patterns that are typically used for datafiles
            that the :py:meth:`DataFile.load` method will read. This is used for the file dialog boxes.
        priority (int): A class attribute used to indivate the order in which the autoloader should attempt to load
            a data file. See :py:meth:`DataFile.load` for details.
    """

    #Class attributes
    #Priority is the load order for the class
    priority=32
    patterns=["*.txt","*.tdi"] # Recognised filename patterns


    _conv_string=_np_.vectorize(lambda x:str(x))
    _conv_float=_np_.vectorize(lambda x:float(x))

    #   INITIALISATION

    def __init__(self, *args, **kargs):
        """Constructor method for :py:class:`DataFile`.

        various forms are recognised

        .. py:function:: DataFile('filename',<optional filetype>,<args>)
            :noindex:

            Creates the new DataFile object and then executes the :py:class:`DataFile`.load
            method to load data from the given *filename*.

        .. py:function:: DataFile(array)
            :noindex:

            Creates a new DataFile object and assigns the *array* to the
            :py:attr:`DataFile.data`  attribute.

        .. py:function:: DataFile(dictionary)
            :noindex:

            Creates the new DataFile object, but initialises the metadata with
            :parameter dictionary

        .. py:function:: DataFile(array,dictionary)
            :noindex:

            Creates the new DataFile object and does the combination of the
            previous two forms.

        .. py:function:: DataFile(DataFile)
            :noindex:

            Creates the new DataFile object and initialises all data from the
            existing :py:class:`DataFile` instance. This on the face of it does the same as
            the assignment operator, but is more useful when one or other of the
            DataFile objects is an instance of a sub - class of DataFile

        Args:
            args (positional arguments): Variable number of arguments that match one of the
                definitions above
            kargs (keyword Arguments): All keyword arguments that match public attributes are
                used to set those public attributes.

        Returns:
            A new instance of the DataFile class.
        """
        # init instance attributes
        self._setas=_setas()
        self.debug=False
        self._masks=[False]
        self.metadata=typeHintedDict()
        self.data = _ma_.masked_array([])
        self.filename = None
        self.column_headers = list()
        i=len(args) if len(args)<2 else 2
        handler=[None,self._init_single,self._init_double,self._init_many][i]
        self.mask=False
        self._setas._get_cols()
        if handler is not None:
            handler(*args,**kargs)
        self.metadata["Stoner.class"]=self.__class__.__name__
        if len(kargs)>0: # set public attributes from keywords
            myattrs=[]
            for x in dir(self):
                try:
                    if not (x.startswith("_") or callable(self.__getattr__(x))):
                        myattrs.append(x)
                except:
                    pass
            for k in kargs:
                if k in myattrs:
                    self.__setattr__(k,kargs[k])

    # Special Methods

    def _init_single(self,*args,**kargs):
        """Handles constructor with 1 arguement - called from __init__."""
        arg=args[0]
        if (isinstance(arg, string_types) or (isinstance(arg, bool) and not arg)):
                                    # Filename- load datafile
            self.load(filename=arg, **kargs)
        elif isinstance(arg, _np_.ndarray):
                                                # numpy.array - set data
            self.data = _ma_.masked_array(arg)
            self.column_headers = ['Column_{}'.format(x)
                                for x in range(_np_.shape(args[0])[1])]
        elif isinstance(arg, dict):  # Dictionary - use as metadata
            self.metadata = arg.copy()
        elif isinstance(arg, DataFile):
            for a in arg.__dict__:
                super(DataFile,self).__setattr__(a,copy.copy(arg.__getattribute__(a)))
            self.metadata = arg.metadata.copy()
            self.data = _ma_.masked_array(arg.data)
            self._setas=arg._setas
            self._setas.ref=self
        else:
            raise SyntaxError("No constructor for {}".format(type(arg)))
        self._setas.cols.update(self.setas._get_cols())

    def _init_double(self,*args,**kargs):
        """Two argument constructors handled here. Called form __init__"""
        (arg0,arg1)=args
        if isinstance(arg0, _np_.ndarray):
            self.data = _ma_.masked_array(arg0)
        elif isinstance(arg0, dict):
            self.metadata = arg0.copy()
        elif isinstance(arg0, str) and isinstance(arg1, str):
            self.load(arg0, arg1)
        if isinstance(arg1, _np_.ndarray):
            self.data = _ma_.masked_array(arg1)
        elif isinstance(arg1, dict):
            self.metadata = arg1.copy()

    def _init_many(self,*args,**kargs):
        """Handles more than two arguments to the constructor - called from init."""
        self.load(*args, **kargs)


    def __add__(self, other):
        """ Implements a + operator to concatenate rows of data.

        Args:
            other (numpy arra `Stoner.Core.DataFile` or a dictionary or a list):

        Note:
            * If other is a dictionary then the keys of the dictionary are passed to
            :py:meth:`find_col` to see if they match a column, in which case the
            corresponding value will be used for theat column in the new row.
            Columns which do not have a matching key will be set to NaN. If other has keys
            that are not found as columns in self, additional columns are added.
            * If other is a list, then the add method is called recursively for each element
            of the list.
            * Returns: A Datafile object with the rows of @a other appended
            to the rows of the current object.
            * If other is a 1D numopy array with the same number of
            elements as their are columns in @a self.data then the numpy
            array is treated as a new row of data If @a ither is a 2D numpy
            array then it is appended if it has the same number of
            columns and @a self.data."""
        newdata=self.clone
        return self.__add_core__(other,newdata)

    def __add_core__(self,other,newdata):
        """Implements the core work of adding other to self and modifying newdata.

        Args:
            other (DataFile,array,list): The data to be added
            newdata(DataFile): The instance to be modified

        Returns:
            A modified newdata
            """
        if isinstance(other, _np_.ndarray):
            if len(self.data) == 0:
                t = _np_.atleast_2d(other)
                c = t.shape[1]
                if len(self.column_headers) < c:
                    newdata.column_headers.extend(["Column_{}".format(x) for x in range(c - len(self.column_headers))])
                newdata.data = t
                ret=newdata
            elif len(_np_.shape(other)) == 1:
                                    # 1D array, so assume a single row of data
                if _np_.shape(other)[0] == _np_.shape(self.data)[1]:
                    newdata.data = _np_.append(self.data,
                                                _np_.atleast_2d(other), 0)
                    ret=newdata
                else:
                    ret=NotImplemented
            elif len(_np_.shape(other)) == 2 and _np_.shape(
                    other)[1] == _np_.shape(self.data)[1]:
                            # DataFile + array with correct number of columns
                newdata.data = _np_.append(self.data, other, 0)
                ret=newdata
            else:
                ret=NotImplemented
        elif isinstance(other, DataFile):  # Appending another DataFile
            new_data=_np_.ones((other.shape[0], self.shape[1]))*_np_.nan
            for i in range(self.shape[1]):
                column=self.column_headers[i]
                try:
                    new_data[:, i]=other.column(column)
                except KeyError:
                    pass
            newdata.metadata=copy.copy(self.metadata)
            newdata.data = _np_.append(self.data, new_data, axis=0)
            ret=newdata
        elif isinstance(other,dict): # This is a horrible mess that I'm not sure we ever use
            """
            added_row=False
            for k in other:
                try:
                    newdata.find_col(k)
                except KeyError:
                    newdata=newdata.clone
                    if len(newdata)==0:
                        added_row=True
                        cl=1
                    else:
                        cl=len(newdata)
                    newdata=newdata&_np_.ones(cl)*_np_.nan
                    newdata.column_headers[-1]=k
            new_data=_np_.nan*_np_.ones(len((newdata.column_headers)))
            for k in other:
                new_data[newdata.find_col(k)]=other[k]
            new_data=_np_.atleast_2d(new_data)
            newdata.data=_np_.append(newdata.data,new_data,0)
            if added_row:
                newdata.data=newdata.data[1:,:]
            ret=newdata"""
            ret=NotImplemented
        elif isinstance(other,list):
            for o in other:
                newdata=newdata+o
            ret=newdata
        else:
            ret=NotImplemented('Failed in DataFile')
        for attr in self.__dict__:
            if attr not in ("metadata","data","column_headers","mask") and not attr.startswith("_"):
                ret.__dict__[attr]=self.__dict__[attr]
        return ret

    def __and__(self, other):
        """Implements the & operator to concatenate columns of data in a :py:class:`DataFile` object.

        Args:
            other  (numpy array or :py:class:`DataFile`): Data to be added to this DataFile instance

        Returns:
            A :py:class:`DataFile` object with the columns of other con
        catenated as new columns at the end of the self object.

        Note:
            Whether other is a numopy array of :py:class:`DataFile`, it must
            have the same or fewer rows than the self object.
            The size of @a other is increased with zeros for the extra rows.
            If other is a 1D numpy array it is treated as a column vector.
            The new columns are given blank column headers, but the
            length of the :py:meth:`column_headers` is
            increased to match the actual number of columns.
        """
        #Prep the final DataFile
        newdata=self.clone
        return self.__and_core__(other,newdata)

    def __and_core__(self,other,newdata):
        """Implements the core of the & operator, returning data in newdata

        Args:
            other (array,DataFile): Data whose columns are to be added
            newdata (DataFile): instance of DataFile to be modified

        Returns:
            newdata: The modified DataFile (may be self or a clone of self depending
            on the operator's inplaceness)
        """

        if len(newdata.data.shape)<2:
            newdata.data=_np_.atleast_2d(newdata.data)

        #Get other to be a numpy masked array of data
        if isinstance(other,DataFile):
            newdata.metadata.update(other.metadata)
            newdata.column_headers.extend(other.column_headers)
            other=copy.copy(other.data)
        elif isinstance(other, _np_.ndarray):
            other=_ma_.array(copy.copy(other))
        else:
            newdata=NotImplemented


        if len(other.shape) != 2:  # 1D array, make it 2D column
            other = _np_.atleast_2d(other)
            other = other.T
        if _np_.product(self.data.shape)==0: #Special case no data yet
            newdata.data=other
        elif self.data.shape[0]==other.shape[0]:
            newdata.data=_np_.append(newdata.data,other,1)
        elif self.data.shape[0]<other.shape[0]: #Need to extend self.data
            extra_rows=other.shape[0]-self.data.shape[0]
            newdata.data=_np_.append(self.data,_np_.zeros((extra_rows,self.data.shape[1])),0)
            new_mask=newdata.mask
            new_mask[-extra_rows:,:]=True
            newdata.data=_np_.append(newdata.data,other,1)
            other_mask=_ma_.getmaskarray(other)
            new_mask=_np_.append(new_mask,other_mask,1)
            newdata.mask=new_mask
        elif other.shape[0] < self.data.shape[0]:
                # too few rows we can extend with zeros
            extra_rows=self.data.shape[0] - other.shape[0]
            other = _np_.append(other, _np_.zeros((extra_rows, other.shape[1])), 0)
            other_mask=_ma_.getmaskarray(other)
            other_mask[-extra_rows:,:]=True
            new_mask=newdata.mask
            new_mask=_np_.append(new_mask,other_mask,1)
            newdata.data=_np_.append(self.data,other,1)
            newdata.mask=new_mask
        if len(newdata.column_headers)<newdata.shape[1]:
            newdata.column_headers.extend(["Column "+str(i+len(newdata.column_headers)) for i in range(other.shape[1])])
        for attr in self.__dict__:
            if attr not in ("metadata","data","column_headers","mask") and not attr.startswith("_"):
                newdata.__dict__[attr]=self.__dict__[attr]
        return newdata

    def __contains__(self, item):
        """Operator function for membertship tests - used to check metadata contents.

        Args:
            item(string): name of metadata key

        Returns:
            iem in self.metadata"""
        return item in self.metadata

    def __delitem__(self,item):
        """Implements row or metadata deletion.

        Args:
            item (ingteger or string):  row index or name of metadata to delete"""
        if isinstance(item,str):
            del(self.metadata[item])
        else:
            self.del_rows(item)

    def __dir__(self):
        """Reeturns the attributes of the current object by augmenting the keys of self.__dict__ with the attributes that __getattr__ will handle.
        """
        attr=dir(type(self))
        attr.extend(list(self.__dict__.keys()))
        attr.extend(['column_headers','records', 'clone','subclasses', 'shape', 'mask', 'dict_records','setas'])
        col_check={"xcol":"x","xerr":"d","ycol":"y","yerr":"e","zcol":"z","zerr":"f"}
        for k in col_check:
            if "_setas" not in self.__dict__:
                break
            if k.startswith("x"):
                if k in self._setas.cols and self._setas.cols[k] is not None:
                    attr.append(col_check[k])
            else:
                if k in self._setas.cols and len(self._setas.cols[k])>0:
                    attr.append(col_check[k])
        return sorted(attr)

    def __file_dialog(self, mode):
        """Creates a file dialog box for loading or saving ~b DataFile objects.

        Args:
            mode (string): The mode of the file operation  'r' or 'w'

        Returns:
            A filename to be used for the file operation."""
        # Wildcard pattern to be used in file dialogs.

        descs={}
        patterns=self.patterns
        for p in patterns:
            descs[p]=self.__class__.__name__+" file"
        for c in self.subclasses:
            for p in (self.subclasses[c].patterns):
                if p in descs:
                    descs[p]+=", "+self.subclasses[c].__name__+" file"
                else:
                    descs[p]=self.subclasses[c].__name__+" file"

        patterns=[(descs[p],p) for p in sorted(descs.keys())]
        patterns.append(("All File","*.*"))

        if self.filename is not None:
            filename = os.path.basename(self.filename)
            dirname = os.path.dirname(self.filename)
        else:
            filename = ""
            dirname = ""
        if "r" in mode:
            mode="file"
        elif "w" in mode:
            mode="save"
        else:
            mode="directory"
        dlg = get_filedialog(what=mode, initialdir=dirname, initialfile=filename,filetypes=patterns)
        if len(dlg)!=0:
            self.filename = dlg
            return self.filename
        else:
            return None

    def __getattr__(self, name):
        """
        Called for :py:class:`DataFile`.x to handle some special pseudo attributes and otherwise to act as a shortcut for :py:meth:`column`.

        Args:
            name (string): The name of the attribute to be returned.

        Returns:
            the DataFile object in various forms

        Supported attributes:
        - records - return the DataFile data as a numpy structured
        array - i.e. rows of elements whose keys are column headings
        - clone - returns a deep copy of the current DataFile instance

        Otherwise the name parameter is tried as an argument to
        :py:meth:`DataFile.column` and the resultant column isreturned. If
        DataFile.column raises a KeyError this is remapped as an
        AttributeError.
       """

        easy={"clone":self._getattr_clone,
              "dict_records":self._getattr_dict_records,
              "dtype":self._getattr_dtype,
              "mask":self._getattr_mask,
              "T":self._getattr_by_columns,
              "records":self._getattr_records,
              "shape":self._getattr_shape,
              "subclasses":self._getattr_subclasses,
              "setas":self._getattr_setas,
              "column_headers":self.__getattr__column_headers
              }
        if name in easy:
            return easy[name]()
        elif name in ("x","y","z","d","e","f","u","v","w","r","q","p"):
            ret=self._getattr_col(name)
        elif name in dir(self):
            return super(DataFile,self).__getattribute__(name)
        else:
            ret=None
        if ret is not None:
            return ret
        if name in ("_setas",): # clearly not setup yet
            raise KeyError("Tried accessing setas before initialised")
        else:
            try:
                col=self._setas.find_col(name)
                return self.column(col)
            except (KeyError,IndexError):
                pass
        raise AttributeError("{} is not an attribute of DataFile nor a column name".format(name))

    def _getattr_by_columns(self):
        """Gets the current data transposed.
        """
        return self.data.T


    def _getattr_clone(self):
        """Gets a deep copy of the current DataFile.
        """
        c=self.__class__(copy.deepcopy(self))
        c.data=self.data.copy()
        c._setas=_setas()
        c.column_headers=self.column_headers
        c.setas(list(self.setas))
        for attr in self.__dict__:
            if attr not in ("metadata","data","column_headers") and not callable(self.__dict__[attr]):
                c.__dict__[attr]=self.__dict__[attr]
        return c

    def _getattr_col(self,name):
        """Get a column using the setas attribute."""
        col_check={"x":"xcol","d":"xerr","y":"ycol","e":"yerr","z":"zcol","f":"zerr","u":"ucol","v":"vcol","w":"wcol","q":"","p":"","r":""}
        col=col_check[name]
        if col=="" and self._setas.cols and "axes" in self._setas.cols: # inferred quick access columns for cartesian to polar transforms
            axes=int(self._setas.cols["axes"])
            if name=="r": # r in spherical or cylinderical co-ordinate systems
                m=[lambda d:None,
                   lambda d:None,
                   lambda d:_np_.sqrt(d.x**2+d.y**2),
                   lambda d:_np_.sqrt(d.x**2+d.y**2+d.z**2),
                   lambda d:_np_.sqrt(d.x**2+d.y**2+d.z**2),
                   lambda d:_np_.sqrt(d.u**2+d.v**2),
                   lambda d:_np_.sqrt(d.u**2+d.v**2+d.w**2)]
                ret=m[axes](self)
            elif name=="q": # theta in clyinderical or spherical co-ordiante systems
                m=[lambda d:None,
                   lambda d:None,
                   lambda d:_np_.arctan2(d.x,d.y),
                   lambda d:_np_.arctan2(d.x,d.y),
                   lambda d:_np_.arctan2(d.x,d.y),
                   lambda d:_np_.arctan2(d.u,d.v),
                   lambda d:_np_.arctan2(d.u,d.v)]
                ret=m[axes](self)
            elif name=="p": # phi is spherical co-ordinate systems
                m=[lambda d:None,
                   lambda d:None,
                   lambda d:None,
                   lambda d:_np_.arcsin(d.z),
                   lambda d:_np_.arsin(d.z),
                   lambda d:_np_.arcsin(d.w),
                   lambda d:_np_.arcsin(d.w)]
                ret=m[axes](self)


        elif col.startswith("x"):
            if self._setas.cols[col] is not None:
                ret= self.column(self._setas.cols[col])
            else:
                ret=None
        else:
            if len(self._setas.cols[col])>0:
                ret=self.column(self._setas.cols[col])
                if len(self._setas.cols[col])==1:
                    ret=ret[:,0]
            else:
                ret=None
        return ret

    def __getattr__column_headers(self):
        """Pass through to the setas attribute."""
        return self._setas.column_headers

    def _getattr_dict_records(self):
        """Return the data as a dictionary of single columns with column headers for the keys.
        """
        return _np_.array([dict(zip(self.column_headers, r)) for r in self.rows()])

    def _getattr_dtype(self):
        """Return the _np_ dtype attribute of the data
        """
        return self.data.dtype

    def _getattr_mask(self):
        """Returns the mask of the data array.
        """
        self.data.mask=_ma_.getmaskarray(self.data)
        return self.data.mask

    def _getattr_records(self):
        """Returns the data as a _np_ structured data array. If columns names are duplicated then they
        are made unique.
        """
        f=self.data.flags
        if not f["C_CONTIGUOUS"] and not f["F_CONTIGUOUS"]: # We need our data to be contiguous before we try a records view
            self.data=self.data.copy()
        ch=copy.copy(self.column_headers) # renoved duplicated column headers for structured record
        for i in range(len(ch)):
            header=ch[i]
            j=0
            while ch[i] in ch[i+1:] or ch[i] in ch[0:i]:
                j=j+1
                ch[i]="{}_{}".format(header,j)
        dtype = [(x, self.dtype) for x in ch]
        return self.data.view(dtype=dtype).reshape(len(self))

    def _getattr_shape(self):
        """Pass through the numpy shape attribute of the data.
        """
        return self.data.shape

    def _getattr_setas(self):
        """Get the list of column assignments."""
        self._setas.ref=self #Reset the reference to point to ourselves
        return self._setas

    def _getattr_subclasses(self):
        """Return a list of all in memory subclasses of this DataFile.
        """
        subclasses={x:x.priority for x in itersubclasses(DataFile)}
        ret=OrderedDict()
        ret["DataFile"]=DataFile
        for cls, priority in sorted(list(subclasses.items()), key=lambda c: c[1]):
            ret[cls.__name__]=cls
        return ret

    def __getitem__(self, name):
        """Called for DataFile[x] to return either a row or iterm of metadata.

        Args:
            name (string or slice or int): The name, slice or number of the part of the
            :py:class:`DataFile` to be returned.

        Returns:
            an item of metadata or row(s) of data.

        - If name is an integer then the corresponding single row will be returned
        - if name is a slice, then the corresponding rows of data will be returned.
        - If name is a string then the metadata dictionary item             with the correspondoing key will be returned.
        - If name is a numpy array then the corresponding rows of the data are returned.
        - If a tuple is supplied as the arguement then there are a number of possible behaviours.
            - If the first element of the tuple is a string, then it is assumed that it is the nth element of the named metadata is required.
            - Otherwise itis assumed that it is a particular element within a column determined by the second part of the tuple that is required.

        Examples:
            DataFile['Temp',5] would return the 6th element of the
            list of elements in the metadata called 'Temp', while

            DataFile[5,'Temp'] would return the 6th row of the data column
            called 'Temp'

            and DataFile[5,3] would return the 6th element of the
            4th column.
        """
        if isinstance(name, slice):
            indices = name.indices(len(self))
            name = range(*indices)
            d = self.data[name[0], :]
            d = _np_.atleast_2d(d)
            for x in range(1, len(name)):
                d = _np_.append(d, _np_.atleast_2d(self.data[x, :]), 0)
            return d
        elif isinstance(name, int):
            return self.data[name, :]
        elif isinstance(name, _np_.ndarray) and len(name.shape)==1:
            return self.data[name, :]
        elif isinstance(name, string_types) or isinstance(name,re._pattern_type):
             return self.__meta__(name)
        elif isinstance(name, tuple) and len(name) == 2:
            x, y = name
            if isinstance(x, str):
                return self[x][y]
            else:
                d = _np_.atleast_2d(self[x])
                y = self.find_col(y)
                r = d[:, y]
                if len(r) == 1:
                    r = r[0]
                return r
        else:
            raise TypeError("Key must be either numeric of string")

    def __getstate__(self):
        return {"data": self.data,  "column_headers":
            self.column_headers,  "metadata": self.metadata}

    def __iadd__(self, other):
        """ Implements a += operator to concatenate rows of data inplace.

        Args:
            other (numpy arra `Stoner.Core.DataFile` or a dictionary or a list):

        Note:
            * If other is a dictionary then the keys of the dictionary are passed to
            :py:meth:`find_col` to see if they match a column, in which case the
            corresponding value will be used for theat column in the new row.
            Columns which do not have a matching key will be set to NaN. If other has keys
            that are not found as columns in self, additional columns are added.
            * If other is a list, then the add method is called recursively for each element
            of the list.
            * Returns: A Datafile object with the rows of @a other appended
            to the rows of the current object.
            * If other is a 1D numopy array with the same number of
            elements as their are columns in @a self.data then the numpy
            array is treated as a new row of data If @a ither is a 2D numpy
            array then it is appended if it has the same number of
            columns and @a self.data."""
        newdata=self
        return self.__add_core__(other,newdata)

    def __iand__(self, other):
        """Implements the &= operator to concatenate columns of data in a :py:class:`DataFile` object.

        Args:
            other  (numpy array or :py:class:`DataFile`): Data to be added to this DataFile instance

        Returns:
            A :py:class:`DataFile` object with the columns of other con
        catenated as new columns at the end of the self object.

        Note:
            Whether other is a numopy array of :py:class:`DataFile`, it must
            have the same or fewer rows than the self object.
            The size of @a other is increased with zeros for the extra rows.
            If other is a 1D numpy array it is treated as a column vector.
            The new columns are given blank column headers, but the
            length of the :py:meth:`column_headers` is
            increased to match the actual number of columns.
        """
        newdata=self
        return self.__and_core__(other,newdata)

    def _load(self,filename,*args,**kargs):
        """Replace __parse_data with method that is more compatible with subclasses."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(self.filename,"r") as datafile:
            reader=csv.reader(datafile,dialect=_tab_delimited())
            row=next(reader)
            if row[0].strip()=="TDI Format 1.5":
                format=1.5
            elif row[0].strip()=="TDI Format=Text 1.0":
                format=1.0
            else:
                raise RuntimeError("Not a TDI File")
            col_headers_tmp=[x.strip() for x in row[1:]]
            datarow=0
            metadatarow=0
            cols=0
            for row in reader: # Now read through the metadata columns
                if len(row)>1 and row[1].strip()!="":
                    datarow+=1
                    still_data=True
                else:
                    still_data=False
                if row[0].strip()=="": #end of metadata:
                    break
                else:
                    cols=max(cols,len(row))
                    metadatarow+=1
                    md=row[0].split('=')
                    val="=".join(md[1:])
                    self.metadata[md[0].strip()]=val.strip()
        #End of metadata reading, close filke and reopen to read data
        if still_data: # data extends beyond metada - read with genfromtxt
            self.data=_np_.genfromtxt(self.filename,skip_header=1,usemask=True,delimiter="\t",usecols=range(1,cols),invalid_raise=False,comments="\0")
        elif datarow>0: # some data less than metadata
            footer=metadatarow-datarow
            self.data=_np_.genfromtxt(self.filename,skip_header=1,skip_footer=footer,usemask=True,delimiter="\t",comments="\0", usecols=range(1,cols))
        else:
            self.data=_np_.atleast_2d(_np_.array([]))
        if len(self.data.shape)>=2 and self.data.shape[1]>0:
            self.column_headers=["Column "+str(i) for i in range(self.data.shape[1])]
            for i in range(len(col_headers_tmp)):
                self.column_headers[i]=col_headers_tmp[i]


    def __len__(self):
        """Return the length of the data.

        Returns: Returns the number of rows of data
                """
        return _np_.shape(self.data)[0]

    def __lshift__(self, other):
        """Overird the left shift << operator for a string or an iterable object to import using the :py:meth:`__read_iterable` function.

        Args:
            other (string or iterable object): Used to source the DataFile object

        Returns:
            A new DataFile object

        TODO:
            Make code work better with streams
        """
        newdata=DataFile()
        if isinstance(other, str):
            lines=itertools.imap(lambda x:x,  other.splitlines())
            newdata.__read_iterable(lines)
        elif isinstance(other, Iterable):
            newdata.__read_iterable(other)
        return self.__class__(newdata)

    def __meta__(self, ky):
        """Returns specific items of  metadata.

        This is equivalent to doing DataFile.metadata[key]

        Args:
            ky (string): The name of the metadata item to be returned.

        Returns:
            Returns the item of metadata.

        Note:
           If key is not an exact match for an item of metadata,
            then a regular expression match is carried out.
            """
        if isinstance(ky, string_types):  # Ok we go at it with a string
            if str(ky) in self.metadata:
                ret = self.metadata[str(ky)]
            else:
                test = re.compile(ky)
                ret = self.__regexp_meta__(test)
        elif isinstance(ky,re._pattern_type):
            ret = self.__regexp_meta__(ky)
        else:
            raise TypeError("Only strings and regular expressions  are supported as search keys for metadata")
        return ret


    def __parse_metadata(self, key, value):
        """Parse the metadata string, removing the type hints into a separate dictionary from the metadata.

        Args:
            key (string): The name of the metadata parameter to be written,
        possibly including a type hinting string.
            value (any): The value of the item of metadata.

        Returns:
            Nothing, but the current instance's metadata is changed.

        Note:
            Uses the typehint to set the type correctly in the dictionary

            All the clever work of managing the typehinting is done in the
        metadata dictionary object now.
        """
        self.metadata[key] = value

    def __read_iterable(self, reader):
        """Internal method to read a string representation of py:class:`DataFile` in line by line."""

        if "next" in dir(reader):
            readline=reader.next
        elif "readline" in dir(reader):
            readline=reader.readline
        else:
            raise AttributeError("No method to read a line in {}".format(reader))
        row=readline().split('\t')
        if row[0].strip()=="TDI Format 1.5":
            format=1.5
        elif row[0].strip()=="TDI Format=Text 1.0":
            format=1.0
        else:
            raise RuntimeError("Not a TDI File")
        col_headers_tmp=[x.strip() for x in row[1:]]
        cols=len(col_headers_tmp)
        self._setas=_setas("."*cols)
        self.data=_ma_.masked_array([])
        for r in reader:
            if r.strip()=="": # Blank line
                continue
            row=r.rstrip().split('\t')
            cols=max(cols, len(row)-1)
            if row[0].strip()!='':
                md=row[0].split('=')
                if len(md)==2:
                    md[1]="=".join(md[1:])
                elif len(md)<=1:
                    md.extend(['',''])

                if format==1.5:
                    self.metadata[md[0].strip()]=md[1].strip()
                elif format==1.0:
                    self.metadata[md[0].strip()]=self.metadata.string_to_type(md[1].strip())
            if len(row)<2:
                continue
            self.data=_np_.append(self.data, self._conv_float(row[1:]))
        self.data=_np_.reshape(self.data, (-1, cols))
        self.column_headers=["Column "+str(i) for i in range(cols)]
        for i in range(len(col_headers_tmp)):
            self.column_headers[i]=col_headers_tmp[i]


    def __reduce_ex__(self, p):
        return (DataFile, (), self.__getstate__())

    def __regexp_meta__(self,test):
        """Do a regular expression search for all meta data items.

        Args:
            test (compiled regular expression): Regular expression to test against meta data key names

        Returns:
            Either a single metadata item or a dictionary of metadata items
        """
        possible = [x for x in self.metadata if test.search(x)]
        if len(possible) == 0:
            raise KeyError("No metadata with keyname: " + str(test))
        elif len(possible) == 1:
            ret = self.metadata[possible[0]]
        else:
            d = dict()
            for p in possible:
                d[p] = self.metadata[p]
            ret = d
        return ret

    def __repr__(self):
        """Outputs the :py:class:`DataFile` object in TDI format.

        This allows one to print any :py:class:`DataFile` to a stream based
        object andgenerate a reasonable textual representation of the data.shape

                Returns:
                    self in a textual format. """
        return self.__repr_core__(256)

    def __repr_core__(self,shorten=1000):
        """Actuall do the repr work, but allow for a shorten parameter to
        save printing big files out to disc."""

        outp = "TDI Format 1.5\t" + "\t".join(self.column_headers)+"\n"
        m = len(self.metadata)
        self.data=_ma_.masked_array(_np_.atleast_2d(self.data))
        r = _np_.shape(self.data)[0]
        md = [self.metadata.export(x) for x in sorted(self.metadata)]
        for x in range(min(r, m)):
            outp = outp + md[x] + "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
        if m > r:  # More metadata
            for x in range(r, m):
                outp = outp + md[x] + "\n"
        elif r > m:  # More data than metadata
            if shorten is not None and shorten and r-m>shorten:
                for x in range(m,m+shorten-100):
                    outp += "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
                outp+="... {} lines skipped...\n".format(r-m-shorten+100)
                for x in range(-100,-1):
                    outp += "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
            else:
                for x in range(m, r):
                    outp = outp + "\t" + "\t".join([str(y) for y in self.data[x].filled()])+ "\n"
        return outp


    def __search_index(self,xcol,value,accuracy):
        """Helper for the search method that returns an array of booleans for indexing matching rows."""
        x=self.find_col(xcol)
        if isinstance(value,(int,float)):
            ix=_np_.less_equal(_np_.abs(self.data[:,x]-value),accuracy)
        elif isinstance(value,tuple) and len(value)==2:
            (l,u)=(min(value),max(value))
            delta=u-l+2*accuracy
            ix=_np_.less_equal(_np_.abs(self.data[:,x]-l-accuracy),delta)
        elif isinstance(value,(list,_np_.ndarray)):
            ix=_np_.zeros(len(self),dtype=bool)
            for v in value:
                ix=_np_.logical_or(ix,self.__search_index(xcol,v))
        elif callable(value):
            ix=_np_.array([value(self.column(x)[i],self.data[i]) for i in range(len(self))],dtype=bool)
        else:
            raise RuntimeError("Unknown search value type {}".format(value))
        return ix


    def __setattr__(self, name, value):
        """Handles attempts to set attributes not covered with class attribute variables.

        Args:
            name (string): Name of attribute to set. Details of possible attributes below:

        - mask Passes through to the mask attribute of self.data (which is a numpy masked array). Also handles the case where you pass a callable object to nask where we pass each row to the function and use the return reult as the mask
        - data Ensures that the :py:attr:`data` attribute is always a :py:class:`numpy.ma.maskedarray`
        """

        easy={"nask":self.__setattr_mask,
              "data":self.__setattr_data,
              "T":self.__setattr_T,
              "setas":self.__setattr_setas,
              "column_headers":self.__setattr_column_headers}

        if name in easy:
            easy[name](value)
        elif len(name)==1 and name in "xyzuvwdef" and len(self.setas[name])!=0:
            self.__setattr_col(name,value)
        else:
            super(DataFile,self).__setattr__(name,value)

    def __setattr_mask(self,value):
        """Set the mask attribute by setting the data.mask."""
        if callable(value):
            self._set_mask(value, invert=False)
        else:
            self.data.mask=value

    def __setattr_data(self,value):
        """Set the data attribute, but force it through numpy.ma.masked_array first."""
        self.__dict__["data"]=_ma_.masked_array(value)
        self._setas.shape=self.__dict__["data"].shape

    def __setattr_T(self,value):
        """Write directly to the transposed data."""
        self.data.T=value

    def __setattr_setas(self,value):
        """Sets a new setas assignment by calling the setas object."""
        self._setas(value)

    def __setattr_column_headers(self,value):
        """Write the column_headers attribute (delagated to the setas object)."""
        self._setas.column_headers=value

    def __setattr_col(self,name,value):
        """Attempts to either assign data columns if set up, or setas setting.

        Args:
            name (length 1 string): Column type to work with (one of x,y,z,u,v,w,d,e or f)
            value (nd array or column index): If an ndarray and the column type corresponding to *name* is set up,
                then overwrite the column(s) of data with this new data. If an index type, then set the corresponding setas
                assignment to these columns.
        """

        if isinstance(value,_np_.ndarray):
            value=_np_.atleast_2d(value)
            if value.shape[0]==self.data.shape[0]:
                pass
            elif value.shape[1]==self.data.shape[0]:
                value=value.T
            else:
                raise RuntimeErrpr("Value to be assigned to data columns is the wrong shape!")
            for i,ix in enumerate(self.find_col(self.setas[name],force_list=True)):
                self.data[:,ix]=value[:,i]
        elif isinstance(value,indices):
            self._set_setas({name:value})

    def __setitem__(self, name, value):
        """Called for :py:class:`DataFile`[name ] = value to write mewtadata entries.

        Args:
            name (string): The string key used to access the metadata
            value (any): The value to be written into the metadata. Currently bool, int, float and string values are correctly handled. Everythign else is treated as a string.

        Returns:
            Nothing."""
        self.metadata[name] = value

    def __setstate__(self, state):
        """Internal function for pickling."""
        self.data = _ma_.masked_array(state["data"])
        self.column_headers = state["column_headers"]
        self.metadata = state["metadata"]

    #Private Functions

    def _set_mask(self, func, invert=False,  cumulative=False, col=0):
        """Applies func to each row in self.data and uses the result to set the mask for the row.

        Args:
            func (callable): A Callable object of the form lambda x:True where x is a row of data (numpy
            invert (bool): Optionally invert te reult of the func test so that it unmasks data instead
            cumulative (bool): if tru, then an unmask value doesn't unmask the data, it just leaves it as it is."""

        i=-1
        args=len(_inspect_.getargs(func.__code__)[0])
        for r in self.rows():
            i+=1
            if args==2:
                t=func(r[col], r)
            else:
                t=func(r)
            if isinstance(t, bool) or isinstance(t, _np_.bool_):
                if t^invert:
                    self.data[i]=_ma_.masked
                elif not cumulative:
                    self.data[i]=self.data.data[i]
            else:
                for j in range(min(len(t), _np_.shape(self.data)[1])):
                    if t[j]^invert:
                        self.data[i, j]=_ma_.masked
                    elif not cumulative:
                        self.data[i, j]=self.data.data[i, j]

    def __str__(self):
        """Provides an implementation for str(DataFile) that does not shorten the output."""
        return self.__repr_core__(False)

    def _push_mask(self, mask=None):
        """Copy the current data mask to a temporary store and replace it with a new mask if supplied.

        Args:
            mask (:py:class:numpy.array of bool or bool or None):
                The new data mask to apply (defaults to None = unmask the data

        Returns:
            Nothing"""
        self._masks.append(self.mask)
        if mask is None:
            self.data.mask=False
        else:
            self.mask=mask

    def _pop_mask(self):
        """Replaces the mask on the data with the last one stored by _push_mask().

        Returns:
            Nothing"""
        self.mask=False
        self.mask=self._masks.pop()
        if len(self._masks)==0:
            self._masks=[False]

    #   PUBLIC METHODS

    def add_column(self, column_data, column_header=None, index=None, func_args=None, replace=False):
        """Appends a column of data or inserts a column to a datafile instance.

        Args:
            column_data (:py:class:`numpy.array` or list or callable): Data to append or insert or a callable function that will generate new data

        Keyword Arguments:
            column_header (string): The text to set the column header to,
                if not supplied then defaults to 'col#'
            index (int or string): The  index (numeric or string) to insert (or replace) the data
            func_args (dict): If column_data is a callable object, then this argument
                can be used to supply a dictionary of function arguments to the callable object.
            replace (bool): Replace the data or insert the data (default)

        Returns:
            A :py:class:`DataFile` instance with the additonal column inserted.

        Note:
            Like most :py:class:`DataFile` methods, this method operates in-place in that it also modifies
            the original DataFile Instance as well as returning it."""
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
        if isinstance(column_data, _np_.ndarray):
            if len(_np_.shape(column_data)) != 1:
                raise ValueError('Column data must be 1 dimensional')
            else:
                _np__data = column_data
        elif callable(column_data):
            if isinstance(func_args, dict):
                new_data = [column_data(x, **func_args) for x in self]
            else:
                new_data = [column_data(x) for x in self]
            _np__data = _np_.array(new_data)
        elif isinstance(column_data,  list):
            _np__data=_np_.array(column_data)
        else:
            return NotImplemented
        #Sort out the sizes of the arrays
        cl=len(_np__data)
        if len(self.data.shape)==2:
            (dr,dc)=self.data.shape
        elif len(self.data.shape)==1:
            self.data=_np_.atleast_2d(self.data).T
            (dr,dc)=self.data.shape
        elif len(self.data.shape)==0:
            self.data=_np_.array([[]])
            (dr,dc)=(0,0)
        if cl>dr and dc*dr>0:
            self.data=_np_.append(self.data,_np_.zeros((cl-dr,dc)),0)
        elif cl<dr:
            _np__data=_np_.append(_np__data,_np_.zeros(dr-cl))
        if replace:
            self.data[:, index] = _np__data
        else:
            if dc*dr == 0:
                self.data=_ma_.masked_array(_np_.transpose(_np_.atleast_2d(_np__data)))
            else:
                self.data = _ma_.masked_array(_np_.insert(self.data, index, _np__data, 1))
        return self

    def column(self, col):
        """Extracts one or more columns of data from the datafile by name, partial name, regular expression or numeric index.

        Args:
            col (int, string, list or re): is the column index as defined for :py:meth:`DataFile.find_col`

        Returns:
            one or more columns of data as a :py:class:`numpy.ndarray`."""
        return self.data[:, self.find_col(col)]

    def columns(self):
        """Generator method that will iterate over the columns of data int he datafile.

        Yields:
            Returns the next column of data."""
        for col in range(self.data.shape[1]):
            yield self.column(col)

    def del_column(self, col=None,duplicates=False):
        """Deletes a column from the current :py:class:`DataFile` object.

        Args:
            col (int, string, list or re): is the column index as defined for :py:meth:`DataFile.find_col` to the column to be deleted

        Keyword Arguments:
            duplicates (bool): (default False) look for duplicated columns

        Returns:
            The :py:class:`DataFile` object with the column deleted.

        Note:
            - If duplicates is True and col is None then all duplicate columns are removed,
            - if col is not None and duplicates is True then all duplicates of the specified column are removed.
            - If duplicates is False then *col* must not be None otherwise a RuntimeError is raised.
            - If col is a list (duplicates should not be None) then the all the matching columns are found.
            - If col is None and duplicates is None, then all columns with at least one elelemtn masked
                    will be deleted
            """

        if duplicates:
            ch=self.column_headers
            dups=[]
            if col is None:
                for i in range(len(ch)):
                    if ch[i] in ch[i+1:]:
                        dups.append(ch.index(ch[i],i+1))
            else:
                col=ch[self.find_col(col)]
                i=ch.index(col)
                while True:
                    try:
                        i=ch.index(col,i+1)
                        dups.append(i)
                    except ValueError:
                        break
            return self.del_column(dups,duplicates=False)
        else:
            if col is None:
                self.data=_ma_.mask_cols(self.data)
                t=_ma_.masked_array(self.column_headers)
                t.mask=self.mask[0]
                self.column_headers=list(_ma_.compressed(t))
                self.data=_ma_.compress_cols(self.data)
            else:
                c = self.find_col(col)
                self.data = _ma_.masked_array(_np_.delete(self.data, c, 1), mask=_np_.delete(self.data.mask, c, 1))
                if isinstance(c, list):
                    c.sort(reverse=True)
                else:
                    c = [c]
                for col in c:
                    del self.column_headers[col]
            return self

    def del_rows(self, col=None, val=None,invert=False):
        """Searchs in the numerica data for the lines that match and deletes the corresponding rows.

        Args:
            col (list,slice,int,string, re or None): Column containg values to search for.
            val (float or callable): Specifies rows to delete. Maybe:

                - None - in which case the *col* argument is used to identify rows to be deleted,
                - a float in which case rows whose columncol = val are deleted
                - or a function - in which case rows where the function evaluates to be true are deleted.
                - a tuple, in which case rows where column col takes value between the minium and maximum of the tuple
                  are deleted.

        Keyword Arguments:
            invert (bool): Specifies whether to invert the logic of the test to delete a row. If True, keep the rows
                that would have been deleted otherwise.

        Returns:
            The current object

        Note:
            If col is None, then all rows with masked data are deleted

            If val is a function it should take two arguments - a float and a
            list. The float is the value of the current row that corresponds to column col abd the second
            argument is the current row.

        TODO:
            Implement val is a tuple for deletinging in a range of values.
            """
        if col is None:
            self.data=_ma_.mask_rows(self.data)
            self.data=_ma_.compress_rows(self.data)
        else:
            if isinstance(col, slice) and val is None:
                indices = col.indices(len(self))
                col -= range(*indices)
            if isinstance(col, list) and val is None and not invert:
                col.sort(reverse=True)
                for c in col:
                    self.del_rows(c)
            elif isinstance(col, list) and val is None and invert:
                for i in range(len(self)-1,-1,-1):
                    if i not in col:
                        self.del_rows(i)
            elif isinstance(col,  int) and val is None and not invert:
                self.data = _np_.delete(self.data, col, 0)
            elif isinstance(col,  int) and val is None and invert:
                self.del_rows([c],invert=invert)
            else:
                col = self.find_col(col)
                d = self.column(col)
                if callable(val):
                    rows = _np_.nonzero([(bool(val(x[col], x) & bool(x[col] is not _ma_.masked))!=invert)  for x in self])[0]
                elif isinstance(val, float):
                    rows = _np_.nonzero([bool(x == val)!=invert for x in d])[0]
                elif isinstance(val,Iterable) and len(val)==2:
                    (upper,lower)=(max(list(val)),min(list(val)))
                    rows= _np_.nonzero([bool(lower<=x<=upper)!=invert for x in d])[0]
                else:
                    raise SyntaxError("If val is specified it must be a float,callable, or iterable object of length 2")
                self.data = _ma_.masked_array(_np_.delete(self.data, rows, 0), mask=_np_.delete(self.data.mask, rows, 0))
        return self

    def dir(self, pattern=None):
        """ Return a list of keys in the metadata, filtering wiht a regular expression if necessary.

        Keyword Arguments:
            pattern (string or re): is a regular expression or None to list all keys

        Returns:
            Returns a list of metadata keys."""
        if pattern is None:
            return list(self.metadata.keys())
        else:
            if isinstance(pattern,re._pattern_type):
                test=pattern
            else:
                test = re.compile(pattern)
            possible = [x for x in self.metadata.keys() if test.search(x)]
            return possible

    def filter(self,func=None,cols=None,reset=True):
        """Sets the mask on rows of data by evaluating a function for each row.

        Args:
            func (callable): is a callable object that should take a single list as a p[arameter representing one row.
            cols (list): a list of column indices that are used to form the list of values passed to func.
            reset (bool): determines whether the mask is reset before doing the filter (otherwise rows already masked out will be ignored in the filter (so the filter is logically or'd)) The default value of None results in a complete row being passed into func.

        Returns:
            The current object with the mask set
        """
        if cols is None:
            cols=range(self.data.shape[1])
        cols=[self.find_col(c) for c in cols]
        self.data.mask=_ma_.getmaskarray(self.data)
        i=0
        if reset: self.data.mask=False
        for r in self.rows():
            self.data.mask[i,:]=not func(r[cols])
            i=i+1
        return self

    def find_col(self, col,force_list=False):
        """Indexes the column headers in order to locate a column of data.shape.

        Indexing can be by supplying an integer, a string, a regular experssion, a slice or a list of any of the above.

        -   Integer indices are simply checked to ensure that they are in range
        -   String indices are first checked for an exact match against a column header
            if that fails they are then compiled to a regular expression and the first
            match to a column header is taken.
        -   A regular expression index is simply matched against the column headers and the
            first match found is taken. This allows additional regular expression options
            such as case insensitivity.
        -   A slice index is converted to a list of integers and processed as below
        -   A list index returns the results of feading each item in the list at :py:meth:`find_col`
            in turn.

        Args:
            col (int, a string, a re, a slice or a list):  Which column(s) to retuirn indices for.

        Keyword Arguments:
            force_list (bool): Force the output always to be a list. Mainly for internal use only

        Returns:
            The matching column index as an integer or a KeyError
        """
        return self._setas.find_col(col,force_list)

    def get(self, item):
        """A wrapper around __get_item__ that handles missing keys by returning None.

        This is useful for the :py:class:`Stoner.Folder.DataFolder` class.

        Args:
            item (string): A string representing the metadata keyname

        Returns:
            self.metadata[item] or None if item not in self.metadata"""
        try:
            return self[item]
        except KeyError:
            return None

    def get_filename(self, mode):
        """Forces the user to choose a new filename using a system dialog box.

        Args:
            mode (string): The mode of file operation to be used when calling the dialog box

        Returns:
            The new filename

        Note:
            The filename attribute of the current instance is updated by this method as well.

        """
        self.filename = self.__file_dialog(mode)
        return self.filename

    def insert_rows(self, row, new_data):
        """Insert new_data into the data array at position row. This is a wrapper for numpy.insert.

        Args:
            row (int):  Data row to insert into
            new_data (numpy array): An array with an equal number of columns as the main data array containing the new row(s) of data to insert

        Returns:
            A copy of the modified DataFile object"""
        self.data=_np_.insert(self.data, row,  new_data, 0)
        return self

    def keys(self):
        """An alias for :py:meth:`DataFile.dir(None)` .

        Returns:
            a list of all the keys in the metadata dictionary"""
        return self.dir(None)

    def load(self, filename=None, auto_load=True,  filetype=None,  *args, **kargs):
        """Loads the :py:class:`DataFile` in from disc guessing a better subclass if necessary.

        Args:
            filename (string or None): path to file to load

        Keyword Arguments:
            auto_load (bool): If True (default) then the load routine tries all the subclasses of :py:class:`DataFile` in turn to load the file
            filetype (:py:class:`DataFile`): If not none then tries using filetype as the loader

        Returns:
            A copy of the loaded instance

        Note:
            Possible subclasses to try and load from are identified at run time using the speciall :py:attr:`DataFile.subclasses` attribute.

            Some subclasses can be found in the :py:mod:`Stoner.FileFormats` module.

            Each subclass is scanned in turn for a class attribute priority which governs the order in which they are tried. Subclasses which can
            make an early positive determination that a file has the correct format can have higher priority levels. Classes should return
            a suitable expcetion if they fail to load the file.

            If not class can load a file successfully then a RunttimeError exception is raised.
            """

        if filename is None or (isinstance(filename, bool) and not filename):
            filename = self.__file_dialog('r')
        else:
            self.filename = filename

        if not path.exists(self.filename):
            raise IOError("Cannot find {} to load".format(self.filename))
        cls=self.__class__
        failed=True
        if auto_load: # We're going to try every subclass we can
            for cls in self.subclasses.values():
                if self.debug:
                    print(cls.__name__)
                try:
                    test=cls()
                    test._load(self.filename, auto_load=False)
                    failed=False
                    self["Loaded as"]=cls.__name__
                    self._setas=test._setas
                    self._setas.ref=self
                    break
                except Exception as e:
                    continue
            else:
                raise IOError("Ran out of subclasses to try and load as.")
        else:
            if filetype is None:
                test=cls()
                test._load(self.filename)
                self["Loaded as"]=cls.__name__
                self._setas=test._setas
                self._setas.ref=self
                failed=False
            elif issublcass(filetype,DataFile):
                test=filetype(filename)
                self["Loaded as"]=filetype.__name__
                self._setas=test._setas
                self._setas.ref=self
                failed=False
        if failed:
            raise SyntaxError("Failed to load file")
        else:
            self.data=_ma_.masked_array(test.data)
            self.metadata.update(test.metadata)
            self.column_headers=test.column_headers
            self._setas=test._setas
            self._setas.ref=self
        return self

    def rename(self, old_col, new_col):
        """Renames columns without changing the underlying data.

        Args:
            old_col (string, int, re):  Old column index or name (using standard rules)
            new_col (string): New name of column

        Returns:
            A copy of self
        """

        old_col=self.find_col(old_col)
        self.column_headers[old_col]=new_col
        return self

    def reorder_columns(self, cols, headers_too=True):
        """Construct a new data array from the original data by assembling the columns in the order given.

        Args:
            cols (list of column indices): (referred to the oriignal
                data set) from which to assemble the new data set
            headers_too (bool): Reorder the column headers in the same
                way as the data (defaults to True)

        Returns:
            A copy of the modified DataFile object"""
        if headers_too:
            self.column_headers = [self.column_headers[self.find_col(x)]
                                                            for x in cols]

        newdata = _np_.atleast_2d(self.data[:, self.find_col(cols.pop(0))])
        for col in cols:
            newdata = _np_.append(newdata, _np_.atleast_2d(self.data[:,
                                                self.find_col(col)]), axis=0)
        self.data = _ma_.masked_array(_np_.transpose(newdata))
        return self

    def rows(self):
        """Generator method that will iterate over rows of data

        Yields:
            Returns the next row of data"""
        r = _np_.shape(self.data)[0]
        for row in range(r):
            yield self.data[row]


    def save(self, filename=None):
        """Saves a string representation of the current DataFile object into the file 'filename'.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If f
                ilename is False then a file dialog is forced.

        Returns:
            The current object
                """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):
            # now go and ask for one
            filename = self.__file_dialog('w')
        header=["TDI Format 1.5"]
        header.extend(self.column_headers)
        header="\t".join(header)
        mdkeys=sorted(self.metadata)
        if len(mdkeys)>len(self):
            mdremains=mdkeys[len(self):]
            mdkeys=mdkeys[0:len(self)]
        else:
            mdremains=[]
        mdtext=_np_.array([self.metadata.export(k) for k in mdkeys])
        if len(mdtext)<len(self):
            mdtext=_np_.append(mdtext,_np_.zeros(len(self)-len(mdtext),dtype=str))
        data_out=_np_.column_stack([mdtext,self.data])
        fmt=["%s"]*data_out.shape[1]
        with open(filename, 'w') as f:
            _np_.savetxt(f,data_out,fmt=fmt,header=header,delimiter="\t",comments="")
            for k in mdremains:
                f.write(self.metadata.export(k)+"\n")

        self.filename = filename
        return self

    def search(self, xcol,value,columns=None,accuracy=0.0):
        """Searches in the numerica data part of the file for lines that match and returns  the corresponding rows.

        Args:
            xcol (int,string.re) is a Search Column Index
            value (float, tuple, list or callable): Value to look for

        Keyword Arguments:
            columns (index or array of indices or None (default)): columns of data to return - none represents all columns.
            accuracy (float): Uncertainty to accept when testing equalities

        Returns:
            numpy array of matching rows or column values depending on the arguements.

        Note:
            The value is interpreted as follows:

            - a float looks for an exact match
            - a list is a list of exact matches
            - a tuple should contain a (min,max) value.
            - A callable object should have accept a float and an array representing the value of
              the search col for the the current row and the entire row.


        """
        ix=self.__search_index(xcol,value,accuracy)
        if columns is None: #Get the whole slice
            data=self.data[ix,:]
        else:
            columns=self.find_col(columns)
            if not isinstance(columns,list):
                data=self.data[ix,columns]
            else:
                data=self.data[ix,columns[0]]
                for c in columns[1:]:
                    data=_np_.column_stack((data,self.data[ix,c]))
        return data

    def section(self,**kargs):
        """Assuming data has x,y or x,y,z co-ordinates, return data from a section of the parameter space.

        Keyword Arguments:
            x (float, tuple, list or callable): x values ,atch this condition are included inth e section
            y (float, tuple, list  or callable): y values ,atch this condition are included inth e section
            z (float, tuple,list  or callable): z values ,atch this condition are included inth e section
            r (callable): a function that takes a tuple (x,y,z) and returns True if the line is to be incluided in section

        Returns:
            A DataFile like object that includes only those lines from the original that match the section specification

        Internally this function is calling :py:meth:`DataFile.search` to pull out matching sections of the data array.
        To extract a 2D section of the parameter space orthogonal to one axis you just specify a condition on that axis. Specifying
        conditions on two axes will return a line of points along the third axis. The final keyword parameter allows you to select
        data points that lie in an arbitary plane or line. eg::

            d.section(r=lambda x,y,z:abs(2+3*x-2*y)<0.1 and z==2)

        would extract points along the line 2y=3x+2 (note the use of an < operator to avoid floating point rounding errors) where
        the z-co-ordinate is 2.
        """
        cols=self.setas._get_cols()
        tmp=self.clone
        xcol=cols["xcol"]
        ycol=cols["ycol"][0]
        zcol=cols["zcol"][0]

        if "accuracy" in kargs:
            accuracy=kargs["accuracy"]
        else:
            accuracy=0.0

        if "x" in kargs:
            tmp.data=tmp.search(xcol,kargs["x"],accuracy=accuracy)
        if "y" in kargs:
            tmp.data=tmp.search(ycol,kargs["y"],accuracy=accuracy)
        if "z" in kargs:
            tmp.data=tmp.search(zcol,kargs["z"],accuracy=accuracy)
        if "r" in kargs:
            func=lambda x,r:kargs["r"](r[xcol],r[ycol],r[zcol])
            tmp.data=tmp.search(0,func,accuracy=accuracy)
        return tmp

    def sort(self, order=None,reverse=False):
        """Sorts the data by column name. Sorts in place and returns a copy of the sorted data object for chaining methods.

        Keyword Arguments:
            order (column index or list of indices or callable function): Represent the sort order
                If order is a callable function then it should take a two tuple arguments and
                return +1,0,-1 depending on whether the first argument is bigger, equal or smaller.
            reverse (boolean): If true, the sorted array isreversed.

        Returns:
            A copy of the sorted object
        """

        if order is None:
            order=list(range(len(self.column_headers)))
        recs=self.records
        if callable(order):
            d=sorted(recs,cmp=order)
        elif isinstance(order, list) or isinstance(order, tuple):
            order = [recs.dtype.names[self.find_col(x)] for x in order]
            d = _np_.sort(recs, order=order)
        else:
            order = [recs.dtype.names[self.find_col(order)]]
            d = _np_.sort(recs, order=order)
        if reverse:
            d=d[::-1]
        self.data = _ma_.masked_array(d.view(dtype=self.dtype).reshape(len(self), len(self.
                                                              column_headers)))
        return self

    def swap_column(self, swp, headers_too=True):
        """Swaps pairs of columns in the data.

        Useful for reordering data for idiot programs that expect columns in a fixed order.

        Args:
            swp  (tuple of list of tuples of two elements): Each
                element will be iused as a column index (using the normal rules
                for matching columns).  The two elements represent the two
                columns that are to be swapped.
            headers_too (bool): Indicates the column headers
                are swapped as well

        Returns:
            A copy of the modified DataFile objects

        Note:
            If swp is a list, then the function is called recursively on each
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

    def unique(self, col, return_index=False, return_inverse=False):
        """Return the unique values from the specified column - pass through for numpy.unique.

        Args:
            col (index): Column to look for unique values in

        Keyword Arguments:

        """
        return _np_.unique(self.column(col), return_index, return_inverse)


# Module level functions

def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls).

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
