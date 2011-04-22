#############################################
#
# Core object of the Stoner Package
#
# $Id: Core.py,v 1.13 2011/04/22 14:44:04 cvs Exp $
#
# $Log: Core.py,v $
# Revision 1.13  2011/04/22 14:44:04  cvs
# Add code to return data as a structured record and to to provide a DataFile.sort() method
#
# Revision 1.12  2011/03/02 14:56:20  cvs
# Colon missing from else command in search function (Line 591)
#
# Revision 1.11  2011/03/02 13:16:52  cvs
# Fix buglet in DataFile.search
#
# Revision 1.10  2011/02/23 21:42:16  cvs
# Experimental code for displaying grid included
#
# Revision 1.9  2011/02/17 23:36:51  cvs
# Updated doxygen comment strings
#
# Revision 1.8  2011/02/14 17:00:03  cvs
# Updated documentation. More doxygen comments
#
# Revision 1.7  2011/02/13 15:51:08  cvs
# Merge in ma gui branch back to HEAD
#
# Revision 1.6  2011/02/12 22:12:43  cvs
# Added some doxygen compatible doc strings
#
# Revision 1.5  2011/02/11 00:00:58  cvs
# Add a DataFile.unique method
#
# Revision 1.4  2011/01/17 10:12:08  cvs
# Added code for mac implementation of wx.FileDialog()
#
# Revision 1.3  2011/01/13 22:30:56  cvs
# Enable chi^2 analysi where the parameters are varied and choi^2 calculated.
# Extra comments in the ini file
# Give DataFile some file dialog boxes
#
# Revision 1.2  2011/01/12 22:56:33  cvs
# Update documentation, add support for slices in some of the DataFile methods
#
# Revision 1.1  2011/01/08 20:30:02  cvs
# Complete splitting Stoner into a package with sub-packages - Core, Analysis and Plot.
# Setup some imports in __init__ so that import Stoner still gets all the subclasses - Gavin
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
        cols=max(len(dfile.column_headers), 4)
        rows=max(len(dfile), 20)
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
        
        
        


class DataFolder(object):
    
    #   CONSTANTS
    
    #   INITIALISATION
    
    def __init__(self, foldername):
        self.data = [];
        self.metadata = dict();
        self.foldername = foldername;
        self.__parseFolder(); 
        
    #   PRIVATE FUNCTIONS
    
    def __parseFolder(self):
        path="C:/Documents and Settings/pymn/workspace/Stonerlab/src/folder/run1"  # insert the path to the directory of interest
        dirList=os.listdir(path)
        for fname in dirList:
            print(fname)

#   PUBLIC METHODS
     
class DataFile(object): #Now a new style class so that we can use super()
    """@b Stoner.Core.DataFile is the base class object that represents a matrix of data, associated metadata and column headers.
    
    @b DataFile provides the mthods to load, save, add and delete data, index and slice data, manipulate metadata and column headings.
   
    Authors: Matt Newman, Chris Allen and Gavin Burnell    
    """
#   CONSTANTS

    __regexGetType = re.compile(r'([^\{]*)\{([^\}]*)\}') # Match the contents of the inner most{}
    __typeSignedInteger = "I64 I32 I16 I8"
    __typeUnsignedInteger="U64 U32 U16 U8"
    __typeInteger=__typeSignedInteger+__typeUnsignedInteger
    __typeFloat = "Extended Float Double Float Single Float"
    __typeBoolean = "Boolean"
    __typeString="String"
    defaultDumpLocation='C:\\dump.csv'
    
#   INITIALISATION

    def __init__(self, *args):
        """Constructor method
        
        various forms are recognised: 
        @li DataFile('filename',<optional filetype>,<args>)
        Creates the new DataFile object and then executes the \b DataFile.load method to load data from the given \a filename
        @li DataFile(array)
        Creates a new DataFile object and assigns the \a array to the \b DataFile.data attribute.
        @li DataFile(dictionary)
        Creates the new DataFile object, but initialises the metadata with \a dictionary
        @li  DataFile(array,dictionary), 
        Creates the new DataFile object and does the combination of the previous two forms.
        @li DataFile(DataFile)
        Creates the new DataFile object and initialises all data from the existing \DataFile instance. This on the face of it does the same as the assignment operator, 
        but is more useful when one or other of the DataFile objects is an instance of a sub-class of DataFile
        
        @param *args Variable number of arguments that match one of the definitions above
        @return A new instance of the DataFile class.
        """
        self.data = numpy.array([])
        self.metadata = dict()
        self.typehint = dict()
        self.filename = None
        self.column_headers=list()
        # Now check for arguments t the constructor
        if len(args)==1:
            if isinstance(args[0], str): # Filename- load datafile
                self.load(args[0])
            elif isinstance(args[0], numpy.ndarray): # numpy.array - set data
                self.data=args[0]
                self.column_headers=['Column'+str(x) for x in range(numpy.shape(args[0])[1])]
            elif isinstance(args[0], dict): # Dictionary - use as metadata
                self.metadata=args[0]
            elif isinstance(args[0], DataFile):
                self.metadata=args[0].metadata
                self.data=args[0].data
                self.typehint=args[0].typehint
                self.column_headers=args[0].column_headers
        elif len(args)==2: # 2 argument forms either array,dict or dict,array
            if isinstance(args[0], numpy.ndarray):
                self.data=args[0]
            elif isinstance(args[0], dict):
                self.metadata=args[0]
            elif isinstance(args[0], str) and isinstance(args[1], str):
                self.load(args[0], args[1])
            if isinstance(args[1], numpy.ndarray):
                self.data=args[1]
            elif isinstance(args[1], dict):
                self.metadata=args[1]
        elif len(args)>2:
            apply(self.load, args)
                
# Special Methods

    def __getattr__(self, name):
        """
        Called for \bDataFile.x to handle some special pseudo attributes
        
        @param name The name of the attribute to be returned. These include: records
        @return For Records, returns the data as an array of structures
        """
        if name=="records":
            dtype=[(x, numpy.float64) for x in self.column_headers]
            return self.data.view(dtype=dtype).reshape(len(self))

    def __getitem__(self, name): # called for DataFile[x] returns row x if x is integer, or metadata[x] if x is string
        """Called for \b DataFile[x] to return either a row or iterm of metadata
        
        @param name The name, slice or number of the part of the \b DataFile to be returned.
        @return an item of metadata or row(s) of data. \li If \a name is an integer then the corresponding single row will be rturned
        \li if \a name is a slice, then the corresponding rows of data will be returned. \li If \a name is a string then the metadata dictionary item with
        the correspondoing key will be returned.
        
        """
        if isinstance(name, slice):
            indices=name.indices(len(self))
            name=range(*indices)
            d=self.data[name[0], :]
            d=numpy.atleast_2d(d)
            for x in range(1, len(name)):
                d=numpy.append(d, numpy.atleast_2d(self.data[x, :]), 0)
            return d
        elif isinstance(name, int):
            return self.data[name,  :]
        elif isinstance(name, str):
            return self.meta(name)
        elif isinstance(name, tuple) and len(name)==2:
            x, y=name
            if isinstance(x, str):
                return self[x][y]
            else:
                d=numpy.atleast_2d(self[x])
                y=self.find_col(y)
                r=d[:, y]
                if len(r)==1:
                    r=r[0]
                return r
        else:
            raise TypeError("Key must be either numeric of string")

    def __setitem__(self, name, value):
        """Called for \DataFile[\em name ] = \em value to write mewtadata entries.
            @param name The string key used to access the metadata
            @param value The value to be written into the metadata. Currently bool, int, float and string values are correctly handled. Everythign else is treated as a string.
            @return Nothing."""
        if isinstance(value,bool):
            self.typehint[name]="Boolean"
        elif isinstance(value, int):
            self.typehint[name]="I32"
        elif isinstance(value, float):
            self.typehint[name]="Double Float"
        else:
            self.typehint[name]="String"
        self.metadata[name]=value
        
    def __add__(self, other):
        """ Implements a + operator to concatenate rows of data
                @param other Either a numpy array object or an instance of a \b DataFile object.
                @return A Datafile object with the rows of \a other appended to the rows of the current object.
                
                If \a other is a 1D numopy array with the same number of lements as their are columns in \a self.data then the numpy array is treated as a new row of data
                If \a ither is a 2D numpy array then it is appended if it has the same number of columns and \a self.data.
                
"""
        if isinstance(other, numpy.ndarray):
            if len(self.data)==0:
                t=numpy.atleast_2d(other)
                c=numpy.shape(t)[1]
                self.column_headers=map(lambda x:"Column_"+str(x), range(c))
                newdata=copy.deepcopy(self)
                newdata.data=t                
                return newdata
            elif len(numpy.shape(other))==1: # 1D array, so assume a single row of data
                if numpy.shape(other)[0]==numpy.shape(self.data)[1]:
                    newdata=copy.deepcopy(self)
                    newdata.data=numpy.append(self.data, numpy.atleast_2d(other), 0)
                    return newdata
                else:
                    return NotImplemented
            elif len(numpy.shape(other))==2 and numpy.shape(other)[1]==numpy.shape(self.data)[1]: # DataFile + array with correct number of columns
                newdata=copy.deepcopy(self)
                newdata.data=numpy.append(self.data, other, 0)
                return newdata
            else:
                return NotImplemented
        elif isinstance(other, DataFile): # Appending another DataFile
            if self.column_headers==other.column_headers:
                newdata=copy.deepcopy(other)
                for x in self.metadata:
                    newdata[x]=copy.copy(self[x])
                newdata.data=numpy.append(self.data, other.data, 0)
                return newdata
            else:
                return NotImplemented
        else:
            return NotImplemented
        
    def __and__(self, other):
        """Implements the & operator to concatenate columns of data in a \b Stoner.DataFile object.
       
        @param other Either a numpy array or \bStoner.DataFile object
        @return A \b Stoner.DataFile object with the columns of other concatenated as new columns at the end of the self object.
       
        Whether \a other is a numopy array of \b Stoner.DataFile, it must have the same or fewer rows than the self object. 
        The size of \a other is increased with zeros for the extra rows. 
        If \a other is a 1D numpy array it is treated as a column vector.
        The new columns are given blank column headers, but the length of the \b Stoner.DataFile.column_headers is 
        increased to match the actual number of columns.
        """
        if isinstance(other, numpy.ndarray):
            if len(other.shape)!=2: # 1D array, make it 2D column
                other=numpy.atleast_2d(other)
                other=other.T
            if other.shape[0]<=self.data.shape[0]: # DataFile + array with correct number of rows
                if other.shape[0]<self.data.shape[0]: # too few rows we can extend with zeros
                    other=numpy.append(other, numpy.zeros((self.data.shape[0]-other.shape[0], other.shape[1])), 0)
                newdata=copy.deepcopy(self)
                newdata.column_headers.extend(["" for x in range(other.shape[1])]) 
                newdata.data=numpy.append(self.data, other, 1)
                return newdata
            else:
                return NotImplemented
        elif isinstance(other, DataFile): # Appending another datafile
            if self.data.shape[0]==other.data.shape[0]:
                newdata=copy.deepcopy(other)
                newdata.column_headers=copy.copy(self.column_headers)
                newdata.column_headers.extend(self.column_headers)
                for x in self.metadata:
                    newdata[x]=copy.copy(self[x])
                newdata.data=numpy.append(self.data, other.data, 1)
                return newdata
            else:
                return NotImplemented
        else:
             return NotImplemented
        
    def __repr__(self): 
        """Outputs the \b Stoner.DataFile object in TDI format. This allows one to print any \b Stoner.DataFile to a stream based object andgenerate a reasonable textual representation of the data.shape
       @return \a self in a textual format. """
        outp="TDI Format 1.5"+"\t"+reduce(lambda x, y: str(x)+"\t"+str(y), self.column_headers)+"\n"
        m=len(self.metadata)
        (r, c)=numpy.shape(self.data)
        md=map(lambda x:str(x)+"{"+str(self.typehint[x])+"}="+str(self.metadata[x]), sorted(self.metadata))
        for x in range(min(r, m)):
            outp=outp+md[x]+"\t"+reduce(lambda z, y: str(z)+"\t"+str(y), self.data[x])+"\n"
        if m>r: # More metadata
            for x in range(r, m):
                    outp=outp+md[x]+"\n"
        elif r>m: # More data than metadata
            for x in range(m, r):
                    outp=outp+"\t"+reduce(lambda z, y: str(z)+"\t"+str(y), self.data[x])+"\n"
        return outp
        
    def __len__(self):
        return numpy.shape(self.data)[0]

#   PRIVATE FUNCTIONS

    def __file_dialog(self, mode):
        from enthought.pyface.api import FileDialog, OK
        # Wildcard pattern to be used in file dialogs.
        file_wildcard = "Text file (*.txt)|*.txt|Data file (*.dat)|*.dat|All files|*"
        
        if mode=="r":
            mode="open"
        elif mode=="w":
            mode="save"
            
        if self.filename is not None:
            filename=os.path.basename(self.filename)
            dirname=os.path.dirname(self.filename)
        else:
            filename=""
            dirname=""
        dlg = FileDialog(action=mode, wildcard=file_wildcard)
        dlg.open()
        if dlg.return_code==OK:
            self.filename=dlg.path
            return self.filename
        else:
            return None        
            
    def __parse_metadata(self, key, value):
        """Parse the metadata string, removing the type hints into a separate dictionary from the metadata
        
        Uses the typehint to set the type correctly in the dictionary
        """
        m=self.__regexGetType.search(key)
        k= m.group(1)
        t= m.group(2)
        if self.__typeInteger.find(t)>-1:
            value = int(value);
        elif self.__typeFloat.find(t)>-1:
            value = float(value);
        elif self.__typeBoolean.find(t)>-1:
            value = bool(value);
        else:
            value = str(value);
        self.metadata[k]=value
        self.typehint[k]=t

    def __parse_data(self):
        """Internal function to parse the tab deliminated text file
        """
        reader = csv.reader(open(self.filename, "rb"), delimiter='\t', quoting=csv.QUOTE_NONE)
        row=reader.next()
        assert row[0]=="TDI Format 1.5" # Bail out if not the correct format
        self.data=numpy.array([])
        headers = row[1:len(row)]
        maxcol=1
        for row in reader:
            if maxcol<len(row):
                    maxcol=len(row)
            if row[0].find('=')>-1:
                md=row[0].split('=')
                self.__parse_metadata(md[0], md[1])
            if (len(row[1:len(row)]) > 1) or len(row[1]) > 0:
                self.data=numpy.append(self.data, map(lambda x: float(x), row[1:]))
        else:
            shp=(-1, maxcol-1)
            self.data=numpy.reshape(self.data,  shp)
            self.column_headers=["" for x in range(self.data.shape[1])]
            self.column_headers[0:len(headers)]=headers
            
    def __parse_plain_data(self, header_line=3, data_line=7, data_delim=' ', header_delim=','):
        header_string=linecache.getline(self.filename, header_line)
        header_string=re.sub(r'["\n]', '', header_string)
        self.column_headers=map(lambda x: x.strip(),  header_string.split(header_delim))
        self.data=numpy.genfromtxt(self.filename,dtype='float',delimiter=data_delim,skip_header=data_line-1)

    def __loadVSM(self):
         """DataFile.__loadVSM(filename)
         
            Loads Data from a VSM file
            """
         self.__parse_plain_data()
 
    def __loadBigBlue(self,header_line,data_line):
        """DataFile.__loadBigBlue(filename,header_line,data_line)

        Lets you load the data from the files generated by Big Blue. Should work for any flat file 
        with a standard header file and comma separated data.
        
        header_line/data_line=line number of header/start of data
        
        TODO:    Get the metadata from the header
        """
        self.__parse_plain_data(header_line,data_line, data_delim=',', header_delim=',')

#   PUBLIC METHODS

    def load(self,filename=None,fileType="TDI",*args):
        """DataFile.load(filename,type,*args)
        
            Loads data from file filename using routines dependent on the fileType parameter
            fileType is one on TDI,VSM,BigBlue,csv Default is TDI.
            
            Example: To load Big Blue file
            
                d.load(file,"BigBlue",8,10)
            
            Where "BigBlue" is filetype and 8/10 are the line numbers of the headers/start of data respectively
            
            TODO: Implement a filename extension check to more intelligently guess the datafile type
            """
            
        if filename is None:
            filename=self.__file_dialog('r')
        else:
            self.filename = filename;
        
        if fileType=="TDI":
            self.__parse_data()
        elif fileType=="VSM":
            self.__loadVSM()
        elif fileType=="BigBlue":
            self.__loadBigBlue(args[0], args[1])
        elif fileType=="csv":
            self.__parse_plain_data(args[0], args[1], args[2], args[3])
        elif fileType=="NewXRD":
            from .Util import read_XRD_File
            d=read_XRD_File(filename)
            self.column_headers=d.column_headers
            self.data=d.data
            self.metadata=d.metadata
            self.typehint=d.typehint
            
        return self
        
    def save(self, filename=None):
        """DataFile.save(filename)
        
                Saves a string representation of the current DataFile object into the file 'filename' """
        if filename is None:
            filename=self.filename
        if filename is None: # now go and ask for one
            self.__file_dialog('w')
        f=open(filename, 'w')
        f.write(repr(self))
        f.close()
        self.filename=filename
        return self
        
      

    def metadata_value(self, text):
        """Wrapper of DataFile.meta for compatibility"""
        return self.meta(text)

    def data(self):
        return self.data

    def metadata(self):
        return self.metadata
        
    def typehint(self):
        return self.typehint

    def column_headers(self):
        return self.column_headers

    def find_col(self, col):
        if isinstance(col, int): #col is an int so pass on
            if col<0 or col>=len(self.column_headers):
                raise IndexError('Attempting to index a non-existant column')
            pass
        elif isinstance(col, str): # Ok we have a string
            if col in self.column_headers: # and it is an exact string match
                col=self.column_headers.index(col)
            else: # ok we'll try for a regular expression
                test=re.compile(col)
                possible=filter(test.search, self.column_headers)
                if len(possible)==0:
                    raise KeyError('Unable to find any possible column matches')
                col=self.column_headers.index(possible[0])
        elif isinstance(col, slice):
            indices=col.indices(numpy.shape(self.data)[1])
            col=range(*indices)
            col=self.find_col(col)
        elif isinstance(col, list):
            col=map(self.find_col, col)
        else:
            raise TypeError('Column index must be an integer or string')
        return col

    def column(self, col):
        """Extracts a column of data by index or name"""
        if isinstance(col, slice): # convert a slice into a list and then continue
            indices=col.indices(numpy.shape(self.data)[1])
            col=range(*indices)
        if isinstance(col, list):
            d=self.column(col[0])
            d=numpy.reshape(d, (len(d), 1))
            for x in range(1, len(col)):
                t=self.column(col[x])
                t=numpy.reshape(t, (len(t), 1))
                d=numpy.append(d,t , 1)
            return d
        else:
            return self.data[:, self.find_col(col)]
        
    def meta(self, ky):
        """Returns some metadata"""
        if isinstance(ky, str): #Ok we go at it with a string
            if ky in self.metadata:
                return self.metadata[ky]
            else:
                test=re.compile(ky)
                possible=filter(test.search, self.metadata)
                if len(possible)==0:
                    raise KeyError("No metadata with keyname: "+ky)
                elif len(possible)==1:
                    return self.metadata[possible[0]]
                else:
                    d=dict()
                    for p in possible:
                        d[p]=self.metadata[p]
                    return d
        else:
            raise TypeError("Only string are supported as search keys currently")
            # Should implement using a list of strings as well
    
    def dir(self, pattern=None):
        """ Return a list of keys in the metadata, filtering wiht a regular expression if necessary
                
                DataFile.dir(pattern) - pattern is a regular expression or None to list all keys"""
        if pattern==None:
            return self.metadata.keys()
        else:
            test=re.compile(pattern)
            possible=filter(test.search, self.metadata.keys())
            return possible
        
    def search(self, *args):
        """Searches in the numerica data part of the file for lines that match and returns  the corresponding rows

        Find row(s) that match the specified value in column:
        
        search(Column,value,columns=[list])
        
        Find rows that where the column is >= lower_limit and < upper_limit:
        
        search(Column,function ,columns=[list])
        
        Find rows where the function evaluates to true. Function should take two parameters x (float) and y(numpy array of floats).
        e.g. AnalysisFile.search('x',lambda x,y: x<10 and y[0]==2, ['y1','y2'])
        """
        
        if len(args)==2:
            col=args[0]
            targets=[]
            val=args[1]
        elif len(args)==3:
            col=args[0]
            if not isinstance(args[2],list):
                c=[args[2]]
            else:
                c=args[2]
            targets=map(self.find_col, c)
            val=args[1]        
        if len(targets)==0:
            targets=range(self.data.shape[1])
        d=numpy.transpose(numpy.atleast_2d(self.column(col)))
        d=numpy.append(d, self.data[:, targets], 1)
        if callable(val):
            rows=numpy.nonzero([val(x[0], x[1:]) for x in d])[0]
        elif isinstance(val, float):
            rows=numpy.nonzero([x[0]==val for x in d])[0]
        return self.data[rows][:, targets]
        
    def unique(self, col, return_index=False, return_inverse=False):
        """Return the unique values from the specified column - pass through for numpy.unique"""
        return numpy.unique(self.column(col), return_index, return_inverse)
    
    def del_rows(self, col, val=None):
        """Searchs in the numerica data for the lines that match and deletes the corresponding rows
        del_rows(Column, value)
        del_rows(Column,function) """
        if is_instance(col, slice) and val is None:
            indices=col.indices(len(self))
            col-=range(*indices)
        if isinstance(col, list) and val is None:
            col.sort(reverse=True)
            for c in col:
                self.del_rows(c)
        elif is_instance(col,  int) and val is None:
            self.data=numpy.delete(self.data, col, 0)
        else:
            col=self.find_col(col)
            d=self.column(col)
            if callable(val):
                rows=numpy.nonzero([val(x[col], x) for x in self])[0]
            elif isinstance(val, float):
                rows=numpy.nonzero([x==val for x in d])[0]
            self.data=numpy.delete(self.data, rows, 0)
        return self
    
    def add_column(self,column_data,column_header=None, index=None, func_args=None, replace=False):
        """Appends a column of data or inserts a column to a datafile"""
        if index is None:
            index=len(self.column_headers)
            replace=False
            if column_header is None:
                column_header="Col"+str(index)
        else:
            index=self.find_col(index)
            if column_header is None:
                column_header=self.column_headers[index]
        if not replace:
            self.column_headers.insert(index, column_header)
        else:
            self.column_headers[index]=column_header
        
        # The following 2 lines make the array we are adding a
        # [1, x] array, i.e. a column by first making it 2d and
        # then transposing it.
        if isinstance(column_data, numpy.ndarray):
            numpy_data=numpy.atleast_2d(column_data)
        elif callable(column_data):
            if isinstance(func_args, dict):
                new_data=[column_data(x, **func_args) for x in self]
            else:
                new_data=[column_data(x) for x in self]
            new_data=numpy.array(new_data)
            numpy_data=numpy.atleast_2d(new_data)
        else:
            return NotImplemented
        if replace:
            self.data[:, index]=numpy_data[0, :]
        else:
            self.data=numpy.insert(self.data,index, numpy_data,1)
        return self
            
    def del_column(self, col):
        c=self.find_col(col)
        self.data=numpy.delete(self.data, c, 1)
        if isinstance (c, list):
            c.sort(reverse=True)
        else:
            c=[c]
        for col in c:
            del self.column_headers[col]
        return self
    
    def rows(self):
        """Generator method that will iterate over rows of data"""
        (r, c)=numpy.shape(self.data)
        for row in range(r):
            yield self.data[row]
           
    def columns(self):
        """Generator method that will iterate over columns of data"""
        (r, c)=numpy.shape(self.data)
        for col in range(c):
            yield self.data[col]

    def sort(self, order):
        """Sorts the data by column name. Sorts in place and returns a copy of the sorted data object for chaining methods
        @param order Either a scalar integer or string or a list of integer or strings that represent the sort order
        @return A copy of the sorted object
        """
        if isinstance(order, list) or isinstance(order, tuple):
            order=[self.column_headers[self.find_col(x)] for x in order]
        else:
            order=[self.column_headers[self.find_col(order)]]
        d=numpy.sort(self.records, order=order)
        print d
        self.data=d.view(dtype='f8').reshape(len(self), len(self.column_headers))
        return self
    
    
    def csvArray(self,dump_location=defaultDumpLocation):
        spamWriter = csv.writer(open(dump_location, 'wb'), delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        i=0
        spamWriter.writerow(self.column_headers)
        while i< self.data.shape[0]:
            spamWriter.writerow(self.data[i,:])
            i+=1
            
    def edit(self):
        """Produce an editor window with a grid"""
        app = wx.PySimpleApp()
        frame = MyForm(self).Show()
        app.MainLoop()

