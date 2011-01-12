#############################################
#
# Core object of the Stoner Package
#
# $Id: Core.py,v 1.2 2011/01/12 22:56:33 cvs Exp $
#
# $Log: Core.py,v $
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
    """Stoner.DataFile represents a standard Stonerlab data file as an object
    
    Provides methods to read, and manipulate data
    
    Matt Newman, Chris Allen, Gavin Burnell
    
    
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
        
        3 forms are recognised: DataFile('filename',<optional filetype>,<args>), DataFile(array), DataFile(dictionary), DataFile(array,dictionary), DataFile(DataFile)
        """
        self.data = numpy.array([])
        self.metadata = dict()
        self.typehint = dict()
        self.filename = ''
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

    def __getitem__(self, name): # called for DataFile[x] returns row x if x is integer, or metadata[x] if x is string
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
        else:
            raise TypeError("Key must be either numeric of string")

    def __setitem__(self, name, value): # writing the metadata means doing something sensible with the type hints
        if isinstance(value,bool):
            self.typehint[name]="Boolean"
        elif isinstance(value, int):
            self.typehint[name]="I32"
        elif isinstance(value, float):
            self.typehint[name]="Double Float"
        else:
            self.typehint[name]="String"
        self.metadata[name]=value
        
    def __add__(self, other): #Overload the + operator to add data file rows
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
        
    def __and__(self, other): #Overload the & operator to add datafile columns
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
        
    def __repr__(self): # What happens when you print a DataFile object. Also used to save the data
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

    def load(self,filename,fileType="TDI",*args):
        """DataFile.load(filename,type,*args)
        
            Loads data from file filename using routines dependent on the fileType parameter
            fileType is one on TDI,VSM,BigBlue,csv Default is TDI.
            
            Example: To load Big Blue file
            
                d.load(file,"BigBlue",8,10)
            
            Where "BigBlue" is filetype and 8/10 are the line numbers of the headers/start of data respectively
            
            TODO: Implement a filename extension check to more intelligently guess the datafile type
            """
            
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
            targets=map(self.find_col, args[2])
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
    
    def add_column(self,column_data,column_header='', index=None, func_args=None):
        """Appends a column of data or inserts a column to a datafile"""
        if index is None:
                index=len(self.column_headers)
        else:
            index=self.find_col(index)
        self.column_headers.insert(index, column_header)
        
        # The following 2 lines make the array we are adding a
        # [1, x] array, i.e. a column by first making it 2d and
        # then transposing it.
        if isinstance(column_data, numpy.ndarray):
            column_data=numpy.atleast_2d(column_data)
            self.data=numpy.insert(self.data,index, column_data,1)
        elif callable(column_data):
            if is_instance(func_args, dict):
                new_data=[new_data(x, **func_args) for x in self]
            else:
                new_data=[new_data(x) for x in self]
            new_data=numpy.array(new_data)
            numpy_data=numpy.atleast_2d(new_data)
            self.data=numpy.insert(self.data,index, numpy_data,1)
        else:
            return NotImplemented
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


    def csvArray(self,dump_location=defaultDumpLocation):
        spamWriter = csv.writer(open(dump_location, 'wb'), delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        i=0
        spamWriter.writerow(self.column_headers)
        while i< self.data.shape[0]:
            spamWriter.writerow(self.data[i,:])
            i+=1
