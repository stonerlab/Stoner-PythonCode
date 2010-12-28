#----------------------------------------------------------------------------- 
#   $Id: __init__.py,v 1.3 2010/12/28 22:31:25 cvs Exp $
#   AUTHOR:     MATTHEW NEWMAN, CHRIS ALLEN, GAVIN BURNELL
#   DATE:       24/11/2010
#-----------------------------------------------------------------------------
#
# Imports
# If Imort is just used in a sub class of DataFile, consider importing in __init__. See PlotFile for example. GB 23/12/2010
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
    
    TODO: Add X-Ray file reading utility
    
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
        if isinstance(name, int):
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
        
    def save(self, filename):
        """DataFile.save(filename)
        
                Saves a string representation of the current DataFile object into the file 'filename' """
        f=open(filename, 'w')
        f.write(repr(self))
        f.close()
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
        elif isinstance(col, list):
            col=map(self.find_col, col)
        else:
            raise TypeError('Column index must be an integer or string')
        return col

    def column(self, col):
        """Extracts a column of data by index or name"""
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
        
    def del_rows(self, col, val):
        """Searchs in the numerica data for the lines that match and deletes the corresponding rows
        del_rows(Column, value)
        del_rows(Column,function) """
        d=self.column(col)
        if callable(val):
            rows=numpy.nonzero([val(x) for x in d])[0]
        elif isinstance(val, float):
            rows=numpy.nonzero([x==val for x in d])[0]
        self.data=numpy.delete(self.data, rows, 0)
        return self
    
    def add_column(self,column_data,column_header='', index=None):
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
            new_data=map(column_data, self)
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
class PlotFile(DataFile):
    """Extends DataFile with plotting functions"""
    def __init__(self, *args, **kargs): #Do the import of pylab here to speed module load
        global pylab
        import pylab
        super(PlotFile, self).__init__(*args, **kargs)
    def plot_xy(self,column_x, column_y,title='',save_filename='',show_plot=True):
        """plot_xy(x column, y column/s, title,save filename, show plot=True)
        
                Makes and X-Y plot of the specified data."""
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        x=self.column(column_x)
        y=self.column(column_y)
        if show_plot == True:
            pylab.ion()
        pylab.plot(x,y)
        pylab.draw()
        pylab.xlabel(str(self.column_headers[column_x]))
        if isinstance(column_y, list):
            ylabel=column_y
            ylabel[0]=self.column_headers[column_y[0]]
            ylabel=reduce(lambda x, y: x+","+self.column_headers[y],  ylabel)
        else:
            ylabel=self.column_headers[column_y]
        pylab.ylabel(str(ylabel))
        if title=='':
            title=self.filename
        pylab.title(title)
        pylab.grid(True)
        if save_filename != '':
            pylab.savefig(str(save_filename))
     
 
class AnalyseFile(DataFile):
    """Extends DataFile with numpy passthrough functions"""

#Private Helper Functions
    def __SG_calc_coeff(self, num_points, pol_degree=1, diff_order=0):
        
        """ calculates filter coefficients for symmetric savitzky-golay filter.
            see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf
        
            num_points   means that 2*num_points+1 values contribute to the
                     smoother.
        
            pol_degree   is degree of fitting polynomial
        
            diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first 
                                                 derivative of function.
                     and so on ...
        
        """
    
        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....
    
        x = numpy.arange(-num_points, num_points+1, dtype=int)
        monom = lambda x, deg : math.pow(x, deg)
    
        A = numpy.zeros((2*num_points+1, pol_degree+1), float)
        for i in range(2*num_points+1):
            for j in range(pol_degree+1):
                A[i,j] = monom(x[i], j)
            
        # calculate diff_order-th row of inv(A^T A)
        ATA = numpy.dot(A.transpose(), A)
        rhs = numpy.zeros((pol_degree+1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = numpy.linalg.solve(ATA, rhs)
    
        # calculate filter-coefficients
        coeff = numpy.dot(A, wvec)
    
        return coeff

    def __SG_smooth(self, signal, coeff):
        
        """ applies coefficients calculated by calc_coeff()
            to signal """
        
        N = numpy.size(coeff-1)/2
        res = numpy.convolve(signal, coeff)
        return res[N:-N]
        
    def __threshold(self, threshold, data, rising=True, falling=False):
        """ Internal function that implements the threshold method - also used in peak-finder"""
        current=data
        previous=numpy.roll(current, 1)
        index=numpy.arange(len(current))        
        sdat=numpy.column_stack((index, current, previous))
        if rising==True and falling==False:
            expr=lambda x:(x[1]>=threshold) & (x[2]<threshold)
        elif rising==True and falling==True:
            expr=lambda x:((x[1]>=threshold) & (x[2]<threshold)) | ((x[1]<=threshold) & (x[2]>threshold))
        elif rising==False and falling==True:
            expr=lambda x:(x[1]<=threshold) & (x[2]>threshold)
        else:
            expr=lambda x:False
        return filter(lambda x:x>0,  map(lambda x:x[0]-1+(x[1]-threshold)/(x[1]-x[2]), filter(expr, sdat)))
        
    
    def polyfit(self,column_x,column_y,polynomial_order, bounds=lambda x, y:True):
        """ Pass through to numpy.polyfit
        
                AnalysisFile.polyfit(xx_column,y_column,polynomial_order,bounds function)
                
                x_column and y_column can be integers or strings that match the column headings
                bounds function should be a python function that takes a single paramter that represents an x value
                and returns true if the datapoint is to be retained and false if it isn't."""
        working=self.search(column_x, bounds, column_y)
        return numpy.polyfit(working[0],working[1],polynomial_order)
        
    def curve_fit(self, func,  xcol, ycol, p0=None, sigma=None, bounds=lambda x, y: True ):
        """General curve fitting function passed through from numpy
        
                AnalysisFile.Curve_fit(fitting function, x-column,y_column, initial parameters=None, weighting=None, bounds function)
                
                The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
                The x-column and y-column can be either strings to be matched against column headings or integers. 
                The initial parameter values and weightings default to None which corresponds to all parameters starting 
                at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the 
                point is to be used and false if not.
        """
        from scipy.optimize import curve_fit
        working=self.search(xcol, bounds, [xcol, ycol])
        popt, pcov=curve_fit(func,  working[:, 0], working[:, 1], p0, sigma)
        return popt, pcov
        
    def max(self, column):
        """FInd maximum value and index in a column of data
                
                AnalysisFile.max(column)
                """
        col=self.find_col(column)
        return self.data[:, col].max(), self.data[:, col].argmax()
        
    def min(self, column):
        """FInd minimum value and index in a column of data
                
                AnalysisFile.min(column)
                """
        col=self.find_col(column)
        return self.data[:, col].min(), self.data[:, col].argmin()
    
    def apply(self, func, col, insert=False):
        """Applies the given function to each row in the data set and adds to the data set
        
            AnalysisFile.apply(func,column,insert=False)"""
        col=self.find_col(col)
        nc=numpy.array([func(row) for row in self.rows()])
        if insert==True:
            self=self.add_column(nc, func.__name__, col)
        else:
            self.data[:, col]=nc
        return self

    def SG_Filter(self, col, points, poly=1, order=0):
        """ Implements Savitsky-Golay filtering of data for smoothing and differentiating data
        
        SG_Filter(column,points, polynomial order, order of differentuation)
        or
        SG_Filter((x-col,y,col),points,polynomial order, order of differentuation)"""
        p=points
        if isinstance(col, tuple):
            x=self.column(col[0])
            x=numpy.append(numpy.array([x[0]]*p), x)
            x=numpy.append(x, numpy.array([x[-1]]*p))
            y=elf.column(col[1])
            y=anumpy.append(numpy.array([y[0]]*p), y)
            y=anumpy.append(y, numpy.array([y[-1]]*p))
            dx=self.__SG_smooth(x, self.__SG_calc_coeff(points, poly, order))
            dy=self.__SG_smooth(y, self.__SG_calc_coeff(points, poly, order))
            r=dy/dx
            return r[p:-p]
        else:
            d=self.column(col)
            d=numpy.append(numpy.array([d[0]]*p),d)
            d=numpy.append(d, numpy.array([d[-1]]*p))
            r=self.__SG_smooth(d, self.__SG_calc_coeff(points, poly, order))
            return r[p:-p]
    def threshold(self, col, threshold, rising=True, falling=False):
        """AnalysisFile.threshold(column, threshold, rising=True,falling=False)
        
        Finds partial indices where the data in column passes the threshold, rising or falling"""
        current=self.column(col)
        return self.__threshold(threshold, current, rising=rising, falling=falling)
        
    def interpolate(self, newX,kind='linear' ):
        from scipy.interpolate import interp1d
        l=numpy.shape(self.data)[0]
        index=numpy.arange(l)
        inter=interp1d(index, self.data, kind, 0)
        return inter(newX)
    
    def peaks(self, ycol, width, significance , xcol=None, peaks=True, troughs=False, poly=2):
        """AnalysisFile.peaks(ycol,width,signficance, xcol=None.peaks=True, troughs=False)
        
        Locates peaks and/or troughs in a column of data by using SG-differentiation.
        
        ycol is the column name or index of the data in which to search for peaks
        width is the expected minium halalf-width of a peak in terms of the number of data points. 
                This is used in the differnetiation code to find local maxima. Bigger equals less sensitive
                to experimental noise, smaller means better eable to see sharp peaks
            sensitivity is used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensistive, smaller means more likely to detect noise.
            xcol name or index of data column that p[rovides the x-coordinate
            peaks,troughs select whether to measure peaks.troughs in data"""
        from scipy.interpolate import interp1d
        d1=self.SG_Filter(ycol, width, poly, 1)
        i=numpy.arange(len(d1))
        d2=interp1d(i, self.SG_Filter(ycol, width, poly, 2))
        if xcol==None:
            xcol=i
        else:
            xcol=self.column(xcol)
        index=interp1d(i, xcol)
        z=self.__threshold(0, d1, rising=troughs, falling=peaks)
        return index(filter(lambda x: numpy.abs(d2(x))>significance, z))       
        
