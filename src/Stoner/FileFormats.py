"""Stoner.FileFormats is a module within the Stoner package that provides extra classes
that can load data from various instruments into DataFile type objects.

Eacg class has a priority attribute that is used to determine the order in which
they are tried by DataFile and friends where trying to load data. High priority
is run last.

Eacg class should implement a load() method and optionally a save() method.
"""
import linecache
import re
import numpy
import fileinput
import csv
import string
import struct
from re import split
from datetime import datetime
import numpy.ma as ma

from .Core import DataFile
from .pyTDMS import read as tdms_read



class CSVFile(DataFile):
    """A subclass of DataFiule for loading generic deliminated text fiules without metadata."""

    priority=128 # Rather generic file format so make it a low priority

    def load(self,filename=None,header_line=0, data_line=1, data_delim=',', header_delim=',', **kargs):
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
            header_string=linecache.getline(self.filename, header_line)
            header_string=re.sub(r'["\n]', '', header_string)
            try:
                print header_string.index(header_delim)
            except ValueError:
                raise RuntimeError("No Delimiters in header line")
            self.column_headers=[x.strip() for x in header_string.split(header_delim)]
        else:
            self.column_headers=["Column"+str(x) for x in range(numpy.shape(self.data)[1])]
            data_line=linecache.getline(self.filename,data_line)
            try:
                data_line.index(data_delim)
            except ValueError:
                raise RuntimeError("No delimiters in data lines")
            
        self.data=numpy.genfromtxt(self.filename,dtype='float',delimiter=data_delim,skip_header=data_line-1)
        return self

    def save(self,filename, deliminator=','):
        """Overrides the save method to allow CSVFiles to be written out to disc (as a mininmalist output)
        
        Args:
            filename (string): Fielname to save as (using the same rules as for the load routines)
            
        Keyword Arguments:            
            deliminator (string): Record deliniminator (defaults to a comma)
            
        Returns:
            A copy of itself."""
        if filename is None:
            filename=self.filename
        if filename is None or (isinstance(filename, bool) and not filename): # now go and ask for one
            filename=self.__file_dialog('w')
        spamWriter = csv.writer(open(filename, 'wb'), delimiter=deliminator,quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i=0
        spamWriter.writerow(self.column_headers)
        while i< self.data.shape[0]:
            spamWriter.writerow(self.data[i,:])
            i+=1
        return self


class VSMFile(DataFile):
    """Extends DataFile to open VSM Files"""
    
    priority=16 # Now makes a positive ID of its contents

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
        f=fileinput.FileInput(self.filename) # Read filename linewise
        self['Timestamp']=f.next().strip()
        try:
            check=datetime.strptime(self["Timestamp"], "%a %b %d %H:%M:%S %Y")
            assert check is not None            
            assert f.next().strip()==""
        except (ValueError, AssertionError):
            raise RuntimeError('Not a VSM File')
        header_string=f.next()
        header_string=re.sub(r'["\n]', '', header_string)
        unit_string=f.next()
        unit_string=re.sub(r'["\n]', '', unit_string)
        column_headers=zip(header_string.split(header_delim), unit_string.split(header_delim))
        self.column_headers=map(lambda x: x[0].strip()+" ("+x[1].strip()+")",  column_headers)
        f.next()
        f.next()

        self.data=numpy.genfromtxt(f,dtype='float',usemask=True, delimiter=data_delim,
                                   skip_header=data_line-1, missing_values=['         6:0','         ---'],
                                invalid_raise=False)
        self.data=ma.mask_rows(self.data)
        cols=self.data.shape[1]
        self.data=numpy.reshape(self.data.compressed(),(-1,cols))
        f.close()


    def load(self,filename=None,*args, **kargs):
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

    priority=64 # Also rather generic file format so make a lower priority

    def load(self,filename=None,*args, **kargs):
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

        super(BigBlueFile,self).load(self, self.filename,  header_line=3, data_line=7, data_delim=' ', header_delim=',')
        return self

class QDSquidVSMFile(DataFile):
    """Extends DataFile to load files from The SQUID VSM"""

    priority=16 # Is able to make a positive ID of its file content, so get priority to check

    def load(self,filename=None,*args, **kargs):
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
        f=fileinput.FileInput(self.filename) # Read filename linewise
        while f.next().strip()!="[Header]":
            pass
        line=f.next().strip()
        line=f.next().strip()
        if "Quantum Design" not in line:
            raise RuntimeError("Not a Quantum Design File !")
        while line!="[Data]":
            if line[0]==";":
                line=f.next().strip()
                continue
            parts=line.split(',')
            if parts[0]=="INFO":
                key=parts[0]+parts[2]
                key=key.title()
                value=parts[1]
            elif parts[0] in ['BYAPP', 'FILEOPENTIME']:
                key=parts[0].title()
                value=' '.join(parts[1:])
            else:
                key=parts[0]+"."+parts[1]
                key=key.title()
                value=' '.join(parts[2:])
            self.metadata[key]=self.metadata.string_to_type(value)
            line=f.next().strip()
        self.column_headers=f.next().strip().split(',')
        self.data=numpy.genfromtxt(f,dtype='float',delimiter=',', invalid_raise=False)
        return self

class OpenGDAFile(DataFile):
    """Extends DataFile to load files from RASOR"""

    priority=16 # Makes a positive ID of it's file type so give priority

    def load(self,filename=None,*args, **kargs):
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
        f=fileinput.FileInput(self.filename) # Read filename linewise
        line=f.next().strip()
        if line!="&SRS":
            raise RuntimeError("Not a GDA File from Rasor ?"+str(line))
        while f.next().strip()!="<MetaDataAtStart>":
            pass
        line=f.next().strip()
        while line!="</MetaDataAtStart>":
            parts=line.split('=')
            if len(parts)!=2:
                line=f.next().strip()
                continue
            key=parts[0]
            value=parts[1].strip()
            self.metadata[key]=self.metadata.string_to_type(value)
            line=f.next().strip()
        while f.next().strip()!="&END":
            pass
        self.column_headers=f.next().strip().split("\t")
        self.data=numpy.genfromtxt(f,dtype='float', invalid_raise=False)
        return self

class RasorFile(OpenGDAFile):
    """Just an alias for OpenGDAFile"""
    pass


class SPCFile(DataFile):
    """Extends DataFile to load SPC files from Raman"""
    
    priority=64 # Can't make a positive ID of itself
    
    def load(self,filename=None,*args, **kargs):
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
        f=open(filename, 'rb')
        spchdr=struct.unpack('BBBciddiBBBBi9s9sH8f30s130siiBBHf48sfifB187s', f.read(512))
        keys=("ftflgs","fversn","fexper","fexp","fnpts","ffirst","flast","fnsub","fxtype","fytype","fztype","fpost","fres","fsource","fpeakpt","fspare1","fspare2","fspare3","fspare4","fspare5","fspare6","fspare7","fspare8","fcm","nt","fcatx","flogoff","fmods","fprocs","flevel","fsampin","ffactor","fmethod","fzinc","fwplanes","fwinc","fwtype","fwtype","fresv")
        header=dict(zip(keys, spchdr))

        if header['ftflgs'] & 64: # This is the multiple XY curves in file flag.
            raise NotImplemented("Filetype not implemented yet !")
        else: # A single XY curve in the file.
            n=header['fnsub']
            pts=header['fnpts']
            if header['ftflgs'] & 128: # We need to read some X Data
                xvals=f.read(4*pts) # I think storing X vals directly implies that each one is 4 bytes....
                xdata=numpy.array(struct.unpack(str(pts)+"f", xvals))
            else: # Generate the X Data ourselves
                first=header['ffirst']
                last=header['flast']
                incr=(last-first)/(pts-1) # make sure we get first to last inclusive
                xdata=numpy.arange(first, last+(incr/2), incr)
            data=numpy.zeros((pts,  (n+1))) # initialise the data soace
            data[:, 0]=xdata # Put in the X-Data
            xvars=["Arbitrary","Wavenumber (cm-1)","Micrometers (um)","Nanometers (nm)","Seconds","Minutes","Hertz (Hz)","Kilohertz (KHz)","Megahertz (MHz)","Mass (M/z)","Parts per million (PPM)","Days","Years","Raman Shift (cm-1)","Raman Shift (cm-1)","eV","XYZ text labels in fcatxt (old 0x4D version only)","Diode Number","Channel","Degrees","Temperature (F)","Temperature (C)","Temperature (K)","Data Points","Milliseconds (mSec)","Microseconds (uSec)","Nanoseconds (nSec)","Gigahertz (GHz)","Centimeters (cm)","Meters (m)","Millimeters (mm)","Hours","Hours"]
            yvars=["Arbitrary Intensity","Interferogram","Absorbance","Kubelka-Monk","Counts","Volts","Degrees","Milliamps","Millimeters","Millivolts","Log(1/R)","Percent","Percent","Intensity","Relative Intensity","Energy","Decibel","Temperature (F)","Temperature (C)","Temperature (K)","Index of Refraction [N]","Extinction Coeff. [K]","Real","Imaginary","Complex","Complex","Transmission (ALL HIGHER MUST HAVE VALLEYS!)","Reflectance","Arbitrary or Single Beam with Valley Peaks","Emission","Emission"]
            column_headers=[xvars[header['fxtype']]] # And label the X column correctly

            #Now we're going to read the Y-data
            # Start by preping some vars for use

            subhdr_keys=("subflgs","subexp","subindx", "subtime", "subnext", "subnois", "subnpts", "subscan", "subwlevel", "subresv")
            if header['ftflgs'] &1:
                y_width=2
                y_fmt='h'
                divisor=2**16
            else:
                y_width=4
                y_fmt='i'
                divisor=2**32

            for j in range(n): # We have n sub-scans
                # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
                subhdr=struct.unpack('BBHfffIIf4s', f.read(32))
                subheader=dict(zip(["scan"+str(j)+":"+x for x in subhdr_keys], subhdr))

                # Now read the y-data
                exponent=subheader["scan"+str(j)+':subexp']
                if int(exponent) & -128: # Data is unscaled direct floats
                    ydata=numpy.array(struct.unpack(str(pts)+"f", f.read(pts*y_width)))
                else: # Data is scaled by exponent
                    yvals=struct.unpack(str(pts)+y_fmt, f.read(pts*y_width))
                    ydata=numpy.array(yvals, dtype='float64')*(2**exponent)/divisor

                # Pop the y-data into the array and merge the matadata in too.
                data[:, j+1]=ydata
                header=dict(header, **subheader)
                column_headers.append("Scan"+str(j)+":"+yvars[header['fytype']])

            # Now we're going to read any log information
            if header['flogoff']!=0: # Ok, we've got a log, so read the log header and merge into metadata
                logstc=struct.unpack('IIIII44s', f.read(64))
                logstc_keys=("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
                logheader=dict(zip(logstc_keys, logstc))
                header=dict(header, **logheader)

                # Can't handle either binary log information or ion disk log information (wtf is this anyway !)
                f.read(header['logbins']+header['logdsks'])

                # The renishaw seems to put a 16 character timestamp next - it's not in the spec but never mind that.
                header['Date-Time']=f.read(16)
                # Now read the rest of the file as log text
                logtext=f.read()
                # We expect things to be single lines terminated with a CR-LF of the format key=value
                for line in split("[\r\n]+", logtext):
                    if "=" in line:
                        parts= line.split('=')
                        key=parts[0]
                        value=parts[1]
                        header[key]=value
            # Ok now build the Stoner.DataFile instance to return
            self.data=data
            # The next bit generates the metadata. We don't just copy the metadata because we need to figure out the typehints first - hence the loop here to call DataFile.__setitem()
            for x in header:
                self[x]=header[x]
            self.column_headers=column_headers
            f.close() # tidy up and return
            return self


class TDMSFile(DataFile):
    """A first stab at writing a file that will import TDMS files"""

    Objects=dict()
    priority=16 # Makes a positive ID of its file contents

    def load(self, filename=None, *args, **kargs):
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
        f=open(self.filename) # Read filename linewise
        try:
            assert f.read(4) == "TDSm"
        except AssertionError:
            f.close()
            raise RuntimeError('Not a TDMS File')
        f.close()
        (metadata, data)=tdms_read(self.filename)
        for key in metadata:
            self.metadata[key]=metadata[key]
        self.column_headers = list()
        for column in data:
            nd=data[column]
            self.add_column(nd, column)
        return self

class RigakuFile(DataFile):
    """Loads a .ras file as produced by Rigaku X-ray diffractormeters"""
    
    priority=16 #Can make a positive id of file from first line
    
    def load(self, filename=None, *args, **kargs):
        """Reads an Rigaku ras file including handling the metadata nicely        

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.
                
        Returns:
            A copy of the itself after loading the data.
            """
        from ast import literal_eval
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        sh=re.compile(r'^\*([^\s]+)\s+(.*)$') # Regexp to grab the keys
        ka=re.compile(r'(.*)\-(\d+)$')
        f=fileinput.FileInput(self.filename) # Read filename linewise
        if f.next().strip()!="*RAS_DATA_START": # Check we have the corrrect fileformat
                raise RuntimeError("File Format Not Recognized !")
        line=f.next().strip()
        while line!="*RAS_HEADER_START":
            line=f.next().strip()
        header=dict()
        while line!="*RAS_HEADER_END":
            line=f.next().strip()
            m=sh.match(line)
            if m:
                key=m.groups()[0].lower().replace('_','.')
                value=m.groups()[1]
                header[key]=value
        keys=header.keys()
        keys.sort()
        for key in keys:
            m=ka.match(key)
            value=header[key].strip()
            try:
                newvalue=literal_eval(value.strip('"'))
            except Exception:
                newvalue=literal_eval(value)
            if m:
                key=m.groups()[0]
                if key in self.metadata and not (isinstance(self[key], numpy.ndarray) or isinstance(self[key], list)):
                    if isinstance(self[key], str):
                        self[key]=list([self[key]])
                    else:
                        self[key]=numpy.array(self[key])
                if key not in self.metadata:
                    if isinstance(newvalue, str):
                        self[key]=list([newvalue])
                    else:
                        self[key]=numpy.array([newvalue])
                else:
                    if isinstance(self[key][0], str):
                        self[key].append(newvalue)
                    else:
                        self[key]=numpy.append(self[key], newvalue)
            else:
                self.metadata[key]=newvalue
        while(line!="*RAS_INT_START"):
             line=f.next().strip()
        self.data=numpy.genfromtxt(f, dtype='float', delimiter=' ', invalid_raise=False)
        self.column_headers=['Column'+str(i) for i in range(self.data.shape[1])]
        self.column_headers[0:2]=[self.metadata['meas.scan.unit.x'], self.metadata['meas.scan.unit.y']]
        for key in self.metadata:
            if isinstance(self[key], list):
                self[key]=numpy.array(self[key])
        return self

class XRDFile(DataFile):
    """Loads Files from a Brucker D8 Discovery X-Ray Diffractometer"""

    priority=16 # Makes a positive id of its file contents

    def load(self,filename=None,*args, **kargs):
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
        sh=re.compile(r'\[(.+)\]') # Regexp to grab section name
        f=fileinput.FileInput(self.filename) # Read filename linewise
        if f.next().strip()!=";RAW4.00": # Check we have the corrrect fileformat
                raise RuntimeError("File Format Not Recognized !")
        drive=0
        for line in f: #for each line
            m=sh.search(line)
            if m: # This is a new section
                section=m.group(1)
                if section=="Drive": #If this is a Drive section we need to know which Drive Section it is
                    section=section+str(drive)
                    drive=drive+1
                elif section=="Data": # Data section contains the business but has a redundant first line
                    f.next()
                for line in f: #Now start reading lines in this section...
                    if line.strip()=="": # A blank line marks the end of the section, so go back to the outer loop which will handle a new section
                        break
                    elif section=="Data": # In the Data section read lines of data value,vale
                        parts=line.split(',')
                        angle=parts[0].strip()
                        counts=parts[1].strip()
                        dataline=numpy.array([float(angle), float(counts)])
                        self.data=numpy.append(self.data, dataline)
                    else: # Other sections contain metadata
                        parts=line.split('=')
                        key=parts[0].strip()
                        data=parts[1].strip()
                        self[section+":"+key]=data # Keynames in main metadata are section:key - use theDataFile magic to do type determination
        self.column_headers=['Angle', 'Counts'] # Assume the columns were Angles and Counts

        f.close()# Cleanup
        self.data=numpy.reshape(self.data, (-1, 2))
        return self


class BNLFile(DataFile):
    """
    Creates BNLFile a subclass of DataFile that caters for files in the SPEC format given
    by BNL (specifically u4b beamline but hopefully generalisable).

    Author Rowan 12/2011

    The file from BNL must be split into seperate scan files before Stoner can use
    them, a separate python script has been written for this and should be found
    in data/Python/PythonCode/scripts.
    """
    def __init__(self, *params):
        """Constructor modification
        BNLFile('filename')
        Do a normal initiation using the parent class 'self' followed by adding an extra attribute line_numbers,
        line_numbers is a list of important line numbers in the file.
        I've left it open for someone to add options for more args if they wish."""
        super(BNLFile,self).__init__(*params)
        self.line_numbers=[]

    def __find_lines(self):
        """returns an array of ints [header_line,data_line,scan_line,date_line,motor_line]"""
        fp=open(self.filename,'r')
        self.line_numbers=[0,0,0,0,0]
        counter=0
        for line in fp:
            counter+=1
            if counter==1 and line[0]!='#':
                raise RuntimeError("Not a BNL File ?")
            if len(line)<2:continue  #if there's nothing written on the line go to the next
            elif line[0:2]=='#L':self.line_numbers[0]=counter
            elif line[0:2]=='#S':self.line_numbers[2]=counter
            elif line[0:2]=='#D':self.line_numbers[3]=counter
            elif line[0:2]=='#P':self.line_numbers[4]=counter
            elif line[0] in ['0','1','2','3','4','5','6','7','8','9']:
                self.line_numbers[1]=counter
                break

    def __get_metadata(self):
        """Metadata found is scan number 'Snumber', scan type and parameters 'Stype',
        scan date/time 'Sdatetime' and z motor position 'Smotor'."""
        scanLine=linecache.getline(self.filename,self.line_numbers[2])
        dateLine=linecache.getline(self.filename,self.line_numbers[3])
        motorLine=linecache.getline(self.filename,self.line_numbers[4])
        self.__setitem__('Snumber',scanLine.split()[1])
        tmp=string.join(scanLine.split()[2:])
        self.__setitem__('Stype',string.join(tmp.split(','))) #get rid of commas
        self.__setitem__('Sdatetime',dateLine[3:-1])  #don't want \n at end of line so use -1
        self.__setitem__('Smotor',motorLine.split()[3])


    def __parse_BNL_data(self):
        """
        Internal function for parsing BNL data. The meta data is labelled by #L type tags
        so easy to find but #L must be excluded from the result.
        """
        self.__find_lines()
        """creates a list, line_numbers, formatted [header_line,data_line,scan_line,date_line,motor_line]"""
        header_string=linecache.getline(self.filename, self.line_numbers[0])
        header_string=re.sub(r'["\n]', '', header_string) #get rid of new line character
        header_string=re.sub(r'#L', '', header_string) #get rid of line indicator character
        self.column_headers=map(lambda x: x.strip(),  header_string.split())
        self.__get_metadata()
        try: self.data=numpy.genfromtxt(self.filename,skip_header=self.line_numbers[1]-1)
        except IOError:
            self.data=numpy.array([0])
            print 'Did not import any data for %s'% self.filename


    def load(self,filename, *args, **kargs):        #fileType omitted, implicit in class call
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
        self.filename=filename
        self.__parse_BNL_data() #call an internal function rather than put it in load function
        return self


class FmokeFile(DataFile):
    """Extends DataFile to open Fmoke Files"""

    priority=16 # Makes a positive ID check of its contents so give it priority in autoloading

    def load(self,filename=None,*args, **kargs):
        """Sheffield Fovussed MOKE file loader routine.
        
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
        f=fileinput.FileInput(self.filename) # Read filename linewise
        value=[float(x.strip()) for x in f.next().split('\t')]
        label=[ x.strip() for x in f.next().split('\t')]
        if label[0]!="Header:":
            raise RuntimeError("Not a Focussed MOKE file !")
        del(label[0])
        for k,v in zip(label, value):
               self.metadata[k]=v # Create metatdata from first 2 lines
        self.column_headers=[ x.strip() for x in f.next().split('\t')]
        self.data=numpy.genfromtxt(f, dtype='float', delimiter='\t', invalid_raise=False)
        return self


