####################################################
## FileFormats - sub classes of DataFile for different machines
# $Id: FileFormats.py,v 1.30 2013/03/05 16:22:54 cvs Exp $
# $Log: FileFormats.py,v $
# Revision 1.30  2013/03/05 16:22:54  cvs
# Fix to del_rows in Core, mask should not be indexed here
#
# Revision 1.29  2012/12/11 16:08:59  cvs
# Add a Rigaku file reader
#
# Revision 1.28  2012/05/04 16:47:25  cvs
# Fixed a string representation problem in __repr__. Minor changes to BNLFile format.
#
# Revision 1.27  2012/05/02 21:30:25  cvs
# Add more checks to help autofile laoding and priorities on it.
#
# Revision 1.26  2012/05/01 16:11:03  cvs
# Workinf FmokeFile format
#
# Revision 1.23  2012/04/28 20:05:14  cvs
# Switch RasorFile to OpenGDAFile and make it handle blank lines in metadata
#
# Revision 1.22  2012/04/06 19:36:08  cvs
# Update DataFolder to support regexps in pattern and filter. When used as a pattern named capturing groups can be used to feed metadata. Minor improvements in Core and fix to RasorFile
#
# Revision 1.21  2012/04/05 11:32:38  cvs
# Just modified some comments in BNLdata
#
# Revision 1.20  2012/04/03 15:13:44  cvs
# Not sure what was done here !
#
# Revision 1.19  2012/04/02 11:58:07  cvs
# Minor bug fixes and corrections
#
# Revision 1.18  2012/03/26 21:57:55  cvs
# Some improvements to auto-file detection
#
# Revision 1.17  2012/03/25 20:35:06  cvs
# More work to stop load recursiing badly
#
# Revision 1.16  2012/03/25 19:41:31  cvs
# Teach DataFile.load() to try every possible subclass if at first it doesn't suceed.
#
# Revision 1.15  2012/03/11 23:12:33  cvs
# string_to_type function to do a better job of working out python type from string representation when no type hint give.
#
# Revision 1.14  2012/03/11 01:41:56  cvs
# Recompile API help
#
# Revision 1.13  2012/03/10 20:12:58  cvs
# Add new formats for Diamond I10 files
#
# Revision 1.12  2012/01/04 23:07:54  cvs
# Make BigBlueFile really a subclass of CSVFile
#
# Revision 1.11  2012/01/04 22:35:32  cvs
# Give CSVFIle options to skip headers
# Make PlotFile.plot_xy errornar friendly
#
# Revision 1.10  2012/01/03 21:51:04  cvs
# Fix a bug with add_column
# Upload new TDMS data
#
# Revision 1.9  2012/01/03 12:41:50  cvs
# Made Core pep8 compliant
# Added TDMS code and TDMSFile
#
# Revision 1.8  2011/12/12 10:26:32  cvs
# Minor changes in BNLFile class to fix a metadata bug. Rowan
#
# Revision 1.7  2011/12/09 12:10:41  cvs
# Remove cvs writing code from DataFile (use CSVFile.save()). Fixed BNLFile to always call the DataFile constructor
#
# Revision 1.6  2011/12/06 16:35:37  cvs
# added import string line to sort a bug in BNLFile class. Rowan
#
# Revision 1.5  2011/12/06 09:48:49  cvs
# Add BNLFile for Brookhaven Data (Rowan)
#
# Revision 1.4  2011/12/05 22:58:11  cvs
# Make CSVFile able to save as a CSV file and remove csvdump from Core. Update docs
#
# Revision 1.3  2011/12/05 21:56:26  cvs
# Add in DataFile methods swap_column and reorder_columns and update API documentation. Fix some Doxygen problems.
#

import linecache
import re
import numpy
import fileinput
import csv
import string
import struct
from re import split
import codecs
from datetime import datetime

from .Core import DataFile
from .pyTDMS import read as tdms_read



class CSVFile(DataFile):
    """A subclass of DataFiule for loading generic deliminated text fiules without metadata."""

    priority=128 # Rather generic file format so make it a low priority

    def load(self,filename=None,header_line=0, data_line=1, data_delim=',', header_delim=',', **kargs):
        """Generic deliminated file loader routine.

        @param filename File to load. If None then the existing filename is used,
        if False, then a file dialog will be used.
        @param header_line The line in the file that contains the column headers.
        If None, then column headers are auotmatically generated."""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        self.data=numpy.genfromtxt(self.filename,dtype='float',delimiter=data_delim,skip_header=data_line-1)
        if header_line is not None:
            header_string=linecache.getline(self.filename, header_line)
            header_string=re.sub(r'["\n]', '', header_string)
            self.column_headers=map(lambda x: x.strip(),  header_string.split(header_delim))
        else:
            self.column_headers=["Column"+str(x) for x in range(numpy.shape(self.data)[1])]
        return self

    def save(self,filename, deliminator=','):
        """Overrides the save method to allow CSVFiles to be written out to disc (as a mininmalist output)
                @param filename Fielname to save as (using the same rules as for the load routines)
                @param deliminator Record deliniminator (defaults to a comma)
                @return A copy of itself."""
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

    def __parse_VSM(self, header_line=3, data_line=7, data_delim=' ', header_delim=','):
        """An intrernal function for parsing deliminated data without a leading column of metadata.copy
        @param header_line is the line on which the column headers are recorded (default 3)
        @param data_line is the first line of tabulated data (default 7)
        @param data_delim is the deliminator for the data rows (default = space)
        @param header_delim is the deliminator for the header values (default = tab)
        @return Nothing, but updates the current instances data

        NBThe default values are configured fir read VSM data files
        """
        f=fileinput.FileInput(self.filename) # Read filename linewise
        self['Timestamp']=f.next().strip()
        check=datetime.strptime(self["Timestamp"], "%a %b %d %H:%M:%S %Y")
        f.next()
        header_string=f.next()
        header_string=re.sub(r'["\n]', '', header_string)
        unit_string=f.next()
        unit_string=re.sub(r'["\n]', '', unit_string)
        column_headers=zip(header_string.split(header_delim), unit_string.split(header_delim))
        self.column_headers=map(lambda x: x[0].strip()+" ("+x[1].strip()+")",  column_headers)
        f.next()
        f.next()

        self.data=numpy.genfromtxt(f,dtype='float',delimiter=data_delim,skip_header=data_line-1, missing_values=['         ---'], invalid_raise=False)
        f.close()


    def load(self,filename=None,*args, **kargs):
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
        """Just call the parent class but with the right parameters set"""
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
        """Just call the parent class but with the right parameters set"""
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
        """Just call the parent class but with the right parameters set"""
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
    def load(self,filename=None,*args, **kargs):
        """Reads a .scf file produced by the Renishaw Raman system (amongs others)

        @param filename String containing file to be loaded
        @param args Pass through all other arguements
        @return An instance of Stoner.DataFile with the data loaded

        @todo Implement the second form of the file that stores multiple x-y curves in the one file.

        @note Metadata keys are pretty much as specified in the spc.h file that defines the filerformat."""
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
        """Reads a TDMS File

        @param filename String containing file to be loaded
        @param args Pass through all other arguements
        @return An instance of Stoner.DataFile with the data loaded"""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        f=fileinput.FileInput(self.filename) # Read filename linewise
        line=f.netxt()
        assert line[0:4] == "TDSm"
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
        
        @param filename String containing the file to be laoded
        @param args Passthrough of of all other positional arguments
        @kargs Holds all keyword arguments"""
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
            except Exception, e:
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

        @param filename String containing file to be loaded
        @param args Pass through all other arguements
        @return An instance of Stoner.DataFile with the data loaded

        Format is ini file like but not enough to do standard inifile processing - in particular one can have multiple sections with the same name (!)
    """
        from ast import literal_eval
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
        @param filename  Filename to be loaded

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
        """Just call the parent class but with the right parameters set"""
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


