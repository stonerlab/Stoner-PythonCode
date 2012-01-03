####################################################
## FileFormats - sub classes of DataFile for different machines
# $Id: FileFormats.py,v 1.9 2012/01/03 12:41:50 cvs Exp $
# $Log: FileFormats.py,v $
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

from .Core import DataFile
from .pyTDMS import read as tdms_read


class VSMFile(DataFile):
    """Extends DataFile to open VSM Files"""

    def __parse_VSM(self, header_line=3, data_line=7, data_delim=' ', header_delim=','):
        """An intrernal function for parsing deliminated data without a leading column of metadata.copy
        @param header_line is the line on which the column headers are recorded (default 3)
        @param data_line is the first line of tabulated data (default 7)
        @data_delim is the deliminator for the data rows (default = space)
        @param header_delim is the deliminator for the header values (default = tab)
        @return Nothing, but updates the current instances data

        NBThe default values are configured fir read VSM data files
        """
        f=fileinput.FileInput(self.filename) # Read filename linewise
        self['Timestamp']=f.next().strip()
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


    def load(self,filename=None,*args):
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        self.__parse_VSM()
        return self

class BigBlueFile(DataFile):
    """Extends DataFile to load files from BigBlue"""

    def __parse_plain_data(self, header_line=3, data_line=7, data_delim=' ', header_delim=','):
        """An intrernal function for parsing deliminated data without a leading column of metadata.copy
        @param header_line is the line on which the column headers are recorded (default 3)
        @param data_line is the first line of tabulated data (default 7)
        @data_delim is the deliminator for the data rows (default = space)
        @param header_delim is the deliminator for the header values (default = tab)
        @return Nothing, but updates the current instances data

        NBThe default values are configured fir read VSM data files
        """
        header_string=linecache.getline(self.filename, header_line)
        header_string=re.sub(r'["\n]', '', header_string)
        self.column_headers=map(lambda x: x.strip(),  header_string.split(header_delim))
        self.data=numpy.genfromtxt(self.filename,dtype='float',delimiter=data_delim,skip_header=data_line-1)


    def load(self,filename=None,*args):
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        self.__parse_plain_data(header_line,data_line, data_delim=',', header_delim=',')
        return self

class SPCFile(DataFile):
    """Extends DataFile to load SPC files from Raman"""
    def load(self,filename=None,*args):
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

    def load(self, filename=None, *args):
        """Reads a TDMS File

        @param filename String containing file to be loaded
        @param args Pass through all other arguements
        @return An instance of Stoner.DataFile with the data loaded"""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        (metadata, data)=tdms_read(self.filename)
        for key in metadata:
            self.metadata[key]=metadata[key]
        self.column_headers = list()
        for column in data:
            nd=data[column]
            print nd
            self.add_column(nd, column)
        return self

class XRDFile(DataFile):
    """Loads Files from a Brucker D8 Discovery X-Ray Diffractometer"""

    def load(self,filename=None,*args):
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
                        self=self+dataline
                    else: # Other sections contain metadata
                        parts=line.split('=')
                        key=parts[0].strip()
                        data=parts[1].strip()
                        self[section+":"+key]=data # Keynames in main metadata are section:key - use theDataFile magic to do type determination
        self.column_headers=['Angle', 'Counts'] # Assume the columns were Angles and Counts

        f.close()# Cleanup
        return self

class CSVFile(DataFile):
    """A subclass of DataFiule for loading generic deliminated text fiules without metadata."""
    def load(self,filename=None,header_line=0, data_line=1, data_delim=',', header_delim=','):
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        header_string=linecache.getline(self.filename, header_line)
        header_string=re.sub(r'["\n]', '', header_string)
        self.column_headers=map(lambda x: x.strip(),  header_string.split(header_delim))
        self.data=numpy.genfromtxt(self.filename,dtype='float',delimiter=data_delim,skip_header=data_line-1)
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

class BNLFile(DataFile):
    """Author Rowan 12/2011
    Creates BNLFile a subclass of DataFile that caters for files in the format given
    by BNL.

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
        print fp
        self.line_numbers=[0,0,0,0,0]
        counter=0
        for line in fp:
            counter+=1
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
        self.__setitem__('Stype',string.join(scanLine.split()[2:]))
        self.__setitem__('Sdatetime',dateLine[3:])
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


    def load(self,filename):        #fileType omitted, implicit in class call
        """BNLFile.load(filename)

        Overwrites load method in DataFile class, no header positions and data
        positions are needed because of the hash title structure used in BNL files.

        Normally its good to use _parse_plain_data method from DataFile class
        to load data but unfortunately Brookhaven data isn't very plain so there's
        a new method below.
        """
        self.filename=filename
        self.__parse_BNL_data() #call an internal function rather than put it in load function
        return self


