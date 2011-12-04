####################################################
## FielFormats - sub classes of DataFile for different machines

from .Core import DataFile
import linecache
import re
import numpy
import fileinput

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
        @return An instance of Stoner.DataFile with the data loaded
        
        @todo Implement the second form of the file that stores multiple x-y curves in the one file.
        
        @note Metadata keys are pretty much as specified in the spc.h file that defines the filerformat."""
        import struct
        import numpy
        from re import split
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

class XRDFile(DataFile):

    def load(self,filename=None,*args):
        """Reads an XRD datafile as produced by the Brucker diffractometer

        @param filename String containing file to be loaded
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
    
