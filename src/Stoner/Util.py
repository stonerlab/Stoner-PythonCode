# Utility Functions for Stoner Module
# $Id: Util.py,v 1.5 2011/05/08 18:25:00 cvs Exp $
# $Log: Util.py,v $
# Revision 1.5  2011/05/08 18:25:00  cvs
# Correct the Raman load to include the last point in the Xdata
#
# Revision 1.4  2011/05/06 22:35:37  cvs
# Add some tags to the Util file to moniyor version and log
#

def read_spc_File(filename):
    """Reads a .scf file produced by the Renishaw Raman system (amongs others)
    
    @param filename String containing file to be loaded
    @return An instance of Stoner.DataFile with the data loaded
    
    @TODO Implement the second form of the file that stores multiple x-y curves in the one file.
    
    @NB Metadata keys are pretty much as specified in the spc.h file that defines the filerformat."""
    from .Core import DataFile
    import struct
    import numpy
    
    # Open the file and read the main file header and unpack into a dict
    f=open(filename, 'r')
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
            xdata=numpy.array(struct.unpack(str(header['fnpts'])+"f"))
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
        else:
            y_width=4
            y_fmt='i'

        for j in range(n): # We have n sub-scans
            # Read the subheader and import into the main metadata dictionary as scan#:<subheader item>
            subhdr=struct.unpack('BBHfffIIf4s', f.read(32))
            subheader=dict(zip(["scan"+str(j)+":"+x for x in subhdr_keys], subhdr))
            
            # Now read the y-data
            exponent=subheader["scan"+str(j)+':subexp']
            if int(exponent) & 128: # Data is unscaled direct floats
                ydata=numpy.array(struct.unpack(str(pts)+"f", f.read(pts*y_width)))
            else: # Data is scaled by exponent
                ydata=numpy.array(struct.unpack(str(pts)+y_fmt, f.read(pts*y_width)))*(2**exponent)
            
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
            for line in logtext.split("\r\n"):
                if "=" in line:
                    key, value= line.split('=')
                    header[key]=value
        # Ok now build the Stoner.DataFile instance to return
        d=DataFile()
        d.data=data
        # The next bit generates the metadata. We don't just copy the metadata because we need to figure out the typehints first - hence the loop here to call DataFile.__setitem()
        for x in header:
            d[x]=header[x]
        d.column_headers=column_headers
        f.close() # tidy up and return
        return d
        
            
    

def read_XRD_File(filename):
    """Reads an XRD datafile as produced by the Brucker diffractometer
    
    @param filename String containing file to be loaded
    @return An instance of Stoner.DataFile with the data loaded
    
    Format is ini file like but not enough to do standard inifile processing - in particular one can have multiple sections with the same name (!)
"""
    from .Core import DataFile
    import fileinput
    import re
    import numpy
    from ast import literal_eval
    
    sh=re.compile(r'\[(.+)\]') # Regexp to grab section name
    f=fileinput.FileInput(filename) # Read filename linewise
    d=DataFile() # initialise the DataFile Instance
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
                    d=d+numpy.array([float(parts[0].strip()), float(parts[1].strip())])
                else: # Other sections contain metadata
                    parts=line.split('=')
                    key=parts[0].strip()
                    data=parts[1].strip()
                    try: # A bit clusmy but basically tries to interpret as an int, float, boolean or string in that order
                        data2=int(data)
                    except ValueError:
                        try:
                            data2=float(data)
                        except ValueError:
                            try:
                                data2=(['true', 'on', 'yes', 'false', 'off', 'no'].index(data.lower())<3)
                            except ValueError:
                                data2=data   
                    d[section+":"+key]=data2 # Keynames in main metadata are section:key
    f.close()# Cleanup
    d.column_headers=['Angle', 'Counts'] # Assume the columns were Angles and Counts
    return d    

