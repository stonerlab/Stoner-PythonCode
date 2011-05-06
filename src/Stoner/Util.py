# Utility Functions for Stoner Module
# $Id: Util.py,v 1.4 2011/05/06 22:35:37 cvs Exp $
# $Log: Util.py,v $
# Revision 1.4  2011/05/06 22:35:37  cvs
# Add some tags to the Util file to moniyor version and log
#

def read_spc_File(filename):
    from .Core import DataFile
    import struct
    import numpy
    f=open(filename, 'r')
    spchdr=struct.unpack('BBBciddiBBBBi9s9sH8f30s130siiBBHf48sfifB187s', f.read(512))
    keys=("ftflgs","fversn","fexper","fexp","fnpts","ffirst","flast","fnsub","fxtype","fytype","fztype","fpost","fres","fsource","fpeakpt","fspare1","fspare2","fspare3","fspare4","fspare5","fspare6","fspare7","fspare8","fcm","nt","fcatx","flogoff","fmods","fprocs","flevel","fsampin","ffactor","fmethod","fzinc","fwplanes","fwinc","fwtype","fwtype","fresv")
    header=dict(zip(keys, spchdr))
    if header['ftflgs'] & 64: #
        raise NotImplemented("Filetype not implemented yet !")
    else:
        if header['ftflgs'] & 128: # We need to read some X Data
            xvals=f.read(4*header['fnpts'])
            xdata=numpy.array(struct.unpack(str(header['fnpts'])+"f"))
        else: # Generate the X Data ourselves
            first=header['ffirst']
            last=header['flast']
            incr=(last-first)/header['fnpts']
            xdata=numpy.arange(first, last, incr)
        n=header['fnsub']
        pts=header['fnpts']
        data=numpy.zeros((n+1, pts))
        data[0, :]=xdata
        xvars=["Arbitrary","Wavenumber (cm-1)","Micrometers (um)","Nanometers (nm)","Seconds","Minutes","Hertz (Hz)","Kilohertz (KHz)","Megahertz (MHz)","Mass (M/z)","Parts per million (PPM)","Days","Years","Raman Shift (cm-1)","Raman Shift (cm-1)","eV","XYZ text labels in fcatxt (old 0x4D version only)","Diode Number","Channel","Degrees","Temperature (F)","Temperature (C)","Temperature (K)","Data Points","Milliseconds (mSec)","Microseconds (uSec)","Nanoseconds (nSec)","Gigahertz (GHz)","Centimeters (cm)","Meters (m)","Millimeters (mm)","Hours","Hours"]
        yvars=["Arbitrary Intensity","Interferogram","Absorbance","Kubelka-Monk","Counts","Volts","Degrees","Milliamps","Millimeters","Millivolts","Log(1/R)","Percent","Percent","Intensity","Relative Intensity","Energy","Decibel","Temperature (F)","Temperature (C)","Temperature (K)","Index of Refraction [N]","Extinction Coeff. [K]","Real","Imaginary","Complex","Complex","Transmission (ALL HIGHER MUST HAVE VALLEYS!)","Reflectance","Arbitrary or Single Beam with Valley Peaks","Emission","Emission"]
        column_headers=[xvars[header['fxtype']]]
        subhdr_keys=("subflgs","subexp","subindx", "subtime", "subnext", "subnois", "subnpts", "subscan", "subwlevel", "subresv")
        if header['ftflgs'] &1:
            y_width=2
            y_fmt='h'
        else:
            y_width=4
            y_fmt='i'
        for j in range(n):
            subhdr=struct.unpack('BBHfffIIf4s', f.read(32))
            subheader=dict(zip(["scan"+str(j)+":"+x for x in subhdr_keys], subhdr))
            exponent=subheader["scan"+str(j)+':subexp']
            if int(exponent) & 128:
                ydata=numpy.array(struct.unpack(str(pts)+"f", f.read(pts*y_width)))
            else:
                ydata=numpy.array(struct.unpack(str(pts)+y_fmt, f.read(pts*y_width)))*2**exponent
            data[j+1, :]=ydata
            header=dict(header, **subheader)
            column_headers.append("Scan"+str(j)+":"+yvars[header['fytype']])
        if header['flogoff']!=0:
            logstc=struct.unpack('IIIII44s', f.read(64))
            logstc_keys=("logsizd", "logsizm", "logtxto", "logbins", "logdsks", "logrsvr")
            logheader=dict(zip(logstc_keys, logstc))
            header=dict(header, **logheader)
            f.read(header['logbins']+header['logdsks'])
            header['Date-Time']=f.read(16)
            logtext=f.read()
            for line in logtext.split("\r\n"):
                if "=" in line:
                    key, value= line.split('=')
                    header[key]=value
        d=DataFile()
        d.data=numpy.transpose(data)
        for x in header:
            d[x]=header[x]
        d.column_headers=column_headers
        f.close()
        return d
        
            
    

def read_XRD_File(filename):
    from .Core import DataFile
    import fileinput
    import re
    import numpy
    from ast import literal_eval
    
    sh=re.compile(r'\[(.+)\]')
    f=fileinput.FileInput(filename)
    d=DataFile()
    if f.next().strip()!=";RAW4.00":
            raise RuntimeError("File Format Not Recognized !")
    drive=0
    for line in f:
        m=sh.search(line)
        if m:
            section=m.group(1)
            if section=="Drive":
                section=section+str(drive)
                drive=drive+1
            elif section=="Data":
                f.next()
            for line in f:
                if line.strip()=="":
                    break
                elif section=="Data":
                    parts=line.split(',')
                    angle=parts[0].strip()
                    counts=parts[1].strip()
                    d=d+numpy.array([float(parts[0].strip()), float(parts[1].strip())])
                else:
                    parts=line.split('=')
                    key=parts[0].strip()
                    data=parts[1].strip()
                    try:
                        data2=int(data)
                    except ValueError:
                        try:
                            data2=float(data)
                        except ValueError:
                            try:
                                data2=(['true', 'on', 'yes', 'false', 'off', 'no'].index(data.lower())<3)
                            except ValueError:
                                data2=data   
                    d[section+"."+key]=data2
    f.close()
    d.column_headers=['Angle', 'Counts']
    return d    

