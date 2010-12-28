# Utility Functions for Stoner Module

def read_XRD_File(filename):
    from . import DataFile
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

