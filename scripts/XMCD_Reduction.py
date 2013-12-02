# Normaliser with Stoner module
import Stoner.Core as SC
import Stoner.Folders as SF
import Stoner.Analysis as SA
import Stoner.Plot as SP

import re
import numpy as np
import scipy

def helicity(f):
    """Look for either pos or neg in the first column header."""
    if "pos" in f.column_headers[0]:
        return "Pos"
    elif 'neg' in f.column_headers[0]:
        return "Neg"
    else:
        return "Neither"

def position(f):
    """"Look for meta data named magj1yins and us this rounded to 1dp as a key """
    if "magj1yins" in f:
        if f["magj1yins"]==-8.5:
            return "Perp"
        else:
            return "Para"
    else:
        return "None"

def temp(f):
    """Get the temperature data in bins"""
    if "sensor_temp" in f:
        return str(np.round(f["itc2"]))+"K"
    else:
        return "None"

def cycle(f):
    """If run >=100 then we're on the second cycle"""
    if f["run"]>=49100:
        return "cycle2"
    else:
        return "cycle1"


def alt_norm(f,trail,**kargs):
    """Just do the normalisation per file and report run number and normalisation constants"""
    run=f["run"]
    signal=kargs["signal"]
    lfit=kargs["lfit"]
    rfit=kargs["rfit"]
    ec=0

    md=f.find_col(signal)
    f=SA.AnalyseFile(f)
    coeffs=f.polyfit(ec,md,1,lambda x,y:lfit[0]<=x<=lfit[1])
    linearfit=scipy.poly1d(coeffs)
    f.add_column(lambda r:r[md]-linearfit(r[ec]),'minus linear')
    highend=f.mean('minus',lambda r:rfit[0]<=r[ec]<=rfit[1])
    ret=[run]
    ret.extend(coeffs)
    ret.append(highend)
    return ret




def norm_group(pos,trail,**kargs):
    """Takes the drain current for each file in group and builds an analysis file and works out the mean drain"""
    if "signal" in kargs:
        signal=kargs["signal"]
    else:
            signal="fluo"
    lfit=kargs["lfit"]
    rfit=kargs["rfit"]

    posfile=SA.AnalyseFile()
    posfile.metadata=pos[0].metadata
    posfile=posfile&pos[0].column(0)
    posfile.column_headers=['Energy']
    for f in pos:
        print str(f["run"])+str(f.find_col(signal))
        posfile=posfile&f.column(signal)
    posfile.add_column(lambda r:np.mean(r[1:]),"mean drain")
    ec=posfile.find_col('Energy')
    md=posfile.find_col('mean drain')
    posfile=SA.AnalyseFile(posfile)
    linearfit=scipy.poly1d(posfile.polyfit(ec,md,1,lambda x,y:lfit[0]<=x<=lfit[1]))
    posfile.add_column(lambda r:r[md]-linearfit(r[ec]),'minus linear')
    highend=posfile.mean('minus',lambda r:rfit[0]<=r[ec]<=rfit[1])
    ml=posfile.find_col('minus linear')
    posfile.add_column(lambda r:r[ml]/highend,"normalised")
    if "group_key" in kargs:
        posfile[kargs["group_key"]]=pos.key
    return posfile

def asym(grp,trail,**kargs):
    """Takes a group of two files for Pos and Neg helicity and calcualtes the Asymmetry ratio."""
    posfile=grp['Pos']
    negfile=grp['Neg']
    pdat=posfile.column('normalised')
    ndat=negfile.column('normalised')
    edat=posfile.column('Energy')
    ret=posfile.clone
    ret.column_headers=[]
    ret.data=np.array([])
    ret=ret&edat&pdat&ndat
    ret.column_headers=['Energy','I+','I-']
    ret.add_column(lambda r:(r[1]-r[2]),"Asymmetry")
    ret.filename="-".join(trail)
    if "group_key" in kargs:
        ret[kargs["group_key"]]=grp.key
    ret.title=ret.filename
    if "save" in kargs and kargs["save"]:
        ret.save()
    return ret

def collate(grp,trail,**kargs):
    grp.sort()
    final=SC.DataFile()
    final.add_column(grp[0].column('Energy'),'Energy')
    for g in grp:
        final.add_column(g.column('Asym'),g.title)
    if "group_key" in kargs:
        final[kargs["group_key"]]=grp.key
    final["path"]=trail
    if "save" in kargs and kargs["save"]:
        final.save(kargs["filename"])
    return final

 #The start and end runs for this batch
startrun=49173
endrun=52000
# Which column are we analysing ?
signal='fluo'
# A filename pattern that will grab the run number from the filename
pattern=re.compile('i10-(?P<run>\d*)\.dat')
#The Data spool directory
dir='C:\Data\data'
#Set the limits used on the normalisation
rfit=(660,670)
lfit=(615,630)

#Read the directory of data files and sort by run number
fldr=SF.DataFolder(dir,pattern=pattern,read_means=True)
fldr.sort("run")
# Remove files outside of the run number range
fldr.filterout(lambda f: f['run']>endrun or f['run']<startrun)
# group the files by position, temperatures and polarisations
fldr.group([position,temp,helicity])
#Normalise the files grouped by helicity, temperature and position and produce one averaged file
# for each position, temperature and helicity
fldr.walk_groups(norm_group,group=True,replace_terminal=True,walker_args={"signal":signal,"lfit":lfit,"rfit":rfit})
# Calcualte the asymmetry by position and temperature and produce one file per positiona nd temeprature
# Set the walker_args to {"save":True} to save this file to disc
fldr.walk_groups(asym,group=True,replace_terminal=True,walker_args={"save":True,"group_key":"temperature"})
# Collate the asummetry curves into one file and save it
fldr.walk_groups(collate,group=True,replace_terminal=True,walker_args={"group_key":"Position"})

for f in fldr:
    f.filename="Run 2 Compiled Fluorescence Data {}.txt".format(f["Position"])
    f.save()
#norm_data=fldr.walk_groups(alt_norm,walker_args={"rfit":(660,670),"lfit":(615,630),"signal":"fluo"})
