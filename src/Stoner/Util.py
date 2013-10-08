# -*- coding: utf-8 -*-
"""
Stoner.Utils - a module of some slightly experimental routines that use the Stoner classes

Created on Tue Oct 08 20:14:34 2013

@author: phygbu
"""

from Stoner.Analysis import AnalyseFile as _AF_
from Stoner.Folders import DataFolder as _SF_

def split_up_down(data,col,folder=None):
    """Splits the DataFile data into several files where the column \b col is either rising or falling
    
    @param data is a \b Stoner.Core.DataFile or subclass
    @param col is something that Stoner.Core.DataFile.find_col can use
    @param folder if this is an instance of \b Stoner.Folders.DataFolder then add 
    rising and falling files to groups of this DataFolder, otherwise create a new one
    @return A DataFolder object with two groups, rising and falling
    """
    a=_AF_(data)
    width=len(a)/10
    peaks=list(a.peaks(col,width,peaks=True,troughs=False))
    troughs=list(a.peaks(col,width,peaks=False,troughs=True))
    if len(peaks)>0 and len(troughs)>0: #Ok more than up down here
        order=peaks[0]<troughs[0]
    elif len(peaks)>0: #Rise then fall
        order=True
    elif len(troughs)>0: # Fall then rise
        order=False
    else: #No peaks or troughs so just return a single rising
        return ([data],[])
    splits=[0,len(a)]
    splits.extend(peaks)
    splits.extend(troughs)
    splits.sort()
    if not isinstance(folder,_SF_): # Create a new DataFolder object
        output=_SF_()
    else:
        output=folder
    output.add_group("rising")
    output.add_group("falling")
    for i in range(1,len(splits),2):
        working1=data.clone
        working2=data.clone
        working1.data=data.data[splits[i-1]:splits[i],:]
        working2.data=data.data[splits[i]:splits[1+1],:]
        if not order:
            (working1,working2)=(working2,working1)
        output.groups["rising"].files.append(working1)
        output.groups["falling"].files.append(working2)
    return output
