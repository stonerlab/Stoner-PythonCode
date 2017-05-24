# -*- coding: utf-8 -*-
"""
Stoner.Utils - a module of some slightly experimental routines that use the Stoner classes

Created on Tue Oct 08 20:14:34 2013

@author: phygbu
"""

from .compat import int_types
from .tools import format_error
import Stoner.Core  as _SC_
from .Folders import DataFolder as _SF_
from .Fit import linear
from numpy import max, sqrt, diag, argmax, mean,array #pylint: disable=redefined-builtin
from scipy.stats import sem

def _up_down(data):
    """Split data d into rising and falling sections and then add and sort the two sets.

    Args:
        data (Data): DataFile like object with x and y columns set

    Returns:
        (Data, Data): Tuple of two DataFile like instances for the rising and falling data.
    """
    f=split_up_down(data)

    ret=[None,None]
    for i,grp in enumerate(["rising","falling"]):
        ret[i]=f[grp][0]
        for d in f[grp][1:]:
            ret[i]=ret[i]+d
        ret[i].sort(data.setas._get_cols('xcol'))
        ret[i].setas=f["rising"][0].setas.clone #hack due to bug in sort wiping the setas info
    return ret




def split(data, col=None, folder=None, spliton=0, rising=True, falling=False, skip=0):
    """Splits the DataFile data into several files where the column *col* is either rising or falling

    Args:
        data (:py:class:`Stoner.Core.DataFile`): object containign the data to be sorted
        col (index): is something that :py:meth:`Stoner.Core.DataFile.find_col` can use
        folder (:py:class:`Stoner.Folders.DataFolder` or None): if this is an instance of :py:class:`Stoner.Folders.DataFolder` then add
            rising and falling files to groups of this DataFolder, otherwise create a new one
        spliton (str or float): Define where to split the data, 'peak' to split on peaks, 'trough' to split
            on troughs, 'both' to split on peaks and troughs or number to split at that number
        rising (bool): whether to split on threshold crossing when data is rising
        falling (bool): whether to split on threshold crossing when data is falling
        skip (int): skip this number of splitons each time. eg skip=1 picks out odd crossings
    Returns:
        A :py:class:`Sonter.Folder.DataFolder` object with two groups, rising and falling
    """
    if col is None:
        col = data.setas["x"]
    d=_SC_.Data(data)
    if not isinstance(folder, _SF_):  # Create a new DataFolder object
        output = _SF_()
    else:
        output = folder

    if isinstance(spliton, int_types+(float,)):
        spl=d.threshold(threshold=float(spliton),col=col,rising=rising,falling=falling,all_vals=True)

    elif spliton in ['peaks','troughs','both']:
        width = len(d) / 10
        if width % 2 == 0:  # Ensure the window for Satvisky Golay filter is odd
            width += 1
        if spliton=='peaks':
            spl = list(d.peaks(col, width, xcol=False, peaks=True, troughs=False))
        elif spliton=='troughs':
            spl = list(d.peaks(col, width, xcol=False, peaks=False, troughs=True))
        else:
            spl = list(d.peaks(col, width, xcol=False, peaks=True, troughs=True))

    else:
        raise ValueError('Did not recognise spliton')

    spl = [spl[i] for i in range(len(spl)) if i%(skip+1)==0]
    spl.extend([0,len(d)])
    spl.sort()
    for i in range(len(spl)-1):
        tmp=d.clone
        tmp.data=tmp[spl[i]:spl[i+1]]
        output.files.append(tmp)
    return output



def split_up_down(data, col=None, folder=None):
    """Splits the DataFile data into several files where the column *col* is either rising or falling

    Args:
        data (:py:class:`Stoner.Core.DataFile`): object containign the data to be sorted
        col (index): is something that :py:meth:`Stoner.Core.DataFile.find_col` can use
        folder (:py:class:`Stoner.Folders.DataFolder` or None): if this is an instance of :py:class:`Stoner.Folders.DataFolder` then add
            rising and falling files to groups of this DataFolder, otherwise create a new one

    Returns:
        A :py:class:`Sonter.Folder.DataFolder` object with two groups, rising and falling
    """
    a = _SC_.Data(data)
    if col is None:
        _=a._col_args()
        col=_.xcol
    width = len(a) / 10
    if width % 2 == 0:  # Ensure the window for Satvisky Golay filter is odd
        width += 1
    setas=a.setas.clone
    a.setas=""
    peaks = list(a.peaks(col, width,xcol=None, peaks=True, troughs=False,full_data=False))
    troughs = list(a.peaks(col, width, xcol=None, peaks=False, troughs=True,full_data=False))
    a.setas=setas
    if len(peaks) > 0 and len(troughs) > 0:  #Ok more than up down here
        order = peaks[0] < troughs[0]
    elif len(peaks) > 0:  #Rise then fall
        order = True
    elif len(troughs) > 0:  # Fall then rise
        order = False
    else:  #No peaks or troughs so just return a single rising
        ret=_SF_(readlist=False)
        ret+=data
        return ret
    splits = [0, len(a)]
    splits.extend(peaks)
    splits.extend(troughs)
    splits.sort()
    splits=[int(s) for s in splits]
    if not isinstance(folder, _SF_):  # Create a new DataFolder object
        output = _SF_(readlist=False)
    else:
        output = folder
    output.add_group("rising")
    output.add_group("falling")

    if order:
        risefall=["rising","falling"]
    else:
        risefall=["falling","rising"]
    for i in range(len(splits)-1):
        working=data.clone
        working.data = data.data[splits[i]:splits[i+1],:]
        output.groups[risefall[i%2]].append(working)
    return output



Hickeyify = format_error


def ordinal(value):
    """Format an integer into an ordinal string.

    Args:
        value (int): Number to be written as an ordinal string

    Return:
        Ordinal String such as '1st','2nd' etc.
    """
    if not isinstance(value, int):
        raise ValueError

    last_digit = value % 10
    if value % 100 in [11, 12, 13]:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"][last_digit]

    return "{}{}".format(value, suffix)


def hysteresis_correct(data, **kargs):
    """Peform corrections to a hysteresis loop.

    Args:
        data (DataFile): The data containing the hysteresis loop. The :py:attr:`DataFile.setas` attribute
            should be set to give the H and M axes as x and y.

    Keyword Arguments:
        correct_background (bool): Correct for a diamagnetic or paramagnetic background to the hystersis loop
            also recentres the loop about zero moment (default True).
        correct_H (bool): Finds the co-ercive fields and sets them to be equal and opposite. If the loop is sysmmetric
            this will remove any offset in filed due to trapped flux (default True)
        saturated_fraction (float): The fraction of the horizontal (field) range where the moment can be assumed to be
            fully saturated. If an integer is given it will use that many data points at the end of the loop.
        xcol (column index): Column with the x data in it
        ycol (column_index): Column with the y data in it
        setas (string or iterable): Column assignments.

    Returns:
        The original loop with the x and y columns replaced with corrected data and extra metadata added to give the
        background suceptibility, offset in moment, co-ercive fields and saturation magnetisation.
    """
    if isinstance(data, _SC_.DataFile):
        cls = data.__class__
    else:
        cls = _SC_.Data
    data = cls(data)

    if "setas" in kargs: # Allow us to override the setas variable
        data.setas=kargs.pop("setas")

    xcol=kargs.pop("xcol",None)
    ycol=kargs.pop("ycol",None)
    #Get xcol and ycols from kargs if specified
    _=data._col_args(xcol=xcol,ycol=ycol)
    data.setas(x=_.xcol,y=_.ycol)
    #Split into two sets of data:

    #Get other keyword arguments
    correct_background=kargs.pop("correct_background",True)
    correct_H=kargs.pop("correct_H",True)
    saturation_fraction=kargs.pop("saturation_fraction",0.2)

    while True:
        up,down=_up_down(data)

        if isinstance(saturation_fraction,int_types) and  saturation_fraction>0:
            saturation_fraction=saturation_fraction/len(up)+0.001 #add 0.1% to ensure we get the point
        mx = max(data.x) * (1 - saturation_fraction)
        mix = min(data.x) * (1 - saturation_fraction)


        up._push_mask(lambda x, r: x >= mix)
        pts=up.x.count()
        up._pop_mask()
        assert pts>=3,"Not enough points in the negative saturation state.(mix={},pts={},x={})".format(mix,pts,up.x)

        down._push_mask(lambda x, r: x <= mx)
        pts=down.x.count()
        down._pop_mask()
        assert pts>=3,"Not enough points in the positive saturation state(mx={},pts={},x={})".format(mx,pts,down.x)

        #Find upper branch saturated moment slope and offset
        p1, pcov = data.curve_fit(linear, absolute_sigma=False, bounds=lambda x, r: x < mix)
        perr1 = diag(pcov)

        #Find lower branch saturated moment and offset
        p2, pcov = data.curve_fit(linear, absolute_sigma=False, bounds=lambda x, r: x > mx)
        perr2 = diag(pcov)
        if p1[0]>p2[0]:
            data.y=-data.y
        else:
            break


    #Find mean slope and offset
    pm = (p1 + p2) / 2
    perr = sqrt(perr1 + perr2)
    Ms=array([p1[0],p2[0]])
    Ms=list(Ms-mean(Ms))



    data["Ms"] = Ms #mean(Ms)
    data["Ms Error"] = perr[0]/2
    data["Offset Moment"] = pm[0]
    data["Offset Moment Error"] = perr[0]/2
    data["Background susceptibility"] = pm[1]
    data["Background Susceptibility Error"] = perr[1]/2

    p1=p1-pm
    p2=p2-pm

    if correct_background:
        for d in [data,up,down]:
            d.y = d.y - linear(d.x, *pm)
    else:
        for d in [up,down]: #need to do these anyway to find Hc and Hsat
            d.y = d.y - linear(d.x, *pm)
    Hc=[None,None]
    Hc_err=[None,None]
    Hsat=[None,None]
    Hsat_err=[None,None]
    m_sat=[p1[0]+perr[0],p2[0]-perr[0]]
    Mr=[None,None]
    Mr_err=[None,None]

    for i,(d,sat) in enumerate(zip([up,down],m_sat)):
        hc=d.threshold(0.,all_vals=True,rising=True,falling=True) # Get the Hc value
        Hc[i]=mean(hc)
        if hc.size>1:
            Hc_err[i]=sem(hc)
        hs=d.threshold(sat,all_vals=True,rising=True,falling=True) # Get the Hc value
        Hsat[1-i]=mean(hs) # Get the H_sat value
        if hs.size>1:
            Hsat_err[1-i]=sem(hs)
        mr=d.threshold(0.0,col=_.xcol,xcol=_.ycol,all_vals=True,rising=True,falling=True)
        Mr[i]=mean(mr)
        if mr.size>1:
            Mr_err[i]=sem(mr)


    if correct_H:
        Hc_mean=mean(Hc)
        for d in [data,up,down]:
            d.x = d.x - Hc_mean
        data["Exchange Bias offset"]=Hc_mean
    else:
        Hc_mean=0.0

    data["Hc"] = (Hc[1] - Hc_mean, Hc[0] - Hc_mean)
    data["Hsat"] = (Hsat[1] - Hc_mean, Hsat[0] - Hc_mean)
    data["Remenance"] = Mr


    bh = (-data.x) * data.y
    i = argmax(bh)
    data["BH_Max"] = max(bh)
    data["BH_Max_H"] = data.x[i]

    data["Area"] = data.integrate()
    return cls(data)