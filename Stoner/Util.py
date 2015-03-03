# -*- coding: utf-8 -*-
"""
Stoner.Utils - a module of some slightly experimental routines that use the Stoner classes

Created on Tue Oct 08 20:14:34 2013

@author: phygbu
"""

from Stoner.Core import DataFile as _DF_
from Stoner.Analysis import AnalyseFile as _AF_
from Stoner.Plot import PlotFile as _PF_
import Stoner.FileFormats as _SFF_
from Stoner.Folders import DataFolder as _SF_
from Stoner.Fit import linear
from numpy import log10,floor,max,abs,sqrt,diag,argmax
from scipy.integrate import trapz

class Data(_AF_,_PF_):
    """A merged class of AnalyseFile and PlotFile which also has the FielFormats loaded redy for use.
    This 'kitchen-sink' class is intended as a convenience for writing scripts that carry out both plotting and
    analysis on data files."""
    pass


def split_up_down(data,col=None,folder=None):
    """Splits the DataFile data into several files where the column \b col is either rising or falling

    Args:
        data (:py:class:`Stoner.Core.DataFile`): object containign the data to be sorted
        col (index): is something that :py:meth:`Stoner.Core.DataFile.find_col` can use
        folder (:py:class:`Stoner.Folders.DataFolder` or None): if this is an instance of :py:class:`Stoner.Folders.DataFolder` then add
            rising and falling files to groups of this DataFolder, otherwise create a new one

    Returns:
        A :py:class:`Sonter.Folder.DataFolder` object with two groups, rising and falling
    """
    if col is None:
        col=data.setas["x"]
    a=_AF_(data)
    width=len(a)/10
    if width%2==0: # Ensure the window for Satvisky Golay filter is odd
        width+=1
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

def format_error(value,error,latex=False,mode="float",units="",prefix=""):
    """This handles the printing out of the answer with the uncertaintly to 1sf and the
    value to no more sf's than the uncertainty.

    Args:
        value (float): The value to be formated
        error (float): The uncertainty in the value
        latex (bool): If true, then latex formula codes will be used for +/- symbol for matplotlib annotations
        mode (string): If "float" (default) the number is formatted as is, if "eng" the value and error is converted
            to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a scientifc,
            i.e. mantissa and exponent format is used.
        units (string): A suffix providing the units of the value. If si mode is used, then appropriate si prefixes are
            prepended to the units string. In LaTeX mode, the units string is embedded in \\mathrm
        prefix (string): A prefix string that should be included before the value and error string. in LaTeX mode this is
            inside the math-mode markers, but not embedded in \\mathrm.

    Returns:
        String containing the formated number with the eorr to one s.f. and value to no more d.p. than the error.
    """
    if error==0.0: # special case for zero uncertainty
        return repr(value)
    #Sort out special fomatting for different modes
    if mode=="float": # Standard
        suffix_val=""
    elif mode=="eng": #Use SI prefixes
        v_mag=floor(log10(abs(value))/3.0)*3.0
        prefixes={3:"k",6:"M",9:"G",12:"T",15:"P",18:"E",21:"Z",24:"Y",
            -3:"m",-6:"\\mu",-9:"n",-12:"p",-15:"f",-18:"a",-21:"z",-24:"y"}
        if v_mag in prefixes:
            if latex:
                suffix_val=r"\mathrm{{{{{}}}}}".format(prefixes[v_mag])
            else:
                suffix_val=prefixes[v_mag]
            value/=10**v_mag
            error/=10**v_mag
        else: # Implies 10^-3<x<10^3
            suffix_val=""
    elif mode=="sci": # Scientific mode - raise to common power of 10
        v_mag=floor(log10(abs(value)))
        if latex:
            suffix_val=r"\times 10^{{{{{}}}}}".format(int(v_mag))
        else:
            suffix_val="E{} ".format(int(v_mag))
        value/=10**v_mag
        error/=10**v_mag
    else: # Bad mode
        raise RuntimeError("Unrecognised mode: {} in format_error".format(mode))

    # Now do the rounding of the value based on error to 1 s.f.
    e2=error
    u_mag=floor(log10(abs(error))) #work out the scale of the error
    error=round(error/10**u_mag)*10**u_mag # round the error, but this could round to 0.x0
    u_mag=floor(log10(error)) # so go round the loop again
    error=round(e2/10**u_mag)*10**u_mag # and get a new error magnitude
    value=round(value/10**u_mag)*10**u_mag
    u_mag=min(0,u_mag) # Force integer results to have no dp

    #Protect {} in units string
    units=units.replace("{","{{").replace("}","}}")
    prefix=prefix.replace("{","{{").replace("}","}}")
    if latex: # Switch to latex math mode symbols
        val_fmt_str=r"${}{{:.{}f}}\pm ".format(prefix,int(abs(u_mag)))
        if units!="":
            suffix_fmt=r"\mathrm{{{{{}}}}}".format(units)
        else:
            suffix_fmt=""
        suffix_fmt+="$"
    else: # Plain text
        val_fmt_str=r"{}{{:.{}f}}+/-".format(prefix,int(abs(u_mag)))
        suffix_fmt=units
    if u_mag<0: # the error is less than 1, so con strain decimal places
        err_fmt_str=r"{:."+str(int(abs(u_mag)))+"f}"
    else: # We'll be converting it to an integer anyway
        err_fmt_str=r"{}"
    fmt_str=val_fmt_str+err_fmt_str+suffix_val+suffix_fmt
    if error>=1.0:
        error=int(error)
        value=int(value)
    return fmt_str.format(value,error)

Hickeyify=format_error

def ordinal(value):
    """Format an integer into an ordinal string.

    Args:
        value (int): Number to be written as an ordinal string

    Return:
        Ordinal String such as '1st','2nd' etc."""
    if not isinstance(value,int):
        raise ValueError

    last_digit=value%10
    if value%100 in [11,12,13]:
        suffix="th"
    else:
        suffix=["th","st","nd","rd","th","th","th","th","th","th"][last_digit]

    return "{}{}".format(value,suffix)

def hysteresis_correct(data,correct_background=True,correct_H=True, saturation_fraction=0.2):
    """Peform corrections to a hysteresis loop.

    Args:
        data (DataFile): The data containing the hysteresis loop. The :py:attr:`DataFile.setas` attribute
            should be set to give the H and M axes as x and y.

    Keyword Arguments:
        correct_background (bool): Correct for a diamagnetic or paramagnetic background to the hystersis loop
            also recentres the loop about zero moment.
        correct_H (bool): Finds the co-ercive fields and sets them to be equal and opposite. If the loop is sysmmetric
            this will remove any offset in filed due to trapped flux
        saturated_fraction (float): The fraction of the horizontal (field) range where the moment can be assumed to be
            fully saturated.

    Returns:
        The original loop with the x and y columns replaced with corrected data and extra metadata added to give the
        background suceptibility, offset in moment, co-ercive fields and saturation magnetisation.
    """

    if isinstance(data,_DF_):
        cls=data.__class__
    else:
        cls=Data
    data=Data(data)


    xc=data.find_col(data.setas["x"])
    yc=data.find_col(data.setas["y"])

    mx=max(data.x)*(1-saturation_fraction)
    mix=min(data.x)*(1-saturation_fraction)
    p1,pcov=data.curve_fit(linear,absolute_sigma=False,bounds=lambda x,r:x>mx)
    perr1=diag(pcov)
    p2,pcov=data.curve_fit(linear,absolute_sigma=False,bounds=lambda x,r:x<mix)
    perr2=diag(pcov)
    pm=(p1+p2)/2
    perr=sqrt(perr1+perr2)
    data["Ms"]=(abs(p1[0])+abs(p2[0]))/2
    low_m=p2[0]+perr[0]
    high_m=p1[0]-perr[0]
    data["Ms Error"]=perr[0]
    data["Offset Moment"]=pm[0]
    data["Offset Moment Error"]=perr[0]
    data["Background susceptibility"]=pm[1]
    data["Background Susceptibility Error"]=perr[1]

    if correct_background:
        new_y=data.y-linear(data.x,*pm)
        data.data[:,yc]=new_y


    hc1=data.threshold(0.0,rising=True,falling=False)
    hc2=data.threshold(0.0,rising=False,falling=True)

    if correct_H:
        hc_mean=(hc1+hc2)/2
        data["Field Offset"]=hc_mean
        data.data[:,xc]=data.x-hc_mean
    else:
        hc_mean=0.0
    data["Hc"]=(hc1-hc_mean,hc2-hc_mean)

    bh=(-data.x)*data.y
    i=argmax(bh)
    data["BH_Max"]=max(bh)
    data["BH_Max_H"]=data.x[i]
    mr1=data.threshold(0.0,col=xc,xcol=yc,rising=True,falling=False)
    mr2=data.threshold(0.0,col=xc,xcol=yc,rising=False,falling=True)

    data["Remenance"]=abs((mr2-mr1)/2)

    h_sat_data=data.search(data.setas["y"],lambda x,r:low_m<=x<=high_m)[:,xc]
    data["H_sat"]=(min(h_sat_data),max(h_sat_data))

    data["Area"]=data.integrate()
    return cls(data)