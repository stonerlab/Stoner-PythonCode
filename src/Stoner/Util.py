# -*- coding: utf-8 -*-

"""

Stoner.Utils - a module of some slightly experimental routines that use the Stoner classes



Created on Tue Oct 08 20:14:34 2013



@author: phygbu

"""



from Stoner.Analysis import AnalyseFile as _AF_

from Stoner.Folders import DataFolder as _SF_

from numpy import log10,floor



def split_up_down(data,col,folder=None):

    """Splits the DataFile data into several files where the column \b col is either rising or falling



    Args:

        data (:py:class:`Stoner.Core.DataFile`): object containign the data to be sorted

        col (index): is something that :py:meth:`Stoner.Core.DataFile.find_col` can use

        folder (:py:class:`Stoner.Folders.DataFolder` or None): if this is an instance of :py:class:`Stoner.Folders.DataFolder` then add

            rising and falling files to groups of this DataFolder, otherwise create a new one



    Returns:

        A :py:class:`Sonter.Folder.DataFolder` object with two groups, rising and falling

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



def format_error(value,error,latex=False):

    """This handles the printing out of the answer with the uncertaintly to 1sf and the

    value to no more sf's than the uncertainty.



    Args:

        value (float): The value to be formated

        error (float): The uncertainty in the value

        latex (bool): If true, then latex formula codes will be used for +/- symbol for matplotlib annotations



    Returns:

        String containing the formated number with the eorr to one s.f. and value to no more d.p. than the error.

    """

    if error==0.0: # special case for zero uncertainty

        return repr(value)

    e2=error

    u_mag=floor(log10(abs(error))) #work out the scale of the error

    error=round(error/10**u_mag)*10**u_mag # round the error, but this could round to 0.x0

    u_mag=floor(log10(error)) # so go round the loop again

    error=round(e2/10**u_mag)*10**u_mag # and get a new error magnitude

    if latex:

        if u_mag<0:

            fmt_str=r"${}\pm{:."+str(int(abs(u_mag)))+"f}$"

        else:

            fmt_str=r"{}\pm{}$"

    else:

        if u_mag<0:

            fmt_str=r"{}+/-{:."+str(int(abs(u_mag)))+"f}"

        else:

            fmt_str=r"{}+/-{}"





    value=round(value/10**u_mag)*10**u_mag

    return fmt_str.format(value,error)



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

