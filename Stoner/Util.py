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
from numpy import log10, floor, max, abs, sqrt, diag, argmax, mean,array
from scipy.integrate import trapz
from scipy.stats import sem
from sys import float_info
from lmfit import Model
from inspect import isclass

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
        ret[i].setas=data.setas.clone
    return ret


class Data(_AF_, _PF_):
    """A merged class of AnalyseFile and PlotFile which also has the FielFormats loaded redy for use.
    This 'kitchen-sink' class is intended as a convenience for writing scripts that carry out both plotting and
    analysis on data files."""

    def format(self,key,latex=False, mode="float", units=None, prefix=None):
        """Return the contents of key pretty formatted using :py:func:`format_error`.

        Args:
            key (string): Name of a metadata key that identifies the balue to be formatted.
        latex (bool): If true, then latex formula codes will be used for +/- symbol for matplotlib annotations
        mode (string): If "float" (default) the number is formatted as is, if "eng" the value and error is converted
            to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a scientifc,
            i.e. mantissa and exponent format is used.
        units (string): A suffix providing the units of the value. If si mode is used, then appropriate si prefixes are
            prepended to the units string. In LaTeX mode, the units string is embedded in \\mathrm
        prefix (string): A prefix string that should be included before the value and error string. in LaTeX mode this is
            inside the math-mode markers, but not embedded in \\mathrm.

        Returns:
            A pretty string representation.

        The if key="key", then the value is self["key"], the error is self["key err"], the default prefix is self["key label"]+"=" or "key=",
        the units are self["key units"] or "".

        """
        try:
            value=float(self[key])
        except ValueError:
            raise KeyError("{} should be a floating point value of the metadata.",format(key))
        try:
            error=float(self[key+" err"])
        except KeyError:
            error=float_info.epsilon
        if prefix is None:
            try:
                prefix="{} = ".format(self[key+" label"])
            except KeyError:
                prefix="{} =".format(key)
        if units is None:
            try:
                units=self[key+" units"]
            except KeyError:
                units=""
        return format_error(value,error,latex,mode,units,prefix)

    def annotate_fit(self,model,x=None,y=None,z=None,prefix=None,text_only=False,**kargs):
        """Annotate a plot with some information about a fit.

        Args:
            mode (callable or lmfit.Model): The function/model used to describe the fit to be annotated.

        Keyword Parameters:
            x (float): x co-ordinate of the label
            y (float): y co-ordinate of the label
            z (float): z co-ordinbate of the label if the current axes are 3D
            prefix (str): The prefix placed ahead of the model parameters in the metadata.
            text_only (bool): If False (default), add the text to the plot and return the current object, otherwise,
                return just the text and don't add to a plot.

        Returns:
            A copy of the current Data instance if text_only is False, otherwise returns the text.

        If *prefix* is not given, then the first prefix in the metadata lmfit.prefix is used if present,
        otherwise a prefix is generated from the model.prefix attribute. If *x* and *y* are not specified then they
        are set to be 0.75 * maximum x and y limit of the plot.
        """
        if isclass(model) and issubclass(model,Model):
            model=model()
        elif isinstance(model,Model):
            pass
        elif callable(model):
            prefix=model.__name__
            model=Model(model)
        else:
            raise RuntimeError("model should be either an lmfit.Model or a callable function, not a {}".format(type(model)))

        if prefix is not None:
            prefix="" if prefix == "" else prefix+":"
        elif "lmfit.prefix" in self:
            prefix=self["lmfit.prefix"][0]
        else:
            prefix=model.prefix+":"

        if x is None:
            xl,xr=self.xlim()
            x=(xr-xl)*0.75+xl
        if y is None:
            yb,yt=self.ylim()
            y=0.5*(yt-yb)+yb


        text= "\n".join([self.format("{}{}".format(prefix,k),latex=True) for k in model.param_names])
        if not text_only:
            ax=self.fig.gca()
            if "zlim" in ax.properties():
                #3D plot then
                if z is None:
                    zb,zt=ax.properties()["zlim"]
                    z=0.5*(zt-zb)+zb
                ax.text3D(x,y,z,text)
            else:
                ax.annotate(text, xy=(x,y), **kargs)
            ret=self
        else:
            ret=text
        return ret





def split_up_down(data, col=None, folder=None):
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
        col = data.setas["x"]
    width = len(data) / 10
    if width % 2 == 0:  # Ensure the window for Satvisky Golay filter is odd
        width += 1
    peaks = list(data.peaks(col, width,xcol=False, peaks=True, troughs=False))
    troughs = list(data.peaks(col, width, xcol=False, peaks=False, troughs=True))
    if len(peaks) > 0 and len(troughs) > 0:  #Ok more than up down here
        order = peaks[0] < troughs[0]
    elif len(peaks) > 0:  #Rise then fall
        order = True
    elif len(troughs) > 0:  # Fall then rise
        order = False
    else:  #No peaks or troughs so just return a single rising
        return ([data], [])
    splits = [0, len(data)]
    splits.extend(peaks)
    splits.extend(troughs)
    splits.sort()
    if not isinstance(folder, _SF_):  # Create a new DataFolder object
        output = _SF_()
    else:
        output = folder
    output.add_group("rising")
    output.add_group("falling")
    for i in range(1, len(splits), 2):
        working1 = data.clone
        working2 = data.clone
        working1.data = data.data[splits[i - 1]:splits[i],:]
        working2.data = data.data[splits[i]:splits[1 + 1],:]
        if not order:
            (working1, working2) = (working2, working1)
        output.groups["rising"].files.append(working1)
        output.groups["falling"].files.append(working2)
    return output


def format_error(value, error, latex=False, mode="float", units="", prefix=""):
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
    if error == 0.0:  # special case for zero uncertainty
        return repr(value)
    #Sort out special fomatting for different modes
    if mode == "float":  # Standard
        suffix_val = ""
    elif mode == "eng":  #Use SI prefixes
        v_mag = floor(log10(abs(value)) / 3.0) * 3.0
        prefixes = {
            3: "k",
            6: "M",
            9: "G",
            12: "T",
            15: "P",
            18: "E",
            21: "Z",
            24: "Y", -3: "m", -6: "\\mu", -9: "n", -12: "p", -15: "f", -18: "a", -21: "z", -24: "y"
        }
        if v_mag in prefixes:
            if latex:
                suffix_val = r"\mathrm{{{{{}}}}}".format(prefixes[v_mag])
            else:
                suffix_val = prefixes[v_mag]
            value /= 10 ** v_mag
            error /= 10 ** v_mag
        else:  # Implies 10^-3<x<10^3
            suffix_val = ""
    elif mode == "sci":  # Scientific mode - raise to common power of 10
        v_mag = floor(log10(abs(value)))
        if latex:
            suffix_val = r"\times 10^{{{{{}}}}}".format(int(v_mag))
        else:
            suffix_val = "E{} ".format(int(v_mag))
        value /= 10 ** v_mag
        error /= 10 ** v_mag
    else:  # Bad mode
        raise RuntimeError("Unrecognised mode: {} in format_error".format(mode))

# Now do the rounding of the value based on error to 1 s.f.
    e2 = error
    u_mag = floor(log10(abs(error)))  #work out the scale of the error
    error = round(error / 10 ** u_mag) * 10 ** u_mag  # round the error, but this could round to 0.x0
    u_mag = floor(log10(error))  # so go round the loop again
    error = round(e2 / 10 ** u_mag) * 10 ** u_mag  # and get a new error magnitude
    value = round(value / 10 ** u_mag) * 10 ** u_mag
    u_mag = min(0, u_mag)  # Force integer results to have no dp

    #Protect {} in units string
    units = units.replace("{", "{{").replace("}", "}}")
    prefix = prefix.replace("{", "{{").replace("}", "}}")
    if latex:  # Switch to latex math mode symbols
        val_fmt_str = r"${}{{:.{}f}}\pm ".format(prefix, int(abs(u_mag)))
        if units != "":
            suffix_fmt = r"\mathrm{{{{{}}}}}".format(units)
        else:
            suffix_fmt = ""
        suffix_fmt += "$"
    else:  # Plain text
        val_fmt_str = r"{}{{:.{}f}}+/-".format(prefix, int(abs(u_mag)))
        suffix_fmt = units
    if u_mag < 0:  # the error is less than 1, so con strain decimal places
        err_fmt_str = r"{:." + str(int(abs(u_mag))) + "f}"
    else:  # We'll be converting it to an integer anyway
        err_fmt_str = r"{}"
    fmt_str = val_fmt_str + err_fmt_str + suffix_val + suffix_fmt
    if error >= 1.0:
        error = int(error)
        value = int(value)
    return fmt_str.format(value, error)

Hickeyify = format_error


def ordinal(value):
    """Format an integer into an ordinal string.

    Args:
        value (int): Number to be written as an ordinal string

    Return:
        Ordinal String such as '1st','2nd' etc."""
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
            also recentres the loop about zero moment.
        correct_H (bool): Finds the co-ercive fields and sets them to be equal and opposite. If the loop is sysmmetric
            this will remove any offset in filed due to trapped flux
        saturated_fraction (float): The fraction of the horizontal (field) range where the moment can be assumed to be
            fully saturated.
        xcol (column index): Column with the x data in it
        ycol (column_index): Column with the y data in it
        setas (string or iterable): Column assignments.

    Returns:
        The original loop with the x and y columns replaced with corrected data and extra metadata added to give the
        background suceptibility, offset in moment, co-ercive fields and saturation magnetisation.
    """

    if isinstance(data, _DF_):
        cls = data.__class__
    else:
        cls = Data
    data = cls(data)

    if "setas" in kargs: # Allow us to override the setas variable
        d.setas=kargs.pop("setas")

    #Get xcol and ycols from kargs if specified
    xc = kargs.pop("xcol",data.find_col(data.setas["x"]))
    yc = kargs.pop("ycol",data.find_col(data.setas["y"]))
    setas=data.setas.clone
    setas[xc]="x"
    setas[yc]="y"
    data.setas=setas

    #Split into two sets of data:

    #Get other keyword arguments
    correct_background=kargs.pop("correct_background",True)
    correct_H=kargs.pop("correct_H",True)
    saturation_fraction=kargs.pop("saturation_fraction",0.2)

    while True:
        up,down=_up_down(data)
        mx = max(data.x) * (1 - saturation_fraction)
        mix = min(data.x) * (1 - saturation_fraction)


        #Find upper branch saturated moment slope and offset
        p1, pcov = up.curve_fit(linear, absolute_sigma=False, bounds=lambda x, r: x < mix)
        perr1 = diag(pcov)

        #Find lower branch saturated moment and offset
        p2, pcov = down.curve_fit(linear, absolute_sigma=False, bounds=lambda x, r: x > mx)
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
        mr=d.threshold(0.0,col=xc,xcol=yc,all_vals=True,rising=True,falling=True)
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
    return cls(data)
