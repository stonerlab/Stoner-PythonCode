# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:37:39 2017

@author: phygbu
"""
from numpy import where, append, atleast_2d
from scipy.constants import mu_0
from scipy.stats import gmean
from os.path import join

from Stoner import Data, __home__
from Stoner.Fit import FMR_Power, Inverse_Kittel, Linear
from Stoner.plot.formats import DefaultPlotStyle, TexEngFormatter
from Stoner.Folders import PlotFolder

# set_option("multiprocessing",os.name!="nt")

# Customise a plot template
template = DefaultPlotStyle()
template.xformatter = TexEngFormatter()
template.xformatter.unit = "T"
template.yformatter = TexEngFormatter


def field_sign(r):
    """Custom function for split"""
    pos = r["Field"] >= 0
    return where(pos, 1, -1)


def extra(i, j, d):
    """Function for customising each individual plot"""
    d.axvline(x=d["cut"], ls="--")
    d.title = r"$\nu={:.1f}\,$GHz".format(d.mean("Frequency") / 1e9)
    d.xlabel = r"Field $\mu_0H\,$"
    d.ylabel = "Abs. (arb)"
    d.plt_legend(loc=3)
    d.annotate_fit(FMR_Power, fontdict={"size": 8}, x=0, y=1000)


def do_fit(f):
    """Function to fit just one set of data."""
    f.template = template
    f["cut"] = f.threshold(1.75e5, rising=False, falling=True)
    res = f.lmfit(
        FMR_Power,
        result=True,
        header="Fit",
        bounds=lambda x, r: x < f["cut"],
        output="row",
    )
    ch = res.column_headers
    res = append(res, [f.mean("Frequency"), f["Field Sign"]])
    res = atleast_2d(res)
    res.column_headers = ch + ["Frequency (Hz)", "Field Sign"]
    f.setas[-1] = "y"
    return res, res.column_headers


###############################################################################################
#################### Important - if using multiprocessing on Windows, this  block must be ######
#################### Inside a if __name__=="__main__": test. ###################################
###############################################################################################

if __name__ == "__main__":
    # Load data
    d = Data(join(__home__, "..", "sample-data", "FMR-data.txt"))
    # Rename columns and reset plot labels
    d.rename("multi[1]:y", "Field").rename("multi[0]:y", "Frequency").rename(
        "Absorption::X", "FMR"
    )
    d.labels = None

    # Deine x and y columns and normalise to a big number
    d.setas(x="Field", y="FMR")  # pylint: disable=not-callable
    d.normalise(base=(-1e6, 1e6))
    fldr = d.split(field_sign, "Frequency")

    # Split the data file into separate files by frequencies and sign of field
    fldr = PlotFolder(fldr)  # Convert to a PlotFolder
    fldr.template = template  # Set my custom plot template
    for f in fldr[-1]:  # Invert the negative field side
        f.x = -f.x[::-1]
        f.y = -f.y[::-1]

    resfldr = PlotFolder()  # Somewhere to keep the results from +ve and -ve fields

    for s in fldr.groups:  # Fit each FMR spectra
        subfldr = fldr[s]
        subfldr.metadata["Field Sign"] = s

        print("s={}".format(s))
        result = []
        for ix, res in enumerate(subfldr.each.iter(do_fit)):
            result.append(res)

        data, headers = zip(*result)
        new_data = data[0]
        for r in data[1:]:
            new_data = append(new_data, r, axis=0)
        result = Data(new_data)
        result.column_headers = headers[0]

        # Now plot all the fits

        subfldr.plots_per_page = 6  # Plot results
        subfldr.plot(figsize=(8, 8), extra=extra)

        # Work with the overall results
        result.setas(  # pylint: disable=not-callable
            y="H_res", e="H_res.stderr", x="Freq"
        )  # pylint: disable=not-callable
        result.y = result.y / mu_0  # Convert to A/m
        result.e = result.e / mu_0

        resfldr += result  # Stash the results

    # Merge the two field signs into a single file, taking care of the error columns too
    result = resfldr[0].clone
    for c in [0, 2, 4, 6, 8, 9, 10]:
        result.data[:, c] = (resfldr[1][:, c] + resfldr[0][:, c]) / 2.0
    for c in [1, 3, 5, 7]:
        result.data[:, c] = gmean((resfldr[0][:, c], resfldr[1][:, c]), axis=0)

    # Doing the Kittel fit with an orthogonal distance regression as we have x errors not y errors
    p0 = [2, 200e3, 10e3]  # Some sensible guesses
    result.lmfit(
        Inverse_Kittel, p0=p0, result=True, header="Kittel Fit", output="report"
    )
    result.setas[-1] = "y"

    result.template.yformatter = TexEngFormatter
    result.template.xformatter = TexEngFormatter
    result.labels = None
    result.figure(figsize=(6, 8))
    result.subplot(211)
    result.plot(fmt=["r.", "b-"])
    result.annotate_fit(Inverse_Kittel, x=7e9, y=1e5, fontdict={"size": 8})
    result.ylabel = "$H_{res} \\mathrm{(Am^{-1})}$"
    result.title = "Inverse Kittel Fit"

    # Get alpha
    result.subplot(212)
    result.setas(y="Delta_H", e="Delta_H.stderr", x="Freq")
    result.y /= mu_0
    result.e /= mu_0
    result.lmfit(Linear, result=True, header="Width", output="report")
    result.setas[-1] = "y"
    result.plot(fmt=["r.", "b-"])
    result.annotate_fit(Linear, x=5.5e9, y=2.8e3, fontdict={"size": 8})
