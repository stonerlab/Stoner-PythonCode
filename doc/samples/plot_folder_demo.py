# -*- coding: utf-8 -*-
"""Demonstrate PlotFolder Class."""

# pylint: disable=invalid-name
from os.path import join

from numpy import where
from scipy.constants import mu_0
from scipy.stats import gmean

from Stoner import Data, __home__
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.analysis.fitting.models.magnetism import FMR_Power, Inverse_Kittel
from Stoner.folders import PlotFolder
def format_axes(ax):
    """Give each axis its own engineering formatter."""
    ax.xaxis.set_major_formatter(Data().plot.TexEngFormatter(unit="T"))
    ax.yaxis.set_major_formatter(Data().plot.TexEngFormatter())


def field_sign(r):
    """Return string key for sign of field.."""
    pos = r[r.setas.find_col("Field")] >= 0
    return where(pos, "pos", "neg")


def extra(_, __, d):
    """Customise each individual plot."""
    format_axes(d.plot.ax)
    d.plot.ax.axvline(x=d["cut"], ls="--")
    d.plot.ax.set_title(rf"$\nu={d.mean('Frequency') / 1e9:.1f}\,$GHz")
    d.plot.ax.set_xlabel(r"Field $\mu_0H\,$")
    d.plot.ax.set_ylabel("Abs. (arb)")
    d.plot.ax.legend(loc=3)
    d.annotate_fit(FMR_Power, fontdict={"size": 8}, x=0.05, y=0.25)


def do_fit(f):
    """Fit just one set of data."""
    f["cut"] = f.threshold(1.75e5, rising=False, falling=True)
    f["Frequency"] = (f // "Frequency").mean()
    f.lmfit(
        FMR_Power, result=True, header="Fit", bounds=lambda x, r: x < f["cut"]
    )
    return f


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
    d.plot.labels = None

    # Define x and y columns and normalise to a big number
    d.setas(x="Field", y="FMR")  # pylint: disable=not-callable
    d.normalise(base=(-1e6, 1e6))
    fldr = d.split(field_sign, "Frequency")

    # Split the data file into separate files by frequencies and sign of field
    fldr = PlotFolder(fldr)  # Convert to a PlotFolder
    for f in fldr["neg"]:  # Invert the negative field side
        f.x = -f.x[::-1]
        f.y = -f.y[::-1]

    resfldr = (
        PlotFolder()
    )  # Somewhere to keep the results from +ve and -ve fields

    for s in fldr.groups:  # Fit each FMR spectra
        subfldr = fldr[s]
        subfldr.metadata["Field Sign"] = s

        print(f"s={s}")
        (do_fit @ subfldr.each)()
        result = subfldr.metadata.slice("Frequency", FMR_Power, output="Data")
        result.metadata.update(
            {x: subfldr[0][x] for x in subfldr.metadata.common_keys}
        )

        # Now plot all the fits
        subfldr.setas[12] = "y"
        subfldr.plots_per_page = 9  # Plot results
        subfldr.plot(
            figsize=(8, 8), fmt=["k.", "r-"], markersize=1, plot_extra=extra
        )

        # Work with the overall results
        try:
            result.setas(  # pylint: disable=not-callable
                y="H_res$", e="H_res err", x="Frequency"
            )  # pylint: disable=not-callable
            result.y = result.y / mu_0  # Convert to A/m
            result.e = result.e / mu_0

            resfldr += result  # Stash the results
        except KeyError:
            continue

    # Merge the two field signs into a single file, taking care of the error columns too
    result = resfldr[0].clone
    for c in [0, 2, 4, 6, 8]:
        result[:, c] = (resfldr[1][:, c] + resfldr[0][:, c]) / 2.0
    for c in [1, 3, 5, 7]:
        result[:, c] = gmean((resfldr[0][:, c], resfldr[1][:, c]), axis=0)

    # Doing the Kittel fit with an orthogonal distance regression as we have x errors not y errors
    p0 = [2, 200e3, 10e3]  # Some sensible guesses
    result.lmfit(
        Inverse_Kittel, p0=p0, result=True, header="Kittel Fit", output="report"
    )
    result.setas[-1] = "y"

    result.plot.labels = None
    figure = result.plot.new_figure(figsize=(6, 8))
    axes = figure.subplots(2, 1)
    for target in axes:
        target.xaxis.set_major_formatter(result.plot.TexEngFormatter())
        target.yaxis.set_major_formatter(result.plot.TexEngFormatter())
    result.plot(ax=axes[0], fmt=["r.", "b-"])
    result.annotate_fit(Inverse_Kittel, ax=axes[0], x=0.55, y=0.3, fontdict={"size": 8})
    axes[0].set_ylabel("$H_{res} \\mathrm{(Am^{-1})}$")
    axes[0].set_title("Inverse Kittel Fit")

    # Get alpha
    # Draw the second fit into its explicit axes.
    result.setas(y="Delta_H", e="Delta_H err", x="Freq")
    result.y = result.y / mu_0
    result.e = result.e / mu_0
    result.lmfit(Linear, result=True, header="Width", output="report")
    result.setas[-1] = "y"
    result.plot(ax=axes[1], fmt=["r.", "b-"])
    result.annotate_fit(Linear, ax=axes[1], x=0.1, y=0.65, fontdict={"size": 8})
