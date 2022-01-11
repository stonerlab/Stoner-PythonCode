"""Demo of Fitting a directory of files."""
# pylint: disable=invalid-name
from os.path import join

from matplotlib.pyplot import figure

from Stoner import Data, DataFolder, __home__
from Stoner.analysis.fitting.models.generic import Quadratic
from Stoner.plot.formats import TexEngFormatter

if __name__ in ["__main__", "folder_fit"]:
    # Set up the directory with our data
    datafiles = join(__home__, "..", "sample-data", "NLIV")

    # DataFolder of our data files
    fldr = DataFolder(datafiles, pattern="*.txt").sort("y")

    fldr.setas = "yx"

    # Another Data object to keep the results in
    result = Data()
    result.title = "Non-local Resistances"

    # Loop over the files in the DataFolder
    for d in fldr:
        # Fit, returning results as a dictionary
        row = d.lmfit(Quadratic, output="dict")
        # Add Extra information from metadata
        row["Magnetic Field $\\mu_0 H (T)$"] = d["y"]
        # Add the new row of data to the result.
        result += row

    # Set columns and set the plot axis formatting to use SI precfixes
    result.setas(x="Field", y="b", e="d_b")
    result.template.xformatter = TexEngFormatter
    result.template.yformatter = TexEngFormatter

    # Plot
    result.plot(fmt="k.", capsize=2)

    # An alternative way to run the Analysis - this time with
    # an orthogonal didstance regression algorithm

    # Run the fitt for each file in the fldr. Set the outpout to "data" to
    # Have the amended results replace the existing data files
    fldr.each.odr(
        Quadratic, output="data", result=True, header="fit", _mode="ThreadPool"
    )
    fig = figure()
    fldr.each.setas(Current="x", Voltage="y")
    fldr.each.setas[3] = "y"

    fldr.each.plot(
        fmt=["+", "-"], label="Field = {y}mT", figure=fig, _mode="serial"
    )
    fldr[0].legend(ncol=2, fontsize="xx-small")
    fldr[0].title = "Non-local IV Curves"

    # Now take a slice through the metadata to get the files we want.
    result_2 = fldr.metadata.slice(["y", Quadratic], output="Data")

    # Set the columns assignments and plot
    result_2.setas = "x..ye"
    result_2.plot(fmt="r.", figure=result.fig, capsize=2)
    result_2.title = "Non-local Resistances"
