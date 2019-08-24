"""Demo of Fitting a directory of files
"""
from os.path import join

from Stoner import __home__, DataFolder, Data
from Stoner.plot.formats import TexEngFormatter
from Stoner.Fit import Quadratic

# Set up the directory with our data
datafiles = join(__home__, "..", "sample-data", "NLIV")

# DataFolder of our data files
fldr = DataFolder(datafiles, pattern="*.txt", setas="yx.")

# Another Data object to keep the results in
result = Data()

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
result.plot(fmt="k.")

# An alternative way to run the Analysis - this time with
# an orthogonal didstance regression algorithm

# Run the fitt for each file in the fldr. Set the outpout to "data" to
# Have the amended results replace the existing data files
fldr.each.odr(Quadratic, output="data")
# Now take a slice through the metadata to get the files we want.
result_2 = fldr.metadata.slice(["y", "Quadratic:b", "Quadratic:b err"], output="Data")

# Set the columns assignments and plot
result_2.setas = "xye"
result_2.plot(fmt="r.", figure=result.fig)
