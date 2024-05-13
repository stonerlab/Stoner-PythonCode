"""Example customising a plot using default style."""

# pylint: disable=invalid-name
import os.path as path
from cycler import cycler

from Stoner import Data, __home__
from Stoner.plot.formats import DefaultPlotStyle

filename = path.realpath(
    path.join(__home__, "..", "doc", "samples", "sample.txt")
)

d = Data(filename, setas="xyy", template=DefaultPlotStyle)
d.normalise(2, scale=(0.0, 10.0))  # Just to keep the plot sane!

# Change the color and line style cycles
d.template["axes.prop_cycle"] = cycler(color=["Blue", "Purple"]) + cycler(
    linestyle=["-", "--"]
)

# Can also access as an attribute
d.template.template_lines__linewidth = 2.0

# Set the default figure size
d.template["figure.figsize"] = (6, 8)
d.template["figure.autolayout"] = True
# Make figure (before using subplot method) and select first subplot
d.figure(no_axes=True)
d.subplot(212)
# Pkot with our customised defaults
d.plot()
d.grid(True, color="green", linestyle="-.")
d.title = "Customised Plot settings"
# Reset the template to defaults and switch to next subplot
d.template.clear()
d.subplot(211)
# Plot with defaults
d.plot()
d.title = "Style Default settings"
# Fixup layout
d.figwidth = 7  # Magic pass through attribute access
