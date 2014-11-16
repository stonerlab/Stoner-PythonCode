 # A very quick and dirty Kiessig Fringe Analysis code
 # Gavin Burnell 28/12/2010
 # $log$
 #
 # TODO: Imp,ement an error bar on the uncertainity by understanding the significance of the covariance terms

import Stoner.Analysis as SA
import Stoner.Plot as SP
from Stoner.Fit import linear
import Stoner.FileFormats
from Stoner.Util import format_error
import numpy as np
import matplotlib.pyplot as pyplot
from copy import copy

class Keissig(SA.AnalyseFile,SP.PlotFile):
    pass

filename=False
sensitivity=20

critical_edge=0.8

d=Keissig(filename) #Load the low angle scan

#Now get the section of the data file that has the peak positions
# This is really doing the hard work
# We differentiate the data using a Savitsky-Golay filter with a 5 point window fitting quartics.
# This has proved most succesful for me looking at some MdV data.
# We then threshold for zero crossing of the derivative
# And check the second derivative to see whether we like the peak as signficant. This is the significance parameter
# and seems to be largely empirical
# Finally we interpolate back to the complete data set to make sure we get the angle as well as the counts.
d.curve_fit(lambda x,a,b:a*np.exp(-x/b),"Angle","Counts",result=True,replace=False,header="Envelope")
d.subtract("Counts","Envelope",replace=False,header="peaks")
d.del_column("Envelope")
t=Keissig(d.interpolate(d.peaks('peaks',significance=sensitivity,width=4,poly=4)))
t.column_headers=copy(d.column_headers)
t.fig=d.plot_xy("Angle","peaks")
t.plot_xy("Angle","peaks","ro")
d.del_column('peaks')
t.del_column('peaks')
d.setas="xy"
d.column_headers[d.find_col('Angle')]=r"Reflection Angle $\theta$"
t.del_rows(0, lambda x,y: x<critical_edge)
t.setas="xy"
t.figure()
t.plot(fmt='ro',  plotter=pyplot.semilogy)
main_fig=d.plot(figure=t.fig, plotter=pyplot.semilogy)
d.show()
#Now convert the angle to sin^2
t.apply(lambda x: np.sin(np.radians(x[0]/2.0))**2, 0,header=r"$sin^2\theta$")
# Now create the m^2 order
m=np.arange(len(t))+1
m=m**2
#And add it to t
t.add_column(m, column_header='$m^2$')
#Now we can it a straight line
t.setas="x.y"
p, pcov=t.curve_fit(linear,result=True,replace=False,header="Fit")
g=p[1]
gerr=np.sqrt(pcov[1,1])/g
g=np.sqrt(1.0/g)
gerr/=2.0
l=float(d['Lambda'])
th=l/(2*g)
therr=th*(gerr)

t.inset(loc="top right")
t.plot_xy(r"Fit",r"$sin^2\theta$", 'b-')
t.plot_xy(r"$m^2$",r"$sin^2\theta$", 'ro')
t.xlabel="Fringe $m^2$"
t.ylabel=r"$sin^2\theta$"
t.title=None
t.show()
pyplot.sca(t.axes[0])
# Get the wavelength from the metadata
# Calculate thickness and report
pyplot.text (0.05,0.05, "Thickness is: {} $\AA$".format(format_error(th,therr,latex=True)), transform=main_fig.axes[0].transAxes)
