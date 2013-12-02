 # A very quick and dirty Kiessig Fringe Analysis code

 # Gavin Burnell 28/12/2010

 # $log$

 #

 # TODO: Imp,ement an error bar on the uncertainity by understanding the significance of the covariance terms



import Stoner

from Stoner.Util import format_error

import numpy as np

import matplotlib.pyplot as pyplot



class Keissig(Stoner.AnalyseFile,Stoner.PlotFile):

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

d.add_column(lambda x:np.log(x[1]), 'log(Counts)')

t=Keissig(d.interpolate(d.peaks('Counts',5,sensitivity,poly=4)))

t.column_headers=[r"$\theta$","counts"]

d.column_headers[d.find_col('Angle')]=r"Reflection Angle $\theta$"

t.del_rows(0, lambda x,y: x<critical_edge)

t.plot_xy(0, 1, 'ro',  plotter=pyplot.semilogy, figure=None)

main_fig=d.plot_xy(0, 1, figure=t.fig, plotter=pyplot.semilogy)

d.show()

#Now convert the angle to sin^2

t.apply(lambda x: np.sin((x[0]/2)*(np.pi/180))**2, 0,header=r"$sin^2\theta$")

# Now create the m^2 order

m=np.arange(len(t))+1

m=m**2

#And add it to t

t.add_column(m, column_header='$m^2$')

#Now we can it a straight line

linear=lambda x, m, c: m*x+c

p, pcov=t.curve_fit(linear, r'$m^2$', r'$sin^2\theta$')

pcov=np.sqrt(pcov)

fit=linear(t.column(r"$m^2$"),*p)

t.add_column(fit,column_header="Fit")

pyplot.axes([0.5, 0.6, 0.35, 0.25])

fig=t.plot_xy(r"$m^2$","Fit", 'b-')

t.plot_xy(r"$m^2$",r"$sin^2\theta$", 'ro',figure=fig,title="Fitted fringes")

t.show()

#Get the square root of the gradient of the line

g=np.sqrt(p[0])

gerr=.5*pcov[0,0]/g

# Get the wavelength from the metadata

l=float(d['Lambda'])

# Calculate thickness and report

th=l/(2*g)

therr=th*(gerr)



pyplot.text (0.05,0.05, "Thickness is: {} $\AA$".format(format_error(th,therr,latex=True)), transform=main_fig.axes[0].transAxes)

