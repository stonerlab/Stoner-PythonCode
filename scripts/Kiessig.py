 # A very quick and dirty Kiessig Fringe Analysis code
 # Gavin Burnell 28/12/2010
 # $log$
 #
 # TODO: Imp,ement an error bar on the uncertainity by understanding the significance of the covariance terms
 
import Stoner
import numpy as np

filename='New-XRay-Data.dql'
sensitivity=4

d=Stoner.AnalyseFile(filename,'NewXRD') #Load the low angle scan

#Now get the section of the data file that has the peak positions
# This is really doing the hard work
# We differentiate the data using a Savitsky-Golay filter with a 5 point window fitting quartics.
# This has proved most succesful for me looking at some MdV data.
# We then threshold for zero crossing of the derivative
# And check the second derivative to see whether we like the peak as signficant. This is the significance parameter
# and seems to be largely empirical
# Finally we interpolate back to the complete data set to make sure we get the angle as well as the counts.
t=Stoner.AnalyseFile(d.interpolate(d.peaks('Counts',5,sensitivity,poly=4)))
#Now convert the angle to sin^2
t.apply(lambda x: np.sin(x[0]*(np.pi/180))**2, 0)
# Now create the m^2 order
m=np.arange(len(t))+1
m=m**2
#And add it to t
t.add_column(m, 'm^2')
#Now we can it a straight line
def linear(x, m, c):
    return m*x+c
p, pcov=t.curve_fit(linear, 2, 0)
#Get the square root of the gradient of the line
g=np.sqrt(p[0])
# Get the wavelength from the metadata
l=d['Lambda']
# Calculate thickness and report
th=l/(2*g)
print "Thickness is:"+str(th)+"Angstroms - please round correctly before telling Bryan"
