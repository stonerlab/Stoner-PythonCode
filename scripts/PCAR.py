#
# $Id: PCAR.py,v 1.2 2011/01/10 23:11:21 cvs Exp $
#$Log: PCAR.py,v $
#Revision 1.2  2011/01/10 23:11:21  cvs
#Switch to using GLC's version of the mpit module
#Made PlotFile.plot_xy take keyword arguments and return the figure
#Fixed a missing import math in AnalyseFile
#Major rewrite of CSA's PCAR fitting code to use mpfit and all the glory of the Stoner module - GB
#
#Revision 1.1  2011/01/10 19:05:44  cvs
#Start working on PCAR fitting code
#
# Script to fit PCAR data GB Jan 2011

# Initial para,eter
omega={"value":0.55, "fixed":False, "limited":[True, False], "limits":[0.0, 0.0], "parname":"Omega", "step":0, "mpside":0, "mpmaxstep":0, "tied":"", "mpprint":True}
delta={"value":1.5, "fixed":False, "limited":[True, True], "limits":[0.5, 2.0], "parname":"Delta", "step":0, "mpside":0, "mpmaxstep":0, "tied":"", "mpprint":True}
P={"value":0.2, "fixed":False, "limited":[True, True], "limits":[0.0, 1.0], "parname":"Polarisation", "step":0, "mpside":0, "mpmaxstep":0, "tied":"", "mpprint":True}
Z={"value":0.4, "fixed":False, "limited":[True, False], "limits":[0.3, 0.0], "parname":"Barrier", "step":0, "mpside":0, "mpmaxstep":0, "tied":"", "mpprint":True}

#Normal state conductance
Gn=1.0

#Some stuff about the datafile
gcol='G'
vcol='V'
# Import packages
import numpy
import scipy
import Stoner
from Stoner.mpfit import mpfit
import easygui
import sys
import math



#gui to get filename and path
filename=easygui.fileopenbox(title = "Choose your file")
#import data
d=Stoner.AnalyseFile(filename)

gcol=d.find_col(gcol)
vcol=d.find_col(vcol)

def myfunct(p, fjac=None, xdat=None, ydat=None, err=None):
    # Parameter values are passed in "p"
    # If FJAC!=None then partial derivatives must be comptuer.
    # FJAC contains an array of len(p), where each entry
    # is 1 if that parameter is free and 0 if it is fixed. 
    model = strijkers(x, p)
    # stop the calculation.
    status = 0
    return [status, (y-model)/err]
    
def iterfunct(myfunct, p, iter, fnorm, functkw=None,parinfo=None, quiet=0, dof=None):
    sys.stdout.write('.')
    sys.stdout.flush()

def strijkers(V, params):
    """
    strijkers(V, params):
    V = bias voltages, params=list of parameter values, imega, delta,P and Z
    
    Strijkers modified BTK model
        BTK PRB 25 4515 1982, Strijkers PRB 63, 104510 2000

    Only using 1 delta, not modified for proximity
"""
#   Parameters
    omega=params[0]     #Broadening
    delta=params[1]    #SC energy Gap
    P=params[2]         #Interface parameter 
    Z=params[3]         #Current spin polarization through contact
    
    E = scipy.arange(-50, 50, 0.05) # Energy range in meV
    
    #Reflection prob arrays
    Au=scipy.zeros(len(E))
    Bu=scipy.zeros(len(E))
    Bp=scipy.zeros(len(E))
    
    #Conductance calculation
    """    
    % For ease of calculation, epsilon = E/(sqrt(E^2 - delta^2))
    %Calculates reflection probabilities when E < or > delta
    %A denotes Andreev Reflection probability
    %B denotes normal reflection probability
    %subscript p for polarised, u for unpolarised
    %Ap is always zero as the polarised current has 0 prob for an Andreev
    %event
    """
    
    for ss in range(len(E)):
        if abs(E[ss])<=delta:
            Au[ss]=(delta**2)/((E[ss]**2)+(((delta**2)-(E[ss]**2))*(1+2*(Z**2))**2));
            Bu[ss] = 1-Au[ss];
            Bp[ss] = 1;
        else:
            Au[ss] = (((abs(E[ss])/(scipy.sqrt((E[ss]**2)-(delta**2))))**2)-1)/(((abs(E[ss])/(scipy.sqrt((E[ss]**2)-(delta**2)))) + (1+2*(Z**2)))**2);
            Bu[ss] = (4*(Z**2)*(1+(Z**2)))/(((abs(E[ss])/(scipy.sqrt((E[ss]**2)-(delta**2)))) + (1+2*(Z**2)))**2);
            Bp[ss] = Bu[ss]/(1-Au[ss]);
        
    #  Calculates reflection 'probs' for pol and unpol currents
    Guprob = 1+Au-Bu;
    Gpprob = 1-Bp;

    #Calculates pol and unpol conductance and normalises
    Gu = (1-P)*(1+(Z**2))*Guprob;
    Gp = 1*(P)*(1+(Z**2))*Gpprob;
    
    G = Gu + Gp;
    
    
    #Sets up gaus
    gaus=scipy.zeros(len(V));
    cond=scipy.zeros(len(V));
    
    #computes gaussian and integrates over all E(more or less)
    for tt in range(len(V)):
    #Calculates fermi level smearing
        gaus=(1/(2*omega*scipy.sqrt(scipy.pi)))*scipy.exp(-(((E-V[tt])/(2*omega))**2))
        cond[tt]=numpy.trapz(gaus*G,E);
    return cond

# Normalise the data
d.apply(lambda x:x[gcol]/Gn, gcol)

#Centre the data - look for peaks and troughs within 5 of the initial delta value
# take the average of these and then subtract it.
peaks=d.peaks(gcol,len(d)/20,0,xcol=vcol,poly=4,peaks=True,troughs=True)

peaks=filter(lambda x: abs(x)<4*delta['value'], peaks)
offset=numpy.mean(numpy.array(peaks))
print "Mean offset ="+str(offset)
d.apply(lambda x:x[vcol]-offset, vcol)

#Pull out the x and y data separately
x=d.column(vcol)
y=d.column(gcol)
ey=numpy.ones(y.shape,dtype='float64') # equal weights
fa = {'xdat':x, 'ydat':y, 'err':ey}
debug=True
parinfo=[omega, delta, P, Z]
pderiv=None
m = mpfit(myfunct, parinfo=parinfo,functkw=fa,  autoderivative=True,  iterfunct=iterfunct)
if (m.status <= 0): 
    print 'error message = ', m.errmsg

print m.params
d.add_column(strijkers(x, m.params), 'Fit')
p=Stoner.PlotFile(d)
p.plot_xy(vcol,gcol, 'ro')
p.plot_xy(vcol, 'Fit')






