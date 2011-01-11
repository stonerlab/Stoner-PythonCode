#
# $Id: PCAR.py,v 1.6 2011/01/11 18:55:57 cvs Exp $
#$Log: PCAR.py,v $
#Revision 1.6  2011/01/11 18:55:57  cvs
#Move mpfit into a method of AnalyseFile and make the API like AnalyseFile.curvefit
#
#Revision 1.5  2011/01/11 16:26:52  cvs
#Convert code to use a separate ini file to setup problem
#
#Revision 1.4  2011/01/10 23:49:13  cvs
#Fix filename from easygui to be a string (only a problem on Windows !?) - GB
#
#Revision 1.3  2011/01/10 23:28:15  cvs
#Improved comments in code added some pretty printing - GB
#
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

# Import packages
import numpy
import scipy
import Stoner
from scipy.stats import chisquare
import wx
import sys
import math
import os
import ConfigParser

# Read the co nfig file for the model
config=ConfigParser.SafeConfigParser()
config.read("pcar.ini")

show_plot=config.getboolean('Fitting', 'show_plot')
user_iterfunct=config.getboolean('Fitting', 'print_each_step')

pars=dict()
parnames=['omega', 'delta', 'P', 'Z']
for section in parnames:
    pars[section]=dict()
    pars[section]['value']=config.getfloat(section, 'value')
    pars[section]['fixed']=config.getboolean(section, 'fixed')
    pars[section]['limited']=[config.getboolean(section, 'lower_limited'), config.getboolean(section, 'upper_limited')]
    pars[section]['limits']=[config.getfloat(section, 'lower_limit'), config.getfloat(section, 'upper_limit')]
    pars[section]['parname']=config.get(section, 'name')
    pars[section]['step']=config.getfloat(section, 'step')
    pars[section]['mpside']=config.getint(section, 'side')
    pars[section]['mpmaxstep']=config.getfloat(section, 'maxstep')
    pars[section]['tied']=config.get(section, 'tied')
    pars[section]['mpprint']=config.getboolean(section, 'print')

Gn=config.getfloat('data', 'Normal_conductance')
gcol=config.get('data', 'y-column')
vcol=config.get('data', 'x-column')
omega=pars['omega']
delta=pars['delta']
P=pars['P']
Z=pars['Z']



dlg=wx.FileDialog(None, "Select Datafile", "", "", "*.*", wx.OPEN)
if dlg.ShowModal()==wx.ID_OK:
    filename=os.path.join(dlg.Directory, dlg.Filename)
else:
    raise RuntimeError("Must specify a filename !")
        
#import data
d=Stoner.AnalyseFile(str(filename))

# Convert string column headers to numeric column indices
gcol=d.find_col(gcol)
vcol=d.find_col(vcol)

################################################
######### Here is out model functions  #########
###############################################
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

# Main part of code

# Normalise the data
d.apply(lambda x:x[gcol]/Gn, gcol)

#Centre the data - look for peaks and troughs within 5 of the initial delta value
# take the average of these and then subtract it.
peaks=d.peaks(gcol,len(d)/20,0,xcol=vcol,poly=4,peaks=True,troughs=True)
peaks=filter(lambda x: abs(x)<4*delta['value'], peaks)
offset=numpy.mean(numpy.array(peaks))
print "Mean offset ="+str(offset)
d.apply(lambda x:x[vcol]-offset, vcol)

# Initialise the parameter information with the dictionaries defined at top of file
parinfo=[omega, delta, P, Z]

if user_iterfunct==False:
    # Here is the engine that does the work
    m = d.mpfit(strijkers,vcol, gcol, parinfo, iterfunct=d.mpfit_iterfunct)
    print "Finished !"
else:
    m = d.mpfit(strijkers,vcol, gcol, parinfo)

if (m.status <= 0): # error message ?
    raise RuntimeError(m.errmsg)


if show_plot:
    # And show the fit and the data in a nice plot
    d.add_column(strijkers(d.column(vcol), m.params), 'Fit')
    p=Stoner.PlotFile(d)
    p.plot_xy(vcol,gcol, 'ro')
    p.plot_xy(vcol, 'Fit')

# Ok now we can print the answer
for i in range(len(parinfo)):
    print parinfo[i]['parname']+"="+str(m.params[i])

chi2=chisquare(d.column(gcol), d.column('Fit'))
print "Chi^2:"+str(chi2)
    







