#
# $Id: PCAR.py,v 1.10 2011/10/24 12:17:54 cvs Exp $
#$Log: PCAR.py,v $
#Revision 1.10  2011/10/24 12:17:54  cvs
#Update PCAR lab script to save data and fix a bug with save as mode in Stoner.Core
#
#Revision 1.9  2011/06/16 09:40:57  cvs
#Ironed out a bug or two - csa
#
#Revision 1.8  2011/01/13 22:30:56  cvs
#Enable chi^2 analysi where the parameters are varied and choi^2 calculated.
#Extra comments in the ini file
#Give DataFile some file dialog boxes
#
#Revision 1.7  2011/01/11 21:52:26  cvs
#Change the script to make some of the more dangerous data manipulations into options to be turned on in the ini file
#Commented the ini file. - GB
#
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
import time
import ConfigParser
import pylab

# Read the co nfig file for the model
defaults={"filetype":"TDI", "header_line":1,"start_line":2, "separator":",", "v_scale":1 }
config=ConfigParser.SafeConfigParser(defaults)
config.read("pcar.ini")

show_plot=config.getboolean('options', 'show_plot')
save_fit=config.getboolean('options', 'save_fit')
user_iterfunct=config.getboolean('options', 'print_each_step')

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

gcol=config.get('data', 'y-column')
vcol=config.get('data', 'x-column')
omega=pars['omega']
delta=pars['delta']
P=pars['P']
Z=pars['Z']

format=config.get("data", "filetype")
header=config.getint("data", "header_line")
start=config.getint("data", "start_line")
delim=config.get("data", "separator")

       
#import data
d=Stoner.AnalyseFile()
d.load(None, format, header, start, delim, delim)

# Convert string column headers to numeric column indices
gcol=d.find_col(gcol)
vcol=d.find_col(vcol)

# Get filename for title
filenameonly=os.path.basename(d.filename)
filenameonly=os.path.splitext(filenameonly)[0]
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

def Normalise(config, d, gcol, vcol):
    # Normalise the data
    Gn=config.getfloat('data', 'Normal_conductance')
    v_scale=config.getfloat("data", "v_scale")
    if config.has_option("options", "fancy_normaliser") and config.getboolean("options", "fancy_normaliser"):
        def quad(x, a, b, c): # linear fitting function
            return x*x*a+x*b+c
        vmax, vp=d.max(vcol)
        vmin, vp=d.min(vcol)
        p, pv=d.curve_fit(quad, vcol, gcol, bounds=lambda x, y:(x>0.9*vmax) or (x<0.9*vmin)) #fit a striaght line to the outer 10% of data
        print "Fitted normal conductance background of G="+str(p[0])+"V^2 +"+str(p[1])+"V+"+str(p[2])
        d.apply(lambda x:x[gcol]/quad(x[vcol], p[0], p[1], p[2]), gcol)
    else:
        d.apply(lambda x:x[gcol]/Gn, gcol)
    if config.has_option("options", "rescale_v") and config.getboolean("options", "rescale_v"):    
        d.apply(lambda x:x[vcol]*v_scale, vcol)
    return d

def offset_correct(config, d, gcol, vcol):
    """Centre the data - look for peaks and troughs within 5 of the initial delta value
        take the average of these and then subtract it.
    """
    peaks=d.peaks(gcol,len(d)/20,0,xcol=vcol,poly=4,peaks=True,troughs=True)
    peaks=filter(lambda x: abs(x)<4*delta['value'], peaks)
    offset=numpy.mean(numpy.array(peaks))
    print "Mean offset ="+str(offset)
    d.apply(lambda x:x[vcol]-offset, vcol)
    return d

if config.has_option("options", "normalise") and config.getboolean("options", "normalise"):
    d=Normalise(config, d, gcol, vcol)
    
if config.has_option("options", "remove_offset") and config.getboolean("options", "remove_offset"): 
    d=offset_correct(config, d, gcol, vcol)
    
#Plot the data while we do the fitting
if show_plot:
    p=Stoner.PlotFile(d)
    p.plot_xy(vcol,gcol, 'ro',title=filenameonly)
    time.sleep(2)


# Initialise the parameter information with the dictionaries defined at top of file
parinfo=[omega, delta, P, Z]

#Build a list of parameter values to iterate over
r=Stoner.DataFile()
r.column_headers=[x["parname"] for x in parinfo]
r.column_headers.append("chi^2")

steps=["fit"]
if config.has_option("options", "chi2_mapping") and config.getboolean("options", "chi2_mapping"):
    for p in parinfo:
            if p["fixed"]==True and p["step"]<>0:
                t=steps
                steps=[]
                for x in numpy.arange(p["limits"][0], p["limits"][1], p["step"]):
                    steps.extend([(p, x)])
                    steps.extend(t)
d.add_column(lambda x:1, 'Fit')
for step in steps:
    if isinstance(step, str) and step=="fit":
        if user_iterfunct==False:
            # Here is the engine that does the work
            m = d.mpfit(strijkers,vcol, gcol, parinfo, iterfunct=d.mpfit_iterfunct)
            print "Finished !"
        else:
            m = d.mpfit(strijkers,vcol, gcol, parinfo)
        if (m.status <= 0): # error message ?
            raise RuntimeError(m.errmsg)
        else:
            d.add_column(strijkers(d.column(vcol), m.params), index='Fit', replace=True)
            if show_plot:
                # And show the fit and the data in a nice plot
                p=Stoner.PlotFile(d)
                p.plot_xy(vcol,gcol,'ro',title=filenameonly)
                pylab.plot(p.column(vcol), p.column('Fit'),'b-')
            # Ok now we can print the answer
            for i in range(len(parinfo)):
                print parinfo[i]['parname']+"="+str(m.params[i])
                d[parinfo[i]['parname']]=m.params[i]
            
            chi2=chisquare(d.column(gcol), d.column('Fit'))
            d["Chi^2"]=chi2
            print "Chi^2:"+str(chi2)
            if save_fit:
                d.save(False)

            row=m.params
            row=numpy.append(row, chi2[0])
            r=r+row            
    elif isinstance(step, tuple):
        (p, x)=step
        p["value"]=x

if len(steps)>1:
    ch=[x['parname'] for x in parinfo]
    ch.append('Chi^2')
    r.column_headers=ch
    r.save(None)

    







