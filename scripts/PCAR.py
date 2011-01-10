#
# $Id: PCAR.py,v 1.1 2011/01/10 19:05:44 cvs Exp $
#$Log: PCAR.py,v $
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

# Import packages
import numpy
import scipy
import Stoner
import easygui



#gui to get filename and path
filename=easygui.fileopenbox(title = "Choose your file")
#import data
d=Stoner.DataFile(filename)

def myfunct(p, fjac=None, x=None, y=None, err=None)
    # Parameter values are passed in "p"
    # If FJAC!=None then partial derivatives must be comptuer.
    # FJAC contains an array of len(p), where each entry
    # is 1 if that parameter is free and 0 if it is fixed. 
    model = strijkers(x, p)
    # stop the calculation.
    status = 0
    return([status, (y-model)/err, pderiv]
    

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

 



#Starting Parameters
# omega, delta, P,Z
params=[0.55, 1.5, 0.2, 0.4]
#bounds (min,max)
omegaBound=[0.3,0.8]
deltaBound=[1,2.0]
pBound=[0.1,0.5]
zBound=[0.3,0.60]

#Fitting accuracy (termination condition when change in function is less than..)
fitAcc=1e-10
#Step size for derivative estimates
fitStep=1e-3
#Max number of iterations
maxIt=30




'







