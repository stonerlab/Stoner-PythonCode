#############################################
#
# Fit provides some fitting functions for use with the Stoner.AnalyseFile methods
#
# $Id: Fit.py,v 1.1 2011/11/28 14:26:52 cvs Exp $
#
# $Log: Fit.py,v $
# Revision 1.1  2011/11/28 14:26:52  cvs
# Merge latest versions
#

import numpy as np
import scipy.constants.codata as consts

_kb=consts.physical_constants['Boltzmann constant'][0]/consts.physical_constants['elementary charge'][0]

def Linear(x, m, c):
    """Simple linear function"""
    return m*x+c

def Arrhenius(x, A, DE):
    """Arrhenius Equation without T dependendent prefactor"""
    return A*np.exp(-DE/(_kb*x))


def NDimArrhenius(x, A, DE, n):
    """Arrhenius Equation without T dependendent prefactor"""
    return Arrhenius(x**n, A, DE)

def  ModArrhenius(x, A, DE, n):
    """Arrhenius Equation with a variable T power dependent prefactor"""
    return (x**n)*Arrhenius(x, A, DE)

def PowerLaw(x, A, n):
    """Power Law Fitting Equation"""
    return A*x**n
