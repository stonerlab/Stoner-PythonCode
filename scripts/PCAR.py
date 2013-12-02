"""Python  script for Analysing PCAR data

Gavin Burnell g.burnell@leeds.ac.uk

"""

# Import packages

import numpy
import scipy
import Stoner
from scipy.stats import chisquare
import os
import time
import ConfigParser
import pylab

# Read the co nfig file for the model
config=ConfigParser.SafeConfigParser()
config.read("pcar.ini")

show_plot=config.getboolean('prefs', 'show_plot')
save_fit=config.getboolean('prefs', 'save_fit')
user_iterfunct=config.getboolean('prefs', 'print_each_step')

pars=dict()
parnames=['omega', 'delta', 'P', 'Z']
for section in parnames:
    pars[section]=dict()
    pars[section]['value']=config.getfloat(section, 'value')
    pars[section]['fixed']=config.getboolean(section, 'fixed')
    pars[section]['limited']=[config.getboolean(section, 'lower_limited'), config.getboolean(section, 'upper_limited')]
    pars[section]['limits']=[config.getfloat(section, 'lower_limit'), config.getfloat(section, 'upper_limit')]
    pars[section]['parname']=config.get(section, 'name')
    pars[section]['symbol']=config.get(section, 'symbol')
    pars[section]['step']=config.getfloat(section, 'step')
    pars[section]['mpside']=config.getint(section, 'side')
    pars[section]['mpmaxstep']=config.getfloat(section, 'maxstep')
    pars[section]['tied']=config.get(section, 'tied')
    pars[section]['mpprint']=config.getboolean(section, 'print')

gcol=config.get("data_format", 'y-column')
vcol=config.get("data_format", 'x-column')
discard=config.get("data_format",'discard')
if discard:
    v_limit=config.get("data_format",'v_limit')
omega=pars['omega']
delta=pars['delta']
P=pars['P']
Z=pars['Z']

format=config.get("data_format", "filetype")
header=config.getint("data_format", "header_line")
start=config.getint("data_format", "start_line")
delim=config.get("data_format", "separator")


#import data
if format=="csv":
    d=Stoner.FileFormats.CSVFile()
    d.load(None,header,start)
    filename=d.filename
    d=Stoner.AnalyseFile(d)
    d.filename=filename
else:
    d=Stoner.AnalyseFile()
    d.load(None)

# Convert string column headers to numeric column indices
gcol=d.find_col(gcol)
vcol=d.find_col(vcol)

if discard:
    d=d.del_rows(vcol,lambda x,y:abs(x)>v_limit)

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
    Gn=config.getfloat("data_format", 'Normal_conductance')
    v_scale=config.getfloat("data_format", "v_scale")
    if config.has_option("prefs", "fancy_normaliser") and config.getboolean("prefs", "fancy_normaliser"):
        def quad(x, a, b, c): # linear fitting function
            return x*x*a+x*b+c
        vmax, vp=d.max(vcol)
        vmin, vp=d.min(vcol)
        p, pv=d.curve_fit(quad, vcol, gcol, bounds=lambda x, y:(x>0.9*vmax) or (x<0.9*vmin)) #fit a striaght line to the outer 10% of data
        print "Fitted normal conductance background of G="+str(p[0])+"V^2 +"+str(p[1])+"V+"+str(p[2])
        d["normalise.coeffs"]=p
        d["normalise.coeffs_err"]=numpy.sqrt(numpy.diag(pv))
        d.apply(lambda x:x[gcol]/quad(x[vcol], *p), gcol)
    else:
        d.apply(lambda x:x[gcol]/Gn, gcol)
    if config.has_option("prefs", "rescale_v") and config.getboolean("prefs", "rescale_v"):
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

if config.has_option("prefs", "normalise") and config.getboolean("prefs", "normalise"):
    d=Normalise(config, d, gcol, vcol)

if config.has_option("prefs", "remove_offset") and config.getboolean("prefs", "remove_offset"):
    d=offset_correct(config, d, gcol, vcol)

#Plot the data while we do the fitting
if show_plot:
    p=Stoner.PlotFile(d)
    f=p.plot_xy(vcol,gcol, 'ro',title=filenameonly)
    time.sleep(2)


# Initialise the parameter information with the dictionaries defined at top of file
parinfo=[omega, delta, P, Z]

#Build a list of parameter values to iterate over
r=Stoner.DataFile()
r.column_headers=[x["parname"] for x in parinfo]
r.column_headers.append("chi^2")

steps=["fit"]
if config.has_option("prefs", "chi2_mapping") and config.getboolean("prefs", "chi2_mapping"):
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
                p.fig=f
                # Ok now we can print the answer
                answer=[ "${}$={}".format(parinfo[i]['symbol'],round(m.params[i],3)) for i in range(len(parinfo))]
                if len(steps)==1:
                    p.plot_xy(vcol,"Fit","b-",title=filenameonly)
                    pylab.legend()
                    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=2)
                    pylab.annotate("\n".join(answer), xy=(0.05, 0.75), xycoords='axes fraction',bbox=bbox_props,fontsize=14)
                else:
                    p.plot_xy(vcol,"Fit",label=" ".join(answer),title=filenameonly)
                    pylab.legend()
                print "\n".join(answer)
            for i in range(len(parinfo)):
                d[parinfo[i]['parname']]=m.params[i]

            chi2,pp=chisquare(d.column(gcol), d.column('Fit'))
            d["Chi^2"]=float(chi2)
            print "Chi^2: {}".format(round(chi2,5))
            if save_fit:
                for section in config.sections():
                   for key,value in config.items(section):
                       k="{}.{}".format(section.lower(),key.lower())
                       d[k]=d.metadata.string_to_type(value)
                d.save(False)

            row=m.params
            row=numpy.append(row, chi2)
            r=r+row
    elif isinstance(step, tuple):
        (p, x)=step
        p["value"]=x

if len(steps)>1:
    ch=[x['parname'] for x in parinfo]
    ch.append('Chi^2')
    r.column_headers=ch
    r.save(None)









