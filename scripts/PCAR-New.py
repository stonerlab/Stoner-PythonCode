"""Python  script for Analysing PCAR data using Stoner classes and lmfit

Gavin Burnell g.burnell@leeds.ac.uk

"""

# Import packages

import numpy as np
import matplotlib.pyplot as plt
import Stoner.Plot as SP
import Stoner.Analysis as SA
from Stoner.Fit import Strijkers,quad
from Stoner.FileFormats import CSVFile

import ConfigParser
from distutils.util import strtobool

class working(SA.AnalyseFile,SP.PlotFile):
    """Utility class to manipulate data and plot it"""

    def __init__(self,*args,**kargs):
        inifile=__file__.replace(".py",".ini")
        config=ConfigParser.SafeConfigParser()
        config.read(inifile)
        self.config=config


        #Some config variables we'll need later
        self.show_plot=config.getboolean('prefs', 'show_plot')
        self.save_fit=config.getboolean('prefs', 'save_fit')
        self.report=config.getboolean('prefs', 'print_report')
        self.fancyresults=config.has_option("prefs","fancy_result") and config.getboolean("prefs","fancy_result")

        #Build the variables for the fitting
        self.pars=dict()
        parnames=['omega', 'delta', 'P', 'Z']
        vars={"value":1.0,"max":None,"min":None,"vary":True,"step":0.1,"symbol":"?"}
        vartypes={"value":float,"min":float,"max":float,"step":float,"vary":strtobool,"symbol":str}
        for section in parnames:
            self.pars[section]=dict()
            for var in vars:
                if config.has_option(section,var):
                    self.pars[section][var]=vartypes[var](config.get(section,var))
                else:
                    self.pars[section][var]=vars[var]
            self.__setattr__(section,self.pars[section])

        model=Strijkers()
        for section in parnames:
            model.set_param_hint(section,value=self.pars[section]["value"],min=self.pars[section]["min"],max=self.pars[section]["max"],vary=self.pars[section]["vary"])

        self.model=model

        # Take care of loading the data and preparing it.
        self.format=config.get("data_format", "filetype")
        if self.format=="csv":
            self.header=config.getint("data_format", "header_line")
            self.start=config.getint("data_format", "start_line")
            self.delim=config.get("data_format", "separator")

        super(working,self).__init__(*args,**kargs)

    def load(self, filename=None, auto_load=True,  filetype=None,  *args, **kargs):
        if self.format=="csv":
            d=CSVFile()
            d=d.load(filename,self.header,self.start,self.delim,self.delim)
        else:
            d=SP.PlotFile(*args, **kargs)
        self.metadata.update(d.metadata)
        self.data=d.data
        self.column_headers=d.column_headers
        self.filename=d.filename

        gcol=self.find_col(self.config.get("data_format", 'y-column'))
        vcol=self.find_col(self.config.get("data_format", 'x-column'))

        self.setas[gcol]="y"
        self.setas[vcol]="x"
        self.vcol=vcol
        self.gcol=gcol

        discard=self.config.get("data_format",'discard')
        if discard:
            v_limit=self.config.get("data_format",'v_limit')
            self.del_rows(vcol,lambda x,y:abs(x)>v_limit)
        return self

    def Normalise(self):
        """Normalise the data if the relevant options are turned on in the config file.

        Use either a simple normalisation constant or go fancy and try to use a background function.
        """

        if self.config.has_option("prefs", "normalise") and self.config.getboolean("prefs", "normalise"):
            Gn=self.config.getfloat("data_format", 'Normal_conductance')
            v_scale=self.config.getfloat("data_format", "v_scale")
            if self.config.has_option("prefs", "fancy_normaliser") and self.config.getboolean("prefs", "fancy_normaliser"):
                vmax, vp=self.max(self.vcol)
                vmin, vp=self.min(self.vcol)
                p, pv=self.curve_fit(quad, bounds=lambda x, y:(x>0.9*vmax) or (x<0.9*vmin))
                print "Fitted normal conductance background of G="+str(p[0])+"V^2 +"+str(p[1])+"V+"+str(p[2])
                self["normalise.coeffs"]=p
                self["normalise.coeffs_err"]=np.sqrt(np.diag(pv))
                self.apply(lambda x:x[self.gcol]/quad(x[self.vcol], *p), self.gcol)
            else:
                self.apply(lambda x:x[self.gcol]/Gn, self.gcol)
            if self.config.has_option("prefs", "rescale_v") and self.config.getboolean("prefs", "rescale_v"):
                self.apply(lambda x:x[self.vcol]*v_scale, self.vcol)
        return self

    def offset_correct(self):
        """Centre the data - look for peaks and troughs within 5 of the initial delta value
            take the average of these and then subtract it.
        """
        if self.config.has_option("prefs", "remove_offset") and self.config.getboolean("prefs", "remove_offset"):
            peaks=self.peaks(self.gcol,len(self)/20,0,xcol=self.vcol,poly=4,peaks=True,troughs=True)
            peaks=filter(lambda x: abs(x)<4*self.delta['value'], peaks)
            offset=np.mean(np.array(peaks))
            print "Mean offset ="+str(offset)
            self.apply(lambda x:x[self.vcol]-offset, self.vcol)
        return self

    def Prep(self):
        """Prepare to start fitting data. Creating a results file if necessary"""
        self.Normalise()
        self.offset_correct()

        # Construct a list of all the starting values to work through
        grid=dict()
        for section in self.pars:
            if not self.pars[section]["vary"] and  self.pars[section]["step"]>0.0:
                grid[section]=np.arange(self.pars[section]["min"],self.pars[section]["max"],self.pars[section]["step"])
                self.pars[section]["scanned"]=True
            else:
                grid[section]=np.array([self.pars[section]["value"]])
                self.pars[section]["scanned"]=False
        [P,Z,omega,delta]=np.meshgrid(grid["P"],grid["Z"],grid["omega"],grid["delta"])
        self.par_values=np.column_stack((P.ravel(),Z.ravel(),omega.ravel(),delta.ravel()))

        #Copy config into metadata
        for section in self.config.sections():
           for key,value in self.config.items(section):
               k="{}.{}".format(section.lower(),key.lower())
               self[k]=self.metadata.string_to_type(value)

        #Prep the results file
        if self.par_values.shape[0]>1:
            self.results=self.clone
            self.results.data=np.ma.array([])
            self.results.column_headers=[]
        else:
            self.results=None

        self&=np.zeros(len(self))
        self.column_headers[-1]="Fit"
        return self

    def DoOneFit(self,pvals):
        """Assuming a prepared state, run one fit the specified starting values."""
        p0={k:v for k,v in zip(["P","Z","omega","delta"],pvals)}
        fit=self.lmfit(self.model,p0=p0,result="Fit",replace=True,header="Fit")
        if self.par_values.shape[0]>1:
            row=[]
            heads=[]
            for k in fit.best_values:
                row.append(self[k])
                row.append(self[k+"_err"])
                heads.extend([k,"d"+k])
                self.pars[k]["best fit"]=fit.best_values[k]
                self.pars[k]["best fit err"]=self[k+"_err"]
            row.append(fit.chisqr)
            row.append(fit.nfev)
            heads.extend([r"$\chi^2$",r"$N^o \mathrm{Iterations}$"])
            self.results.column_headers=heads
            self.results+=np.array(row)
        if self.show_plot:
            self.plot_results()
        if self.save_fit:
            self.save(False)
        if self.report:
            print fit.fit_report()
        return self

    def plot_results(self):
        """Do the plotting of the data and the results"""
        self.figure()# Make a new figure and show the results
        self.plot_xy(self.vcol,[self.gcol,"Fit"],fmt=['ro','b-'],label=["Data","Fit"])
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=2)
        if self.fancyresults:
            answer=[ "${}$={}".format(self.pars[i]['symbol'],self.quoteresult(i)) for i in self.pars]
            answer.append("$\chi^2={}$".format(round(self["chi^2"],8)))
        else:
            answer=[ "${}={}$".format(self.pars[i]['symbol'],round(self[i],3)) for i in self.pars]
        plt.annotate("\n".join(answer), xy=(0.05, 0.65), xycoords='axes fraction',bbox=bbox_props,fontsize=11)
        return self

    def quoteresult(self,i):
        """Quote with an error"""
        res=self[i]
        err=self[i+"_err"]
        if err<=0.0:
            return "{}".format(res)
        else:
            for j in range(2): # two passes in case we round up the first time
                errmag=int(np.floor(np.log10(abs(err))))
                err=round(err/(10.0**errmag))*10.0**errmag
            if errmag<0:
                res=round(res,-errmag)
                fmt="${{:.{}f}}\\pm{{:.{}f}}$".format(-errmag,-errmag)
            else:
                res=10**errmag*round(res*10**-errmag)
                fmt="${:.0f}\\pm{:.0f}$"
            return fmt.format(res,err)


    def DoAllFits(self):
        self.Prep()
        for pvals in self.par_values:
            self.DoOneFit(pvals)
        if self.results is not None:
            for k in self.pars:
                if self.pars[k]["scanned"]:
                    self.results.figure()
                    self.results.plot_xy(k,"chi",plotter=plt.semilogy)
            self.results.save(False)

d=working(False)
d.DoAllFits()









