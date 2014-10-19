"""
Stoner.nlfit  (non linear fitting)
===================================

Authors: Gavin Burnell, Rowan Temple, Nick Porter
Date: Original fitting code for PCAR GB 2011, sim adaptation and WL model NAP Feb 2013,

Rewritten and updated as object orientated code for generic functions and tunnelling models RCT March 2013
Description:  Generic non linear function fitting code using Levenberg Marquardt algorithm. Main class NLFit
is initialised with function to be fitted and ini file with parameter definitions.
"""

## Import packages
from Stoner.compat import *
import numpy as _np_
import Stoner
from scipy.stats import chisquare
import os
import time
if python_v3:
    import configparser as ConfigParser
else:
    import ConfigParser
import pylab ; pylab.ion() #interactive mode, works best with iPython

def nlfit(ini_file, func, data=None, chi2mapping=False):
    """Runs nlfit, taking data from the file path given in the ini_file and a fitting function

    Args:
        ini_file(string or file): Either the name of a file or an open file descriptor
        func (string or callable): either a string with the name of a function in Stoner.FittingFuncs or a callable object

    Keyword Arguments:
        data (string or DataFile): Either a Stoner.DataFile instance or None to force a file open dialog
        chi2mapping (bool): Indicates whether chi^2 mappingmode is engaged

    Return:
        An NLFit instance
    """
    if isinstance(func,str):
        try:
            func = getattr(Stoner.FittingFuncs, func)
        except:
            raise ValueError('Supplied function not recognised')
    elif not callable(func):
            raise ValueError('Supplied function must be either a string or callable object')
    try:
        if isinstance(ini_file,str):
            f=open(ini_file, 'r')
            f.close()
        elif not isinstance(ini_file,file):
            raise ValueError
    except:
        raise ValueError('Could not open ini file')
    t=NLFit(ini_file, func, data, chi2mapping)
    t.run()#
    #Return Stoner.Analysis file instance of data and an instance of matplotlib.axes for the plot
    if chi2mapping: return t.output, t.plotout, t.param_steps
    return t.output, t.plotout

class NLFit:
    """Class containing the code to do a NLFit or chi^2 mapping of a set of data

    Attributes:
        simulate (bool): Just calculate data, don't fit
        function (callable): Function to fit
        ini_file (string): File containing settings for fitting


    """


    def __init__(self, ini_file, func, data=None, chi2mapping=False):
        """ini_file is file name for .ini options file
           function is a function that you wish to fit with (function takes x array and parameter list as its 2 arguments)
           defaults only needs changing if you don't have a TDI file
        """
        # Config file structure
        if isinstance(func,str):
            try:
                func = getattr(Stoner.FittingFuncs, func)
            except:
                raise ValueError('Supplied function not recognised')
        elif not callable(func):
                raise ValueError('Supplied function must be either a string or callable object')
        self.fit_func = func
        self.ini_file = ini_file

        #Read ini file
        self.config = ConfigParser.SafeConfigParser()
        if isinstance(self.ini_file,str):
            self.config.read(self.ini_file)
        elif isinstance(self.ini_file,file):
            self.config.readfp(self.ini_file)

        parnames = self.config.get('data', 'parnames') #returns a list of comma separated names of parameters
        parnames = [i.strip() for i in parnames.split(',')]
        self.parameter_names=parnames
        for section in parnames:
            pars={  'value': self.config.getfloat(section, 'value'),
                      'fixed': self.config.getboolean(section, 'fixed'),
                     'limited': (self.config.getboolean(section, 'lower_limited'), self.config.getboolean(section, 'upper_limited')),
                      'limits':(self.config.getfloat(section, 'lower_limit'), self.config.getfloat(section, 'upper_limit')),
                      'parname': self.config.get(section, 'name'),
                      'step': self.config.getfloat(section, 'step'),
                      'mpside':self.config.getint(section, 'side'),
                      'mpmaxstep':self.config.getfloat(section, 'maxstep'),
                      'tied':self.config.get(section, 'tied'),
                      'mpprint': self.config.getboolean(section, 'print')}
            self.__setattr__(section,pars[section]) # make each parameter into an attribute of this class




        self.data_input=Stoner.DataFile()
        self.data_input = data
        self.chi2mapping = chi2mapping
        assert hasattr(function, '__call__') #check function is actually a function
        self.output = None #initialise an instance variable that can later be used to store output data
        self.plotout = None #initialise an instance variable later used for storing plot axes instance
        self.param_steps = None

    def run(self,action=None):
        """Simple wrapper to do the default action for the fitting isntance.

        Keyword Arguements:
            action (string or None): What to go an do, may be "simulate","fit", "map" or None (default)

        Returns:
            Nothing.

        """

        #work out whether to simulate or not

        if action is None:
            if self.simulate is None: #Only look in the config file if we're not overriding the simulate property
                self.simulate = self.config.getboolean('options', 'simulate')
        elif action=="simulate":
            self.simulate=True
        elif action=="fit":
            self.simulate=False
            self.chi2mapping=False
        elif action=="map":
            self.simulate=False
            self.chi2mapping=True
        else:
            raise RuntimeError("Unable to decide what to do in run()!")

        #run sim
        if self.simulate:
            self._runsim()
        else:
            self._runfit()

    def _runsim(self):
        """
        Simulate data to check how the function works
        """
        sim_xlim = self.config.get('options', 'sim_xlim') #returns eg '-8,9'
        sim_xlim = [float(i.strip()) for i in sim_xlim.split(',')]
        x = _np_.linspace(sim_xlim[0], sim_xlim[1], 300)
        parnames = self.config.get('data', 'parnames') #returns a list of comma separated names of parameters
        parnames = [i.strip() for i in parnames.split(',')]
        params = [self.config.getfloat(item,'value') for item in parnames]
        fit = self.fit_func(x, params)

        #save the simulation
        sim = Stoner.DataFile(_np_.column_stack((x,fit)))
        xcolname = self.config.get('data', 'x-column')
        ycolname = self.config.get('data', 'y-column')
        sim.column_headers=[xcolname, ycolname]
        for i in range(len(params)):
            print(parnames[i]+'='+str(params[i]))
            sim[parnames[i]]=params[i]

        if self.config.getboolean('options', 'save_fit'):
            sim.save(None)

        self.output = sim #instance variable if further work is needed

        #plot the simulation
        if self.config.getboolean('options', 'show_plot'):
            pylab.xlabel(self.config.get('data', 'x-column'))
            pylab.ylabel(self.config.get('data', 'y-column'))
            pylab.plot(x, fit)
            #pylab.ylim([-0.0005,0.005])
            self.plotout = pylab.gca() #save plot axes in case further work required

    def _runfit(self):
        """runs the fit using Stoner.Analysis.mpfit function
        """
        show_plot = self.config.getboolean('options', 'show_plot')
        annotate_plot = self.config.getboolean('options', 'annotate_plot')
        save_fit = self.config.getboolean('options', 'save_fit')
        print_iterfunct = self.config.getboolean('options', 'print_each_step')

        parlist=[self.__getattr__(section) for section in self.parameter_names]

        #import data
        if self.data_input == None:
            fit_file=None
            if self.config.has_option('data','fit_file'):
                fit_file = self.config.get('data', 'fit_file')
                fit_file = (os.sep).join(os.path.split(fit_file)) #make filename platform independent
            fformat = self.config.get('data', 'filetype')
            if fformat=='csv':
                header = self.config.getint('data', 'header_line')
                start = self.config.getint('data', 'start_line')
                d=Stoner.FileFormats.CSVFile()
                if fit_file==None: print('Open data file to fit')
                d.load(fit_file,header,start)
                filename=d.filename
                d=Stoner.AnalyseFile(d)
                d.filename=filename
            else:
                d=Stoner.AnalyseFile()
                if fit_file==None: print('Open data file to fit')
                d.load(fit_file)
        else:
            assert(isinstance(self.data_input, Stoner.AnalyseFile) or \
                    isinstance(self.data_input, Stoner.DataFile))
            d=Stoner.AnalyseFile(self.data_input)
        # Convert string column headers to numeric column indices
        xcolname = self.config.get('data', 'x-column')
        ycolname = self.config.get('data', 'y-column')
        xcol=d.find_col(xcolname) #finds position of col in array e.g. 0th or 1st column
        ycol=d.find_col(ycolname)

        # Discard unwanted data
        discard = self.config.getboolean('data','discard')
        if discard:
            x_limits = self.config.get('data','x_limits')
            x_limits = [float(i) for i in x_limits.split(',')]
            print('x_limits=', x_limits)
            d=d.del_rows(xcol,lambda x,y:x>x_limits[1] or x<x_limits[0])

        # Get filename for title
        filenameonly='Fit'
        if d.filename is not None:
            filenameonly=os.path.basename(d.filename)
            filenameonly=os.path.splitext(filenameonly)[0]

        self.data_input=d
        # Here is some code to allow the user to supply a preprocessing script.
        # Simply subclass NLFit with a preprocessing method and set the method name in
        # config file key user_options.preprocess
        if self.cconfig.has_attribute('user_options','preprocess'):
            try:
                preproc=self.__getattr__(self.config.get('user_options','preprocess'))
            except AttributeError:
                pass
            else:
                if callable(preproc):
                    preproc(self)

            d=self.data_input

        ################## Fitting ###########################
        #Plot the data while we do the fitting
        if show_plot:
            plotfig = pylab.figure()
            p=Stoner.PlotFile(d)
            p.plot_xy(xcol, ycol, 'ro', title=filenameonly, figure=plotfig)
            time.sleep(2)

        #Build a list of parameter values to iterate over (only to be saved if using chi^2 mapping)
        r=Stoner.DataFile()
        r.column_headers=[x['parname'] for x in parlist]
        r.column_headers.append('chi^2')

        steps=['fit'] #one step for no chi^2 mapping
        if self.chi2mapping:
            for p in parlist:
                    if p['fixed']==True and p['step']!=0:
                        t=steps
                        steps=[]
                        for x in _np_.arange(p['limits'][0], p['limits'][1], p['step']):
                            steps.extend([(p, x)])
                            steps.extend(t)

        #################   fitting    ########################################
        d.add_column(lambda x:1, 'Fit')
        for step in steps:
            if isinstance(step, str) and step=='fit':
                if print_iterfunct==False:
                    # Here is the engine that does the work
                    m = d.mpfit(self.fit_func, xcol, ycol, parlist, iterfunct=d.mpfit_iterfunct)
                    print('Finished!')
                else:
                    m = d.mpfit(self.fit_func,xcol, ycol, parlist)
                if (m.status <= 0): # error message ?
                    raise RuntimeError(m.errmsg)
                else:
                    d.add_column(self.fit_func(d.column(xcol), m.params), index='Fit', replace=True)
                    if show_plot:
                        # And show the fit and the data in a nice plot
                        p=Stoner.PlotFile(d)
                        p.plot_xy(xcol,ycol,'ro',title=filenameonly, figure=plotfig)
                        pylab.plot(p.column(xcol), p.column('Fit'),'b-')
                        ax = pylab.gca() #axes instance for further internal use
                        self.plotout = plotfig #output a figure instance for further editing
                    # Ok now we can print the answer
                    for i in range(len(parlist)):
                        print(self.parameter_names[i]+'='+str(m.params[i]))
                        d[self.parameter_names[i]]=m.params[i]

                    chi2=chisquare(d.column(ycol), d.column('Fit'))
                    d['Chi^2']=chi2[0]  #chi2 is [chi2 value, p value]
                    print('Chi^2:'+str(chi2[0]))
                    if save_fit:
                        d.save(False)
                    self.output = d.clone
                    if show_plot and annotate_plot:
                        param_text = [name+'='+'{:.4g}'.format(d[name]) for name in self.parameter_names]
                        param_text = '\n'.join(param_text)
                        pylab.text(0.1,0.9,param_text,ha='left',va='top',transform=ax.transAxes)
                    row=m.params
                    row=_np_.append(row, chi2[0])
                    r=r+row
            elif isinstance(step, tuple):
                (p, x)=step
                p['value']=x
        if len(steps)>1:
            #if using chisquare mapping save the r data
            if save_fit:
                r.save(None)
            self.param_steps = r









