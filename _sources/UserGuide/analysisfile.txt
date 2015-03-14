********************
Analysing Data Files
********************
.. currentmodule:: Stoner.Analysis

Normalising and Basic Maths Operations
======================================

Several methods are provided to assist with common data fitting and preparation tasks, such as normalising columns,
adding and subtracting columns.::

   a.normalise('data','reference',header='Normalised Data',replace=True)
   a.normalise(0,1)
   a.normalise(0,3.141592654)
   a.normalise(0,a2.column(0))

The :py:meth:`AnalyseFile.normalise` method simply divides the data column by the reference column. By default the normalise method
replaces the data column with the new (normalised) data and appends '(norm)' to the column header. The keyword arguments
*header* and *replace* can override this behaviour. The third variant illustrates normalising to a constant
(note, however, that if the second argument is an integer it is treated as a column index and not a constant).
The final variant takes a 1D array with the same number of elements as rows and uses that to normalise to.
A typical example might be to have some baseline scan that one is normalising to.::

   a.subtract('A','B'm header="A-B",replace=True)
   a.subtract(0,1)
   a.subtract(0,3.141592654)
   a.subtract(0,a2.column(0))

As one might expect form the name, the :py:meth:`AnalyseFile.subtract` method subtracts the second column form the first.
Unlike :py:meth:`AnalyseFile.normalise` the first data column will not be replaced but a new column inserted and a new header
(defaulting to 'column header 1 - column header 2') will be created. This can be overridden with the *header* and *replace*
keyword arguments. The next two variants of the :py:meth:`AnalyseFile.subtract` method work in an analogous manner to the
:py:meth:`AnalyseFile.normalise` methods. Finally the :py:meth:`AnalyseFile.add` method allows one to add two columns in a
similar fashion::

   a.add('A','B',header='A plus B',replace=False)
   a.add(0,1)
   a.add(0,3.141592654)
   a.add(0,a2.column(0))

For completeness we also have::

   a.divide('A','B',header='A/B', replace=True)
   a.multiply('A','B',header='A*B', replace=True)

with variants that take either a 1D array of data or a constant instead of the B column index.

One might wish to split a single data file into several different data files each with the rows of the original
that have a common unique value in one data column, or for which some function of the complete row determines which datafile
each row belongs in. The :py:meth:`AnalyseFile.split` method is useful for this case.::

   a.split('Polarisation')
   a.split('Temperature',lambda x,r:x>100)
   a.split(['Temperature','Polarisation'],[lambda x,r:x>100,None])

In these examples we assume the :py:class:`AnalyseFile` has a data column 'Polarisation' that takes two (or more) discrete values
and a column 'Temperature' that contains numbers above and below 100.

The first example would return a :py:class:`Stoner.Folders.DataFolder` object  containing two separate isntances of :py:class:`AnalyseFile`  which
would each contain the rows from the original data that had each unique value of the polarisation data. The second example would
produce a :py:class:`Stoner.Folders.DataFolder` object containing two :py:class:`AnalyseFile` objects for the rows with temperature above and below 100.
The final example will result in a :py:class:`Stoner.Folder.DataFolder` object that has two groups each of which contains
:py:class:`AnalyseFile` objects for each polarisation value.

Curve Fitting
=============

Simple polynomial Fits
----------------------

Simple least squares fitting of polynomial functions is handled by the
:py:meth:`AnalyseFile.polyfit` method::

   a.polyfit(column_x,column_y,polynomial_order, bounds=lambda x, y:True,result="New Column")

This is a simple pass through to the numpy routine of the same name. The x and y
columns are specified in the first two arguments using the usual index rules for
the Stoner package. The routine will fit multiple columns if *column_y*
is a list or slice. The *polynomial_order* parameter should be a simple integer
greater or equal to 1 to define the degree of polynomial to fit. The *bounds*
function follows the same rules as the *bounds* function in
:py:meth:`Stoner.Core.DataFile.search` to restrict the fitting to a limited range of rows. The
method returns a list of coefficients with the highest power first. If
*column_y* was a list, then a 2D array of coefficients is returned.

If *result* is specified then a new column with the header given by the *result* parameter will be
created and the fitted polynomial evaluated at each point.

Simple function fitting
-----------------------

For more general curve fitting operations the :py:meth:`AnalyseFile.cruve_fit`
method can be employed. Again, this is a pass through to the numpy routine of
the same name.::

   a.curve_fit(func,  xcol, ycol, p0=None, sigma=None,bounds=lambda x, y: True, result=True,replace=False,header="New Column" )

The first parameter is the fitting function. This should have prototype
``y=func(x,p[0],p[1],p[2]...)``: where *p* is a list of fitting parameters.
The *p0* parameter contains the initial guesses at the fitting
parameters, the default value is 1. *xcol* and *ycol* are the x
and y columns to fit. This method cannot handle multiple y columns.
*sigma*, if present, provides the weightings for each datapoint and so
should also be an array of the same length as the x and y data. Fianlly, the
*bounds* function can be used to restrict the fitting to only a subset of the rows
of data.

:py:meth:`AnalyseFile.curve_fit` returns a list of two arrays ``[popt,pcov]``:
where *popt* is an array of the optimal fitting parameters and
*pcov* is a 2D array of the co-variances between the parameters.

If *result* is not **None** then the fitted data is added to the :py:class:`AnalyseFile`
object. Where it is added depends on the combination of the *result*, *replace*
and *header* parameters. If *result* is a string or integer it is interpreted as a column
index at which the fitted data will be inserted (*replace* **False**) or overwritten over the existing data (*replace* **True**).
The fitted data will be given the column header *header* unless *header* is not a string, in which ase the column
will be called 'Fitted with ' and the name of the function *func*.

Fitting with limits
-------------------

The Stoner package has a number of alternative fitting mechanisms that supplement the standard
:py:func:`scipy.optimize.curve_fit` function. New in version 0.2 onwards of the Package is an interface to the
*lmfit* module.

Non-linear curve fitting with lmfit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

lmfit provides a flexible way to fit complex models to experimental data in a pythonesque object-orientated fashion.
A full description of the lmfit module is given in the `lmffit documentation <href=http://lmfit.github.io/lmfit-py/>`. The
:py:meth:`AnalyseFile.lmfit` method is used to interact with lmfit.

In order to use :py:meth:`AnalyseFile.lmfit`, one requires a c instance. This describes a function
and its independent and fittable parameters, whether they have limits and what the limits are. The :py:mod:`Stoner.Fit` module contains
a series of :py:class:lmfit.model.Model` subclasses that represent various models used in condensed matter physics.

The operation of :py:meth:`AnalyseFile.lmfit` is very similar to that of :py:meth:`AnalyseFile.curve_fit`::

    from Stoner.Fit import Arrehenius
    model=Arrehenius(A=1E7,DE=0.01)
    fit=a.lmfit(model,xcol="Temp",ycol="Cond",result=True,header="Fit")
    print fit.fit_report()
    print a["A"],a["A_err"],a["chi^2"],a["nfev"]

In this example we would be fitting an Arrehenius model to data contained inthe 'Temp' and 'Cond' columns. The resulting
fit would be added as an additional colum called fit. In addition, details of the fit are added as metadata to the current :py:class:`AnalyseFile`.

The return value from :py:meth:`AnalyseFile.lmfit` is controlled by the *output* keyword parameter. By default it is the :py:class:`lmfit.model.ModelFit`
instance. This contains all the information about the fit and fitting process.

Non-linear curve fitting with mpfit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current versions of the Stoner Package contain a second constrained non-linear fitting package, *mpfit*.

This is also suitable for cases where one requires more flexibility in fitting data, in particular
where the fitting parameters are constrained, the module is accessed thrpugh the :py:meth:`AnalyseFile.mpfit`
method. This is a pass through to the :py:mod:`Stoner.mpfit` module.::

   a.mpfit(func,  xcol, ycol, p_info,  func_args=dict(), sigma=None,bounds=lambda x, y: True, **mpfit_kargs )

In this case, the *func* argument takes a slightly different
prototype:``def func(x,parameters, **func_args)`` where *parameters*
is a list of the fitting parameters and *func_args* provides a
dictionary of fixed \ie non-fitting parameters. *xcol* and *ycol*
are the column indices for the x and y data, *bounds* is a bounding
function to select only those rows to use for fitting the function, and
*sigma* are the weightings for each data-point. The remaining arguments
are a dictionary of keywords to pass through to the :py:mod:`Stoner.mpfit` routine and
*p_info* which is a list of dictionaries which is used to control the
parameters in the fit. This described below.

*p_info* contains one element for each parameter used to fit the data.
Each element is a dictionary with the following keys:

*   [value] the starting parameter value (but see the START_PARAMS parameter
       for more information).
*   [fixed] a boolean value, whether the parameter is to be held fixed or
       not.  Fixed parameters are not varied by MPFIT, but are passed on to MYFUNCT for
       evaluation.
*   [limited] a two-element boolean array.  If the first/second element is
       set, then the parameter is bounded on the lower/upper side.  A parameter can be
       bounded on both sides.  Both LIMITED and LIMITS must be given together.
*   [limits] a two-element float array.  Gives the parameter limits on the
       lower and upper sides, respectively.  Zero, one or two of these values can be
       set, depending on the values of LIMITED.  Both LIMITED and LIMITS must be given
       together.
*   [parname] a string, giving the name of the parameter.  The fitting code
       of MPFIT does not use this tag in any way.  However, the default iterfunct will
       print the parameter name if available.
*   [step] the step size to be used in calculating the numerical derivatives.
       If set to zero, then the step size is	computed automatically.  Ignored when
       AUTODERIVATIVE=0.
*   [mpside] the sidedness of the finite difference when computing numerical
       derivatives.  This field can take four values:

      *    [0]one-sided derivative computed automatically
      *    [1]one-sided derivative ``(f(x+h) - f(x)  )/h``
      *    [-1] one-sided derivative ``(f(x)   - f(x-h))/h``
      *    [2] two-sided derivative ``(f(x+h) - f(x-h))/(2*h)``
               Where H is the STEP parameter described above.  The "automatic"
               one-sided derivative method will chose a direction for the finite difference
               which does not violate any constraints.  The other methods do
               not perform this check.  The two-sided method is in principle more precise, but
               requires twice as many function evaluations.  **Default: 0**.

*   [mpmaxstep] the maximum change to be made in the parameter value.  During
       the fitting process, the parameter will never be changed by more than this value
       in one iteration. A value of 0 indicates no maximum.  **Default: 0**.
*   [tied] a string expression which 'ties' the parameter to other	free or
       fixed parameters.  Any expression involving	constants and the parameter
       array P are permitted.Example: if parameter 2 is always to be twice parameter 1
       then use the following: ``parinfo(2).tied = '2 * p(1)'``. Since they are
       totally constrained, tied parameters are considered to be fixed; no errors are
       computed for them.[ NOTE: the PARNAME can't be used in expressions. ]
*   [mpprint] if set to 1, then the default iterfunct will print the
       parameter value.  If set to 0, the parameter value will not be printed.  This
       tag can be used to selectively print only a few parameter values out of
       many.**Default: 1** (all parameters printed)

Non-linear curve fitting with initialisation file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to fit your data to a non-linear function more complicated than a polynomial you can use
:py:meth:`Stoner.nlfit.nlfit` or equivalently if you have an :py:class:`AnalyseFile` instance of your data
you can call :py:meth:`AnalyseFile.nlfit`. This performs a non-linear least squares fitting algorithm to your data
and returns the :py:class:`AnalyseFile` instance used with an additional final column that is the fit,
it also plots the fit. There is an example run script, ini file and data file in ``/scripts/`` in the github repository.
Hhave a look at them to see how to use this function.

The function to fit to can either be created by the user and passed in or one of a library of current existing functions
can be used from the :py:mod:`Stoner.FittingFuncs` (just pass in the name of the function you wish to use as a string).
The function takes its fitting parameters information from a .ini file created by the user,
look at the example .ini file mentioned above for the format, you can see that it allows for the parameters to be fixed
or constrained which can be very useful for fitting.

Current functions existing in FittingFunctions.py:

*    Various tunnelling I-V models including BDR, Simmons, Field emission and Tersoff Hamman STM.
*    2D weak localisation
*    Strijkers model for PCAR fitting

Please see the function documentation in :py:mod:`Stoner.FittingFuncs` for more information about these models.
Please do add functions you think would be of use to everybody, have a look at the current functions for examples,
the main thing is that the function must take an x array and a list of parameters, apply a function and then
return the resulting array.

More AnalyseFile Functions
==========================

Applying an arbitary function through the data
----------------------------------------------

:py:meth:`AnalyseFile.apply`::

   a.apply(func, col, replace = True, header = None)

Here *func* is an arbitrary function that will take a complete row in the form of a numpy 1D array, *col*
is the index of a column at which the resulting data is to be inserted or overwrite the
existing data (depending on the values of *replace* and *header*).

Basic Data Inspection
---------------------

:py:meth:`AnalyseFile.max` and :py:meth:`AnalyseFile.min`::

   a.max(column)
   a.min(column)
   a.max(column,bounds=labda x,y:y[2]>1 and y[2]<10)
   a.min(column,bounds=labda x,y:y[2]>1 and y[2]<10)

Hopefully all of the above are fairly obvious ! In the last two cases, one can use a function
to limit the search to particular rows (\eg to search for the maximum y value subject to some constraint
in x). One important point to note is that the routines return a tuple of two numbers, the maximum (or
minimum) and the row number where the maximum or minimum was found.

There are a couple of related functions to help here::

   a.span(column)
   a.span(column, bounds=lambda x,y:y[2]>100)
   a.clip(column,(max_v,min_v)
   a.clip(column,b.span(column))

The :py:meth:`AnalyseFile.span` method simply returns a tuple of minimum and maximum values within either the whole column or
bounded data. Internally this is just calling the :py:meth:`AnalyseFile.max` and :py:meth:`AnalyseFile.min` methods.
The :py:meth:`AnalyseFile.clip` method deletes rows for which the specified column as a value that is either larger or
smaller than the maximum or minimum value within the second argument. This allows one to specify either a tuple --
eg the result of the :py:meth:`AnalyseFile.span` method, or a complete list as in the last example above. Specifying a single
float would have the effect of removing all rows where the column didn't equal the float value. This is probably not a good idea...

It is worth pointing out that these functions will respect the existing mask on the data unless the bounds parameter is set,
in which case the mask is temproarily discarded in favour of one generated from the bounds expression. This can be worked around,
however, as the parameter passed to the bounds function is itself a masked array and thus one can include a test of the mask in the
bounds function::

   a.span(column,bounds=lambda x,y:y[2]>10 or not numpy.any(y.mask))

Data Reduction Methods
======================

:py:class:`AnalyseFile` offers a number of methods to assist in data reduction and data processing.

(Re)Binning Data
----------------

Data binning is the process of taking approximately continuous (x,y) data and grouping them into "bins" of specified x, and average y. Since
this is a data averaging process, the statistical variation in y values is reduced, at the expense of a loss of resolution in x.

:py:class:`AnalyseFile` provides a simple :py:meth:`AnalyseFile.bin` method that can re-bin data::

   (x_bin,y_bin,dy)=a.bin(xcol="Q",ycol="Counts",bins=100,mode="lin")
   (x_bin,y_bin,dy)=a.bin(xcol="Q",ycol="Counts",bins=0.02,mode="log",yerr="dCounts")
   (x_bin,y_bin,dy)=a.bin(mode="log",bins=0.02,bin_start=0.001,bin_stop=0.1)

The mode parameter controls whether linear binning or logarithmic binning is used. The bins parameter is either an integer
in which case it specifies the number of bins to be used, or a float in which case it specifies the bin width. For logarithmic binning
the bin width for bin n is defined as :math:`x_n * w` with :math:`w` being the bin width parameter. Thus the bin boundaries are at
:math:`x_n` and :math:`x_{n+1}=x_n(1+w)` whilst the bin centre is at :math:`x_n(1+\frac{w}{2})`. If a number of bins is specified in logarithmic mode
then the bin boundaries are set from equal logspace boundaries.

If the keyword **yerr** is supplied then the y values in each bin are weighted by their respective error bars when calculating the mean. In this case the
error bar on the bin bceomes the quadrature sum of the individual error bars on each point included in the bin. If **yerr** is nopt specified, then the error bar
is the standard error in the y points in the bin and the y value is the simple mean of the data.

If **xcol** and/or **ycol** are not specified, then they are looked up from the :py:attr:`Stoner.Core.DataFile.setas` attribute. In this case, the **yerr**
is also taken from this attribute if not specified spearately.

Three 1D numpy arrays are returned, representing the x, y and y-errors for the new bins.

Stitching Datasets together
---------------------------

Sometimes a data set is obtained by joing together several separate sets of data,
for example, joinging several scans over different angular or energy ranges to make a single
combinaed scan. In the ideal world, these scans could simple be joing together (e.g. by using the +
operator), in practise one often finds that there are systematic changes in scaling or offsets
between individual scans. The task of stitching data sets together then becomes one of finding the
best mapping between two sets of (x,y) points that are nominally the same. :py:class:`AnalyseFile` provides
a :py:meth:`AnalyseFile.stitch` method to facilitate this.::

    scan1=AnalyseFile('file 1.txt')
    scan2=AnalyseFile('file 2.txt')
    scan2.stitch(scan1,xcol="Angle",ycol="Counts")
    scan1.setas="xy"
    scan2.setas="xy"
    scan2.stitch(scan1)
    scan2.stitch(scan1,mode="shift x")
    scan2.stitch(scan1,mode="scale 7, shift x",overlap=(3.0,5.0))
    scan2.stitch(scan1,func=my_stitcher,p0=[3.14,2.54,2.0])

In these examples, the x and y columns of *scan2* are modified to give the best
possible match to *scan1*. If *xcol* or *ycol* are not given then the default x and y columns as set py the
:py:attr:`Stoner.Core.DataFile.setas` attribute.

The *mode* keyword can be used to specify the types of scaling operations that are
to be allowed. Teh defaults are to shoft both x and y by an offset value and to rescale
the y data by some factor. If even more control of the transformation is needed,
the *func* keyword and *p0* keyword can be used to provide an alternative transformation
function and initial guesses to its parameters. The profotype for the transformation
function should be::

    def my_stitcher(x,y,p1,p2,p3...,pn):
        ...
        return (mapped_x,mapped_y)

In addition to changing the X and Y data in the current :py:class:`AnalyseFile`
instance, two new metadata keys, *Stitching Coefficient* and *Stitching Coeffient Errors*,
with the co-efficients used to modify the scan data.

Thresholding and Interpolating Data
-----------------------------------

Thresholding data is the process of identifying points where y data values cross a given limit (equivalently, finding roots for
:math:`y=f(x)-y_{threshold}`). This is carried out by the :py:meth:`AnalyseFile.threshold` method::

   a.threshold(threshold, col="Y-data", rising=True, falling=False,all_vals=False,xcol="X-data")
   a.threshold(threshold)

If the parameters **col** and **xcol** are not given, they are determined by the :py:attr:`Stoner.Core.DataFile.setas` attribute. The **rising** and **falling**
parameters control whether the y values are rising or falling with row number as they pass the threshold and **all_vals** determines whether the method returns
just the first threshold or all thresholds it can find. The values returned are mapped to the x-column data if it is specified. The thresholding uses just a simple
two point linear fit to find the thresholds.

Interpolating data finds values of y for points x that lie between data points. The :py:meth:`AnalyseFile.interpolate` provides a simple pass-through to the
scipy routine :py:func:`scipy.optimize.interp1d`::

   a.interpolate(newX,kind='linear', xcol="X-Data")

The new values of X are set from the mandetory first argument. **kind** can be either "linear" or "cubic" whilst the xcol data can be omitted in which case the
:py:attr:`Stoner.Core.DataFile.setas` attribute is used. The method will return a new set of data where all columns are interpolated against the new values of X.

Smoothing and Differentiating Data
-----------------------------------

Smoothing and numerical differentation are carried out using the Savitsky-Golay filtering function.
Whilst not the most sophisticated algorithm it is reasonably easy to implement and simple to use.::

    SG_Filter(col=("X"."Y"), points=15, poly=1, order=0,result=None, replace=False, header=None)

If col is a tuple then it is taken to be specifing both x and y data column indices. Otherwise the data
indexed by col is differentiated with repsect to the row. *order* specifies the order of differentiation, where 0
means simply smoothing the data. The algorithm works by locally fitting a polynomial over a certain window of points.
The parameters for this fitting are controlled by the *points* and *poly* parameters. *points*>*poly*>*order* for
the algorithm to work. *resul;t*, *replace* and *header* specify that the calculated data should also be added to
the :py:class:`AnalyseFile` instance, optionally replacing an existing column indexed by *result* and given a new header
*header*. The nature of the local fitting means that the first and last *poly*/2 points are not valid.

Peak Finding
------------

Peak finding is a tricky and often required task in experimental data analysis. When a functional form is known,
it is possible to fit the data to this functional form. However, often a more numerical approach is required.
The :py:meth:`AnalyseFile.peaks` provides a relatively simple and effective method for doing this::

    peak_data=peaks()
    peak_data=peaks(ycol="Y Data", width=0.15, significance=0.001 , xcol="X Data", peaks=True, troughs=False, poly=2,  sort=True)

The *xcol* and *ycol* arguments index the columns with the relevant data. If *xcol* is not provided then the row number is used instead.
If *ycol* is not provided then both x and y data is taken from the *setas* attribute.

The algorithm used is to differentiate the data with a Savitsky-Golay filter - which in effect fits a polynomial locally over the data.
Zero crossing values in the derivative are located and then the second derivative is found for these points and are used to identify
peaks and troughs in the data. The *width* and *poly* keywords are used to control the order of polynomial and the width of the window
used for calculating the derivative - a lower order of polynomial and wider width will make the algroithm less sensitive to narrow peaks.
Only peaks and troughs whose second derivatives are larger than *significance* are returned, and if *sort* is True, the peaks are returned
in order of their significance. *peaks* and *troughs* select whether to return peaks or troughs.

The default values of width and saignificance are set on the assumption that a data set will have less than about 10 peaks of more or
less equal significance. By default, only peaks are returned in order of x position.