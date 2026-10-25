********************
Analysing Data Files
********************
.. currentmodule:: Stoner.analysis

.. _channel_maths_guide:

Normalising and Basic Maths Operations
======================================

Several methods are provided to assist with common data fitting and preparation tasks, such as normalising columns,
adding and subtracting columns::

   a.normalise('data','reference',header='Normalised Data',replace=True)
   a.normalise(0,1)
   a.normalise(0,3.141592654)
   a.normalise(0,a2.column(0))

The :py:meth:`~Stoner.core.data.Data.normalise` method simply divides the data column by the reference column. By default the normalise method
replaces the data column with the new (normalised) data and appends '(norm)' to the column header. The keyword arguments
*header* and *replace* can override this behaviour. The third variant illustrates normalising to a constant
(note, however, that if the second argument is an integer it is treated as a column index and not a constant).
The final variant takes a 1D array with the same number of elements as rows and uses that to normalise to.
A typical example might be to have some baseline scan that one is normalising to::

   a.subtract('A','B', header="A-B",replace=True)
   a.subtract(0,1)
   a.subtract(0,3.141592654)
   a.subtract(0,a2.column(0))

As one might expect from the name, the :py:meth:`~Stoner.core.data.Data.subtract` method subtracts the second column from the first.
Unlike :py:meth:`~Stoner.core.data.Data.normalise` the first data column will not be replaced but a new column inserted and a new header
(defaulting to 'column header 1 - column header 2') will be created. This can be overridden with the *header* and *replace*
keyword arguments. The next two variants of the :py:meth:`~Stoner.core.data.Data.subtract` method work in an analogous manner to the
:py:meth:`~Stoner.core.data.Data.normalise` methods. Finally the :py:meth:`~Stoner.core.data.Data.add` method allows one to add two columns in a
similar fashion::

   a.add('A','B',header='A plus B',replace=False)
   a.add(0,1)
   a.add(0,3.141592654)
   a.add(0,a2.column(0))

For completeness we also have::

   a.divide('A','B',header='A/B', replace=True)
   a.multiply('A','B',header='A*B', replace=True)

with variants that take either a 1D array of data or a constant instead of the B column index.

The final variant for these channel operations is the :py:meth:`~Stoner.core.data.Data.diffsum` which takes the ratio of the difference over the sum of two channels.
This is typically used to calculate asymmetry parameters::

    a.diffsum('I+','I-')

Of course, these elementary operations might look rather pointless given how easy it is to extract single columns of data and then add them to a
:py:class:`Stoner.core.data.Data` object, however if the channels are specified as a **tuple** of two elements, then it is taken as a channel of data and a
second channel of uncertainties. The uncertainty calculation is then propagated correctly for the maths operations. This is particularly useful for the
:py:meth:`~Stoner.core.data.Data.diffsum` method where the error propagation is not entirely trivial.

.. plot:: samples/channel_math.py
    :include-source:
    :outname: channel_math

Splitting Data Up
=================

One might wish to split a single data file into several different data files each with the rows of the original
that have a common unique value in one data column, or for which some function of the complete row determines which datafile
each row belongs in. The :py:meth:`~Stoner.core.data.Data.split` method is useful for this case::

   a.split('Polarisation')
   a.split('Temperature',lambda x,r:x>100)
   a.split(['Temperature','Polarisation'],[lambda x,r:x>100,None])

In these examples we assume the :py:class:`~Stoner.core.data.Data` has a data column 'Polarisation' that takes two (or more) discrete values
and a column 'Temperature' that contains numbers above and below 100.

The first example would return a :py:class:`~Stoner.folders.mixins.DataFolder` object  containing two separate instances of :py:class:`~Stoner.core.data.Data`  which
would each contain the rows from the original data that had each unique value of the polarisation data. The second example would
produce a :py:class:`~Stoner.folders.mixins.DataFolder` object containing two :py:class:`~Stoner.core.data.Data` objects for the rows with temperature above and below 100.
The final example will result in a :py:class:`~Stoner.folders.mixins.DataFolder` object that has two groups each of which contains
:py:class:`~Stoner.core.data.Data` objects for each polarisation value.

More Stoner.Data Functions
==========================

Applying an arbitrary function through the data
-----------------------------------------------

:py:meth:`~Stoner.core.data.Data.apply`::

   a.apply(func, col, replace = True, header = None)

Here *func* is an arbitrary function that will take a complete row in the form of a numpy 1D array, *col*
is the index of a column at which the resulting data is to be inserted or overwrite the
existing data (depending on the values of *replace* and *header*).

Basic Data Inspection
---------------------

:py:meth:`~Stoner.core.data.Data.max` and :py:meth:`~Stoner.core.data.Data.min`::

   a.max(column)
   a.min(column)
   a.max(column,bounds=lambda x,y:y[2]>1 and y[2]<10)
   a.min(column,bounds=lambda x,y:y[2]>1 and y[2]<10)

``max`` and ``min`` return a pair: the extreme value and its row number.
The optional *bounds* function selects which rows to consider. In the last two
examples, only rows whose third column lies between 1 and 10 are considered.

There are a couple of related functions to help here::

   a.span(column)
   a.span(column, bounds=lambda x,y:y[2]>100)
   a.clip((min_v, max_v), column=column)
   a.clip(b.span(column), column=column)

The :py:meth:`~Stoner.core.data.Data.span` method simply returns a tuple of minimum and maximum values within either the whole column or
bounded data. Internally this is just calling the :py:meth:`~Stoner.core.data.Data.max` and :py:meth:`~Stoner.core.data.Data.min` methods.
The :py:meth:`~Stoner.core.data.Data.clip` method deletes rows whose selected column lies outside
the minimum and maximum of its first argument. Supply a tuple of limits or a sequence of values --
eg the result of the :py:meth:`~Stoner.core.data.Data.span` method, or a complete list as in the last example above. To keep only rows equal to a value, use ``a.select(...)`` instead; ``clip``
expects a sequence of limits.

It is worth pointing out that these functions will respect the existing mask on the data unless the bounds parameter is set,
in which case the mask is temporarily discarded in favour of one generated from the bounds expression. This can be worked around,
however, as the parameter passed to the bounds function is itself a masked array and thus one can include a test of the mask in the
bounds function::

   a.span(column,bounds=lambda x,y:y[2]>10 or not numpy.any(y.mask))

Data Reduction Methods
======================

:py:class:`~Stoner.core.data.Data` offers a number of methods to assist in data reduction and data processing.

.. _binning_guide:

(Re)Binning Data
----------------

Data binning is the process of taking approximately continuous (x,y) data and grouping them into "bins" of specified x, and average y. Since
this is a data averaging process, the statistical variation in y values is reduced, at the expense of a loss of resolution in x. The original measurements are
unchanged unless you explicitly replace them with the binned result.

:py:class:`~Stoner.core.data.Data` provides a simple :py:meth:`~Stoner.core.data.Data.bin` method that can re-bin data::

   (x_bin,y_bin,dy,counts)=a.bin(xcol="Q",ycol="Counts",bins=100,mode="lin",clone=False,yerr=False)
   (x_bin,y_bin,dy,counts)=a.bin(xcol="Q",ycol="Counts",bins=0.02,mode="log",yerr="dCounts",clone=False)
   (x_bin,y_bin,dy,counts)=a.bin(mode="log",bins=0.02,bin_start=0.001,bin_stop=0.1,clone=False)

The mode parameter controls whether linear binning or logarithmic binning is used. The bins parameter is either an integer
in which case it specifies the number of bins to be used, or a float in which case it specifies the bin width. For logarithmic binning
the bin width for bin n is defined as :math:`x_n * w` with :math:`w` being the bin width parameter. Thus the bin boundaries are at
:math:`x_n` and :math:`x_{n+1}=x_n(1+w)` whilst the bin centre is at :math:`x_n(1+\frac{w}{2})`. If a number of bins is specified in logarithmic mode
then the bin boundaries are set from equal logspace boundaries.

If **yerr** is supplied, each bin uses an inverse-variance weighted mean:
points with smaller uncertainties have greater weight. The reported uncertainty
is calculated from the weighted scatter of the points around that mean, divided
by the number of points. It is not the quadrature sum of the input uncertainties.
With no uncertainties, all points have equal weight and the result is a standard
error based on the population variance. Empty bins have NaN means and zero
reported scatter and count; do not interpret zero scatter as a measured precision.

If **xcol** or **ycol** is omitted, the :py:attr:`Stoner.core.data.Data.setas`
assignments supply it. Assigned y uncertainties are also used unless overridden;
pass ``yerr=False`` to request equal weights explicitly.

By default, ``bin`` returns a new Data object and leaves the original unchanged.
The result has a bin-centre column followed by a value, uncertainty and count
column for each selected y column. With ``clone=False``, it instead returns four
arrays: centres, values, uncertainties and counts. Centres are one-dimensional;
the other arrays have one column per selected y column, even for a single trace.

.. plot:: samples/bins.py
    :include-source:
    :outname: bins


.. _smoothing_guide:

Smoothing and Filtering Data
----------------------------

As experimental data normally includes noise in some form, it is useful to be able to filter and smooth data to better see
underlying trends. The Stoner package offers a  number of approaches to filtering data.

    - Smoothing by convoluting with a window

        This is a powerful method of smoothing data by constructing an appropriate length and shape of 'window function' that
        is then convolved with the data so that every point becomes some form of weighted average of surrounding points as
        defined by the window function. This is handled by the :py:meth:`~Stoner.core.data.Data.smooth` method::

            d.smooth("boxcar",size=10)
            d.smooth(("gaussian",1.5),size=0.4,result=True,replace=False,header="Smoothed data")

        In both these examples, the data to be smoothed is determined from the :py:attr:`Stoner.core.data.Data.setas` attribute.
        The first argument is passed to :py:func:`scipy.signal.get_window` to define the window function. The *size* argument
        can either be an integer to specify the number of rows in the window, or a float to specify the size of the window in
        terms of the x data. In the latter case, the data is first reinterpolated to an evenly space set in terms of the x-column
        and then smoothed and then reinterpolated back to the original x data coordinates.

        .. warning::

            This will fail for hysteretic data. In this case it would be better to use an integer size argument and to ensure
            the data is evenly spaced to start with.

        The *result* and *replace* arguments are passed through to :py:meth:`Stoner.core.data.Data.add_column` unless *result*
        is **False** in which case, the smoothed data is passed back as the return value and the numerical data is left unchanged. A floating-point window size also records
        the smoothing window size in the metadata.

    - Savitzky-Golay filtering

        This is a common filtering technique, particularly for spectroscopic data as it is good at keeping major peak locations
        and widths. In essence it is equivalent to least-squares fitting a low order polynomial to a window of the data and using
        the coefficients of the fitting polynomial to determine the smoothed (or differentiated) data. This is implemented as
        :py:meth:`~Stoner.core.data.Data.SG_Filter` method.

    - Spline

        An alternative approach is to use a smoothing spline to fit the data locally. Depending on the spline smoothing setting
        this will create a function that is continuous in both value and derivative that approaches the data. Unlike Savitzky-
        Golay filtering it cannot be used to calculate a derivative easily, but it can handle y data with uncertainties. It is
        implemented as the :py:meth:`~Stoner.core.data.Data.spline` method.

    - Rebinning

        As illustrated above, rebinning the data is a common way for reducing noise by combining several data points. This is simple
        and effective, but does reduce the length of the data !

These approaches are illustrated in the example below:

.. plot:: samples/Smoothing_Data.py
    :include-source:
    :outname: smooth

.. _stitch_guide:

Stitching Datasets together
---------------------------

Measurements may be collected as separate scans covering different angular or
energy ranges. Before combining them, you may need to correct differences in
scale or offset between scans. The :py:meth:`~Stoner.core.data.Data.stitch`
method adjusts one scan to match another over their common range. It finds a
transformation between the two sets of (x, y) points in that overlap.

.. plot:: samples/stitch.py
    :include-source:
    :outname:  stitch

The stitch method can be fine tuned by specifying the possible scaling and shifting operations, overlap
region to use or even a custom stitching transformation function::

    s2.stitch(s1,mode="shift both")
    s2.stitch(s1,mode="scale and shift y",overlap=(3.0,5.0))
    s2.stitch(s1,func=my_stitcher,p0=[3.14,2.54,2.0])

In these examples, the x and y columns of *s2* are modified to give the best
possible match to *s1*. If *xcol* or *ycol* is omitted, the corresponding
:py:attr:`Stoner.core.data.Data.setas` assignment is used.

The *mode* keyword can be used to specify the types of scaling operations that are
to be allowed. The defaults are to shift both x and y by an offset value and to rescale
the y data by some factor. If even more control of the transformation is needed,
the *func* keyword and *p0* keyword can be used to provide an alternative transformation
function and initial guesses to its parameters. The prototype for the transformation
function should be::

    def my_stitcher(x, y, dx, scale, dy):
        return x + dx, scale * y + dy

In addition to changing the X and Y data in the current :py:class:`~Stoner.core.data.Data`
instance, the method records *Stitching Coefficients* and *Stitching Coefficient Errors*
in metadata, together with the overlap range and number of matched points.
It changes the receiving scan and returns that same object; it does not append
the scans into a single table.

Thresholding, Interpolating and Extrapolation of Data
-----------------------------------------------------

Thresholding data is the process of identifying points where y data values cross a given limit (equivalently, finding roots for
:math:`y=f(x)-y_{threshold}`). This is carried out by the :py:meth:`~Stoner.core.data.Data.threshold` method::

   a.threshold(threshold, col="Y-data", rising=True, falling=False,all_vals=False,xcol="X-data")
   a.threshold(threshold)

If the parameters **col** and **xcol** are not given, they are determined by the :py:attr:`Stoner.core.data.Data.setas` attribute. The **rising** and **falling**
parameters control whether the y values are rising or falling with row number as they pass the threshold and **all_vals** determines whether the method returns
just the first threshold or all thresholds it can find. The values returned are mapped to the x-column data if it is specified. Crossing positions are refined with cubic interpolation and numerical root
finding, then mapped to the selected x data. With ``xcol=False``, complete
interpolated rows are returned. If no crossing is found, the default call
returns an empty list.

Interpolating data finds values of y for points x that lie between data points. The :py:meth:`~Stoner.core.data.Data.interpolate` provides a simple pass-through to the
SciPy class :py:class:`scipy.interpolate.interp1d`::

   a.interpolate(newX,kind='linear', xcol="X-Data")

The new values of X are set from the mandatory first argument. **kind** can be either "linear" or "cubic" whilst the xcol data can be omitted in which case the
:py:attr:`Stoner.core.data.Data.setas` attribute is used. The method will return a new set of data where all columns are interpolated against the new values of X.

The :py:meth:`~Stoner.core.data.Data.interpolate` method will return values that are obtained from 'joining the dots' - which is
appropriate if the uncertainties (and hence scatter) in the data is small. With more scatter in the data, it is better to
use some locally fitted spline function to interpolate with. The :py:meth:`~Stoner.core.data.Data.spline` function can be used for this::

    d.spline("X-Data","Y-Data",header="Spline Data",order=3,smoothing=2.0,replace=True)
    d.spline("X-Data","Y-Data",header="Spline Data",order=2,smoothing=2.0,result=True,replace=False)
    new_y=d.spline("X-Data","Y-Data",order=2,smoothing=2.0,result=None)
    spline=d.spline("X-Data","Y-Data",order=2,smoothing=2.0,result=False)

The *order* keyword gives the polynomial order of the spline (1 to 5).
The *smoothing* factor controls how closely it follows the observations;
``smoothing=0.0`` interpolates through the points.

Use *result* to choose what is returned. ``result=None`` returns the fitted y
values without storing them; ``result=False`` returns a callable
:py:class:`scipy.interpolate.UnivariateSpline` for evaluation at other x values.
``result=True`` appends fitted values, while a column selector stores them at
that position. *replace* controls whether an existing destination is overwritten
or a new column inserted. When storing values, the method returns the modified
Data object. Use increasing x values and avoid repeated x values for strict
interpolation.

Extrapolation is, of course, a dangerous, operation when applied to data as it is essentially 'inventing' new data.
Extrapolating from the spline function, whilst possible, is a little tricky and in many cases the :py:meth:`~Stoner.core.data.Data.extrapolate`
method is likely to be more successful. :py:meth:`Stoner.core.data.Data.extrapolate` works by fitting a function over a window in the
data and using the fit function to predict nearby values. Where the new values lie within the range of data, this is strictly
a form of interpolation and the window of data fitted to the extrapolation function is centred around the new x-data point. As
the new x-data point  approaches and passes the limits of the existing data, the window used to provide the fit function
runs up to and stops at the limit of the supplied data. Thus when extrapolating, the new values of data are increasingly less certain
as one moves further from the end of the data.

.. plot:: samples/extrapolate-demo.py
    :include-source:
    :outname: extrapolate


Extrapolation is of course most successful if one has a physical model that should describe the data.
To allow for this, you can pass an arbitrary fitting function as the *kind* parameter.


Smoothing and Differentiating Data
----------------------------------

Smoothing and numerical differentiation are carried out using the Savitzky-Golay filtering function.
Whilst not the most sophisticated algorithm it is reasonably easy to implement and simple to use::

    smoothed = a.SG_Filter(col="Y", xcol="X", points=15, poly=2, order=0)

Use *col* to select the y data and *xcol* to select the x data; omitted
selectors use the column assignments. *order* is the derivative order, with zero
meaning smoothing only. *points* controls the window length and *poly* the fitted
polynomial degree. Choose an odd window longer than the polynomial degree, with
``poly >= order``. Without x data, differentiation is with respect to row number.

By default, the method returns an array rather than storing the result. Set
``result=True`` to append it, or supply a destination column with *result* and
use *replace* to choose insertion or replacement. *header* names stored columns.
The default padding replaces edge results with interior mean values to reduce
edge artefacts. Treat these edge values cautiously; the affected width depends
on the window and derivative order. Use ``pad=False`` to keep the filter's edge
estimates instead.

.. _peak_finding:

Peak Finding
------------

Peak finding is a tricky and often required task in experimental data analysis. When a functional form is known,
it is possible to fit the data to this functional form. However, often a more numerical approach is required.
The :py:meth:`~Stoner.core.data.Data.peaks` provides a relatively simple and effective method for doing this::

    peak_data=a.peaks()
    peak_data=a.peaks(ycol="Y Data", width=0.15, significance=0.001 , xcol="X Data", peaks=True, troughs=False, poly=2,  sort=True)

The *xcol* and *ycol* arguments index the columns with the relevant data. If *xcol* is not provided then the row number is used instead.
If *ycol* is not provided then both x and y data is taken from the *setas* attribute.

The algorithm used is to differentiate the data with a Savitzky-Golay filter - which in effect fits a polynomial locally over the data.
Zero crossing values in the derivative are located and then the second derivative is found for these points and are used to identify
peaks and troughs in the data. The *width* and *poly* keywords are used to control the order of polynomial and the width of the window
used for calculating the derivative - a lower order  of polynomial and wider width will make the algorithm less sensitive to narrow peaks.
The *significance* parameter controls which peaks and troughs are returned. If it
is a float, only features whose absolute second derivative exceeds that value
are returned. If *significance* is an integer, then maximum snd derivative in the data is divided
by the supplied significance and used as the threshold on which peaks and troughs to return. If *significance* is omitted, the maximum absolute second derivative is divided
by twice the working window width.
With ``sort=True``, features are ordered by increasing absolute curvature;
otherwise they follow their position in the data. *peaks* and *troughs* select whether to return peaks or troughs.

The default values of width and significance are set on the assumption that a data set will have less than about 10 peaks of more or
less equal significance. By default, only peaks are returned in order of x position.

The example below shows how to use :py:meth:`~Stoner.core.data.Data.peaks` to filter out just the peak positions in a set of data.

.. plot:: samples/peaks_example.py
    :include-source:
    :outname: peaks

By default, ``peaks`` returns interpolated complete rows at the detected features.
Use ``full_data=False`` for x positions alone. ``modify=True`` instead replaces
the receiving object's rows with the interpolated feature rows and returns that
object; use a clone first if you need the full scan afterwards.
