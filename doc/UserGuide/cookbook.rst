*********
Cookbook
*********

This section gives some short examples to give an idea of things that can be
done with the Stoner python module in just a few lines.

The Util module
===============

.. currentmodule:: Stoner.Util

The **Stoner** package comes with an extra :py:mod:`Stoner.Util` module that includes some handy utility
functions.

The :py:class:`Data` Class
--------------------------

In practice, one often requires both the analysis functions of :py:class:`Stoner.Analysis.AnalysisMixin` and
the plotting functions of :py:class:`Stoner.plot.PlotMixin`. This can be done easily by creating a
subclass that inherits from both, but for convenience, the :py:mod:`Stoner.Util` module provides the :py:class:`Data`
class that does this for you.::

    from Stoner.Core import Data
    d=Data("File-of-data.txt")

Splitting Data into rising and falling values
---------------------------------------------

So far the module just contains one function that will take a single :py:class:`Stoner.Core.DataFile`
object and split it into a series of :py:class:`Stoner.Core.DataFile` objects where one column is either
rising or falling. This is designed to help deal with analysis problems involving hysteretic data.::

    from Stoner.Util import split_up_down
    folder=split_up_down(data,column)

In this example *folder* is a :py:class:`Stoner.Folders.DataFolder` instance with two groups, one for rising values of the column
and one for falling values of the column. The :py:func:`split\_up_down` will take an optional third parameter
which is an existing :py:class:`Stoner.Core.DataFolder` instance to which the new groups (if they
don't already exist) and files will be added.

Analysis of Hysteresis Loops
----------------------------

Since much of our group's work is concerned with measuring magnetic hystersis loops, the :py:func:`hysteresis_correct` function
provides a handy way to correct some instrumental artifacts and measure properties of hysteresis loops.::

    from Stoner.Util import hysteresis_correct
    d=hysteresis_correct('QD-SQUID-VSM.dat',correct_background=False,correct_H=False)
    e=hysteresis_correct(d)

    print "Saturation Magnetisation = {}+/-{}".format(e["Ms"],e["Ms Error"])
    print "Coercive Field = {}".format(e["Hc"])
    print "Remanance = {}".format(e["Remenance"])

The keyword arguments provide options to turn on or off corrections to remove diamagnetic background (and offset in M),
and any offset in H (e.g. due to trapped flux in the magnetometer). The latter option is slightly dangerous aas it will
also remove the effect of any eexhange bias that moves the coercive field. As well as performing the corrections, the code
will add metadata items for:

    * Background susceptibility (from fitting straight lines to the out part of the data)
    * Saturation magnetisation and uncertainty (also from fitting lines to the out part of the data)
    * Coervice Fields (H for zero M)
    * Remenance (M for zero H)
    * Saturation Fields (H where M deviates by the standard error from saturation)
    * Maximum BH product (the point where -H * M is maximum)
    * Loop Area (from integrating the area inside the hysteresis loop - only valid for complete loops)

Some of these parameters are determined by fitting a straight line to the outer portions of the data (i.e. at the
extrema in H). The keyword parameter *saturation_fraction* controls the extent of the data assumed to be saturated.


Formatting Error Values
-----------------------

In experimental physics, the usual practice (unless one has good reason to do otherwise) is to quote uncertainties in
a measurement to one significant figure, and then quote the value to the same number of decimal places. Whilst doing this
might sound simple, actually doing it seems something that many students find difficult. To hep with this task, the :py:mod:`Stoner.Util` module
provides the :py:func:`Stoner.Util.format_error` function.::

    from Stoner.Util import format_error
    from scipy.constants import hbar,m_e,m_u
    print format_error(value,error)
    print format_error(m_e,hbar,latex=True)
    print format_error(hbar/m_e,hbar/m_u,latex=True,mode="eng")
    print format_error(hbar/m_e,hbar/m_u,latex=True,mode="eng",units="Js rads^{-1}kg^{-1})

The simplest form just outputs value+/- error with suitable rounding. Adding the *latex* keyword argument wraps the output in
$...$ and replaces +/- with the equivalent latex math symbol. The *mode* argument can be *eng*,*sci* or *float*(default). The first
uses SI prefixes in presenting the number, the second uses mantissa-exponent notation (or x10^{exponent} in latex mode). The third,
default, option does no particular scaling. Finally the units keyword allows the inclusion of the units for the quantity in the string
- particularly useful in combination with the *eng* mode.


.. _fitting_tricks:

Fitting Tricks
==============

Fitting 3D Data
---------------

:py:meth:`Stoner.Analysis.AnalysisMixin.curve_fit` can also be used to fit 3D (or higher order) data - i.e. where there are two independent
variables. In order to do this, the *xcol* parameter needs to be an iterable (e.g. list or tuple or array), and
the function to be fitted needs to take a tuple of scalars or arrays as the first argument. The following example
illustrates this by fitting a plane to a collection of points in 3D space.

.. plot:: samples/curve_fit_plane.py
    :include-source:
    :outname: curvefit_plane2

Fitting to Minimize a Function
------------------------------

If *ycol* is a numpy array of the same length as the data then the values of ycol are
assumed to be the points to fit rather than index of a column. This is useful if
the function you want to fit can be written as :math:`f(x_1,x_2,\cdots ,x_n)=0`.
In thus case pass *xcol* a list or tuple of columns that make up :math:`x_1,x_2,\cdots ,x_n`
and make your function take a tuple of 1D-arrays and pass *ycol* as an array of zeros.
For example:

.. plot:: samples/sphere_fit.py
    :include-source:
    :outname:  curvefit_sphere_2


Other Recipes
==============

Extract X-Y(Z) from X-Y-Z data
------------------------------

In a number of measurement systems the data is returned as 3 parameters X, Y and
Z and one wishes to extract X-Y as a function of constant Z. For example, *I-V*
sweeps as a function of gate voltage *V:sub:G*. Assuming we have a data file with
columns *Current*, *Voltage*,*Gate*::

   d=DataFile('data.txt')
   t=d
   for gate in d.unique('Gate'):
       t.data=d.search('Gate',gate)
       t.save('Data Gate='+str(gate)+'.txt')

The first line opens the data file containing the *I-V(V_G)* data. The second
creates a temporary copy of the :py:class:`Stoner.Core.DataFile` object - ensuring that we get a copy of
all metadata and column headers. The **for** loop iterates over all unique
values of the data in the gate column and then inside the for loop, searches for
the corresponding *I-V* data, sets it as the data of the temporary DataFile and
then saves it.

Mapping X-Y-Z data to Z(X,Y) data
----------------------------------

In a similar fashion to the previous section, where data has been recorded with
fixed values of *X* and *Y* eg *I* measured for fixed *V* and *V_*, it can be
useful to map the data to a matrix.::

   d=DataFile('Data,.txt')
   t=d
   for gate in d.unique('Gate'):
      t=t+d.search('Gate',gate)[:,d.find_col('Current')]
   t.column_headers=['Bias='+str(x) for x in d.unique('Voltage')]
   t.add_column(d.unique('Gate'),'Gate Voltage',0)

The start of the script follows the previous section, however this time in the
for loop the addition operator is used to add a single row to the temporary
:py:class:`Stoner.Core.DataFile` *t*. In this case we are using the utility method
:py:meth:`Stoner.Core.DataFile.find_col` to find the index of the column with the current
data. After the **for** loop we set the column headers in *t* and then insert
an additional column at the start with the gate voltage values.

The matrix generated by this code is suitable for feeding directly into
:py:meth:`Stoner.plot.PlotMixin.plot_matrix`, however, the same plot could be generated
directly from the :py:meth:`Stoner.plot.PlotMixin.plot_xyz` method too.

Sampling a A 3D Surface function
--------------------------------

Suppose you have a function Z(x,y) that is defined over a range -X0..+X0 and
-Y0..y0 and want to quickly get a view of the surface. This problem can be easily done
by creating a random distribution of (x,y) points, evaluating the function and then
using the :py:class:`PlotFile` methods to generate the surface.::

    from Stoner import Data
    from numpy import uniform

    samples=5000

    surf=Data()
    surf&=uniform(-X0,X0,samples)
    surf&=uniform(-Y0,Y0,samples)
    surf.setas="xy"
    surf&=Z(surf.x,surf.y)
    surf.setas="xyz"
    surf.plot() # or surf.plot(cstride=1,rstride=1,linewidth=0) for finer detail.

Quickly Sectioning a 3D dataset
-------------------------------

We often model the magnetic state in our experiments using a variety of micromagnetic
modelling codes, such OOMMF or MuMax. When modelling a 3D system, it is often useful to
be able to examine a cross-section of the simulation. The Stoner package provides tools
to quickly examine the output data::

    from Stoner.FileFormats import OVFFile # reads OOMMF vector field files
    import Stoner.plot
    p=SP.PlotFile('my_simulation.ovf')
    p.setas="xyzuvw"
    p=p.section(z=10.5) # Take a slice in the xy plane where z is 10.5 nm
    p.plot() # A 3D plot with cones
    p.setas="xy.uvw"
    p.plot() # a 2D colour wheel plot with triangular glyphs showing vector direction.
