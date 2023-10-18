**************************
Loading and Examining Data
**************************


Loading a data file
====================

.. currentmodule:: Stoner.Core

The first step in using the Stoner module is to load some data from a
measurement.::

   from Stoner import Data
   d=Data('my_data.txt')

In this example we have loaded data from ``my_data.txt`` which should be in the
current directory

The **Stoner.Data** class is actually a shorthand for importing the :py:class:`Stoner.core.data.Data`
class which in turn is a superset of many of the classes in the Stoner package. This includes code to automatically
detect the format of many of the measurement files that we use in our research.

The native file format for the Stoner package is known as the *TDI 1.5* format - a tab delimited text file
that stores arbitrary metadata and a single 2D data set. It closely matches the :py:class:`DataFile` class of the
:py:mod:`Stoner.Core` module.

.. note::
    :py:class:`Data` will also read a related text format where the first column of the first line contains the string
    *TDI Format=Text 1.0* which are produced by some of the LabVIEW rigs used by the Device Materials Group in
    Cambridge.



The Various Flavours of the :py:class:`DataFile` Class
------------------------------------------------------

To support a variety of different input file formats, the Stoner package provides a slew of subclasses of the base
:py:class:`DataFile` class. Each subclass typically provides its own version of the :py:meth:`DataFile._load` method that
understands how to reqad the relevant file.

Base Classes and Generic Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    :py:class:`DataFile`
        Tagged Data Interchange Format 1.5 -- the default format produced by the LabVIEW measurement rigs in the
        CM Physics group in Leeds
    :py:class:`Stoner.formats.generic.CSVFile`
        Reads a generic comma separated value file. The :py:meth:`Stoner.FileFormats.CSVFile.load`
        routine takes four additional parameters to the constructor and load methods. In
        addition to the two extra arguments used for the *BigBlue& variant, a
        further two parameters specify the deliminators for the data and header rows. :py:class:`Stoner.FileFormats.CSVFile`
        also offers a **save** method to allow data to be saved in a simple deliminated text way (see Section :ref:`save` for details).
    :py:class:`Stoner.formats.generic.JustNumbersFile`
        This is a subclass of CSVFile dedicated for reading a text file that consists purely of rows of numbers with no header or metadata.
    :py:class:`Stoner.formats.instruments.SPCFile`
        Loads a Raman scan file (.spc format) produced by the Rensihaw and Horiba
        Raman spectrometers. This may also work for other instruments that produce spc files,
        but has not been extensively tested.
    :py:class:`Stoner.formats.generic.TDMSFile`
        Loads a file saved in the National Instruments TDMS format
    :py:class:`Stoner.formats.sinstruments.QDFile`
        Loads data from various Quantum Design instruments, cincluding PPMS, MPMS and  SQUID VSM.
    :py:class:`Stoner.formats.simulations.OVFFile`
        OVF files are output by a variety of micomagnetics simulators. The standard was designed for the OOMMF code. This class will handle rectangualr mesh files with text or binary formats, versions 1.0 and 2.0

Classes for Specific Instruments (Mainly ones owned by the CM Physics Group in Leeds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    :py:class:`Stoner.formats.instruments.VSMFile`
        The text files produced by the group's Oxford Instruments VSM
    :py:class:`Stoner.formats.rigs.BigBlueFile`
        Datafiles were produced by VB Code running on the Big Blue cryostat. The
        :py:class:`Stoner.FileFormats.BigBlueFile` version of the :py:meth:`DataFile.load` and :py:class:`DataFile`
        constructors takes two additional parameters that specify the row on which the column headers will
        be found and the row on which the data starts. *This class is now largely obsolete as no new data files have
        been produced in this format for more than 5 years.*
    :py:class:`Stoner.forms.instruments.XRDFile`
        Loads a scan file produced by Arkengarthdale - the group's Brucker D8
        XRD Machine.
    :py:class:`Stoner.formats.rig.MokeFile`
        Loads a file from Leeds Condensed Matter Physics MOKE system in it's old vb6-code. Like the BigBlueFile, this
        format is largely obsolete now.
    :py:class:`Stoner.formats.rigs.FmokeFile`
        Loads a file from Dan Allwood's Focussed MOKE System in Sheffield.
    :py:class:`Stoner.forms.instruments.LSTemperatureFile`
        Loads and saves data from Lakeshore inc's .340 Temperature Calibration file format.

Classes for Instruments at Major Facilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    :py:class:`Stoner.formats.facilities.BNLFile`
        Loads a SPEC file from Brookhaven (so far only tested on u4b files but may well work with
        other synchrotron data). Produces metadata Snumber: Scan number, Stype: Type of scan, Sdatetime: date time stamp for the measurement, Smotor: z motor position.
    :py:class:`Stoner.formats.facilities.OpenGDAFile` Reads an ascii  scan file generated by OpenGDA -- a software
        suite used for synchrotron such as Diamond.
    :py:class:`Stoner.formats.facilities.RasorFile`
        Simply an alias for OpenGDAFile used for the RASOR instrument on I10 at Diamond.
    :py:class:`Stoner.formats.facilities.SNSFile`
        Reads the ascii export file format from `QuickNXS <http://sourceforge.net/projects/quicknxs/>`_
        the data reduction software used on the BL-4A Beam Line at the SNS in Oak Ridge.
    :py:class:`Stoner.formats.facilities.MDAASCIIFile`
        This class will read some variants of the output of mda2ascii as used at the APS in Argonne.
    :py:class:`Stoner.HDF5.SLS_STXMFile`
        This reads an HDF file from the Swiss Light Source Pollux beam line as a data file (as opposed to an image)
    :py:class:`Stoner.formats.maximus.MAximusSpectra`
        This reads a .hdr/.xsp spectra file from the Maximus STXM beamline at Bessy.

These classes can be used directly to load data from the appropriate format.::

   import Stoner
   d=Stoner.Data.load("my_data.txt")
   v=Stoner.Data.load("my_VSM_data.fld", filetype=Stoner.formats.instruments.VSMFile)
   c=Stoner.Data.load('data.csv',1,0,',',', filetype=Stoner.formats.generic.CSVFile)

.. note::
    The :py:meth:`Data.load` is a class method, meaning it creates (and returns) a new
    instance of the :py:class:`Data` class. Most of the methods of :py:class:`Data` objects will return a copy of
    the modified instance, allowing a several methods to be chained together into a single operation.

Sometimes you won't know exactly which subclass of :py:class:`Data` is the one
to use. Unfortunately, there is no sure fire way of telling, but :py:meth:`Data.load` will try to do
the best it can and will try all of the subclasses in memory in turn to see if one will
load the file without throwing an error. If this succeeds then the actual type of file that
worked is stored in the metadata of the loaded file.

.. warning::
   The automatic loading assumes that each load routine does sufficient sanity checking that it will
   throw and error if it gets bad data. Whilst one might wish this was always true it relies on
   whoever writes the load method to make sure of this ! If you want to stop the automatic guessing
   from happening use the ``auto_load=False`` keyword in the *load()* method, or provide an explicit *filetype*
   parameter.

You can also specify a *filetype* parameter to the :py:meth:`Data.load` method or directly to the
:py:class:`Stoner.Data` constructor as illustrated below to load a simple text file of un labelled numbers::

    from Stoner import Data
    d=Data("numbers.txt",filetype="JustNumbers",column_headers=["z","I","dI"],setas="xye")

If *filetype* is a string, it can match either the complete name of the subclass to use to load the file, or
part of it.


Loading Data from a string or iterable object
---------------------------------------------

In some circumstances you may have a string representation of a :py:class:`DataFile` object and want to
transform this into a proper :py:class:`DataFile` object. This might be, for example, from transmitting
the data over a network connection or receiving it from another program. In these situations the
*left shift operator* ``<<`` can be used.::

   data=Stoner.Core.DataFile() << string_of_data
   data=Stoner.Core.DataFile() << iterable_object

The second example would allow any object that can be iterated (i.e. has a *next()* method that returns lines
of the data file, to be used as the source of the data. The :py:meth:`DataFile()` creates an empty object so
that the left shift operator calls the method in :py:class:`DataFile` to read the data in. It also
determines the type of the object ``data``. This also provides an alternative syntax for reading a file
from disk::

   data=Stoner.Core.DataFile()<<open("File on Disk.txt")

Constructing :py:class:`DataFile` s from Scratch
------------------------------------------------------------

The constructor :py:class:`DataFile`, :py:meth:`DataFile.__init__` will try its best to guess what your intention
was in constructing a new instance of a DataFile. First of all a constructor function is called based on the number of positional
arguments were passed:

Single Argument Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single argument passed to :py:meth:`DataFile.__init__` is interpreted as follows:

-   A string is assumed to be a filename, and therefore a DataFile is created by loading a file.
-   A 2D numpy array is taken as the numeric data for the new DataFile
-   A list or other iterable of strings is assumed to represent the column headers
-   A list or other iterable of numpy arrays is assumed to represent a sequence of columns
-   A dictionary with string keys and numpy array values of equal length is taken as a set of columns whose
    header labels are the keys of the dictionaries.
-   A *pandas.DataFrame* is used to provide data, column headers and if it has a suitable multi-level column index,
    the :py:attr:`Stoner.Data.setas` attribute.
-   Otherwise a dictionary is treated as the metadata for the new DataFile instance.

Two Argument Constructor
^^^^^^^^^^^^^^^^^^^^^^^^

If the second argument is a dictionary or a list or other iterables of strings, then the first argument interpreted as for
the sibgle argument constructors above, and then the second argument is interpreted in the same way. (Internally this done by calling the
single argument constructor function twice, once for each argument.)

Many Argument Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^

If there are more than two arguments and they are numpy arrays, then they are used as the columns of the new DataFile.

Keyword Arguments in Constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the positional arguments are dealt with, any keywords that match attriobutes of the DataFile are used to set the corresponding
attribute.

Examples
^^^^^^^^

.. code::

    # Load a file from disc, set the setas attribute and column headers
    d=Data("filename.txt",setas="xy", column_headers=["X-Data","Y-Data"])
    # Create a DataFile from a dictionary:
    d=Data({"Temperature":temp_array,"Resistance":res_data})
    # The same, but set metadata too
    d=Data({"Temperature":temp_array,"Resistance":res_data},{"User":"Fred","Sample":"X234_a","Field":2.4})
    # From a pandas DataFrame
    df=pd.DataFrame(...)
    d=Data(df)


Examining and Basic Manipulations of Data
=========================================

Data Structure
--------------

Data, Column headers and metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Having loaded some data, the next stage might be to take a look at it.
Internally, data is represented as a 2D numpy masked array of floating point numbers,
along with a list of column headers and a dictionary that keeps the metadata and
also keeps track of the expected type of the metadata (ie the meta-metadata).
These can be accessed like so::

  d.data
  d.column_headers
  d.metadata

.. image:: https://i.imgur.com/2wBGSYh.png
    :target: https://www.youtube.com/watch?v=D67-VZeg7gc
    :alt: Data, column headers and metadata
    :width: 320



.. _maskeddata:

Masked Data and Why You Care
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Masked data arrays differ from normal data arrays in that they include an option to mask or hide individual data elements.
This can be useful to temporarily discount parts of your data when, for example, fitting a curve or calculating a mean
value or plotting some data. One could, of course, simply ignore the masking option and use the data as is,
however, masking does have a number of practical uses.

The data mask can be accessed via the :py:attr:`DataFile.mask` attribute of :py:class:`DataFile`::

   import numpy.ma as ma
   print d.mask
   d.mask=False
   d.mask=ma.nomask
   d.mask=numpy.array([[True, True, Fale,...False],...,[False,True,...True]])
   d.mask=lambda x: x[0]<50
   d.mask=lambda x:[y<50 for y in x]

The first line is simply the import statement for the numpy masked arrays in order to get the *nomask*
symbol. The second line will simply print the current mask. The next two examples will unmask all the data
i.e. make the values visible and usable. The next example illustrates using a numpy array of booleans to
set the mask - every element in the mask array that evaluates as a boolean True will be masked and every
False value unmasked. So far the semantics here are the same as if one had accessed the mask directly on
the data via ``d.data.mask`` but the final two examples illustrate an extension that setting the
:py:class:`DataFile` mask attribute allows. If you pass a callable object to the mask attribute it will
be executed, passing each row of the data array to the user supplied function as a numpy array. The user
supplied function can then either return a single boolean value -- in which case it will be used to mask
the entire row -- or a list of boolean values to mask individual cells in the current row.

By default when the :py:class:`DataFile` object is printed or saved, data values that have been masked are replaced
with a "fill" value of 10^20.

.. warning::
   This is somewhat dangerous behaviour. Be very careful to remove a mask before saving data if there is any
   chance that you will need the masked data values again later !

.. note::
    Strictgly speaking, the :py:attr:`DataFile.data` attribute is a sub-class of the numpy masked array, :py:class:`DataArray`.
    This works the same way as a masked array, but supports some additional magic indexing and attributes discussed below.

.. _setas:

Marking Columns as Dimensions: the magic *setas* attribute
----------------------------------------------------------

Often in a calculation with some data you will be using one column for 'x' values and one or more 'y' columns
or indeed having 'z' column data and uncertainties in all of these (conventionally we call these 'd', 'e' and 'f' columns
so that 'e' data is the error in the y data). :py:class:`DataFile` has a concept of marking a column as containing such data and
will then use these by default in many methods when appropriate to have 'x' and 'y' data.

In addition to identifying columns as 'x','y', or 'z', for data that describes a vector field, you can mark the columns as containing
'u', 'v', 'w' data where (u,v,w) is the vector value at  the point (x,y,z). There's no support at present for uncertainties in (u,v,w) being marked.

.. image:: https://i.imgur.com/vwBUO25.png
    :target: https://www.youtube.com/watch?v=LbSIqxTD9Xc
    :alt: Data, column headers and metadata
    :width: 320


Setting Column Types
^^^^^^^^^^^^^^^^^^^^

To set which columns contain 'x','y' etc data use the :py:attr:`DataFile.setas` attribute. This attribute can take
a list of single character strings from the set 'x','y','z','d','e', 'f', 'u', 'v', 'w' or '.' where each element of the list refers to
the columns of data in order. To specify that a column has unmarked data use the '.' string. The string '-' can also be used - this
indicats that the current assignment is to be left as is.

Alternately, you can pass :py:attr:`DataFile.setas` a string. In the simplest case, the string is just read in the same way that
the list would have been  - each character corresponds to one column. However, if the string contains an integer, then the next
non-numeric character will be interpreted that many times, so::

    d.setas="3.xy"
    d.setas="...xy"
    d.setas=['.','.','.','x','y']

There are still more ways of setting column types with the :py:attr:`DataFile.setas` attribute::

    d.seetas[3]="x"
    d.setas["x"]=3
    d.setas("3.xy")
    d.setas(['.','.','.','x','y'])
    d.setas(x=3,y=4)

All achieve the same effect of setting the same columns as containing x and y data.

Once you have identified columns for the various types, you also have access to utility attributes to access those columns::

    d.setas="3.xye'
    d.x == d.column(3)
    d.y == d.column(4)
    d.e == d.column(5)

Note that if :py:attr:`DataFile.setas` is not set then attempting to use the quick access column attributes will
result in an exception. Once the :py:attr:`DataFile.setas` attribute is set, a further set of *virtual* or *derived* column attributes
become available.::

    d.setas="xyz"
    d.r # The magnitude of the point x,y,z
    d.q # The angle in the x-y plane relative to the x axis
    d.p # The angle relative to the z axis
    d.setas="xyz..uvw"
    d.r # the magnitude of (u,v,w)
    d.q # The angle in the x-y plane of (u,v) relative to +x
    d.p # The angle of (u,v,w) relatiuve to +z

Where fewer than 3 or 6 dimensions are specified, these virtual columns fallback to working with the appropriate
reduced number dimensions.

There are some more convenience ways to set which columns to use as x,y,z etc.::

    d.setas={"x":0,"y":"Y Column title"}
    d.x=0
    d.y="Y Column title"
   d.setas["Temperature"]="y"

In each of these cases, the :py:class:`DataFile` will try to work out what you intended to achieve for maximum flexibility
and convenience when writing code. However it can be fooled if one of your columns is called 'x' or 'y' !

Reading Column Types
^^^^^^^^^^^^^^^^^^^^

The normal representation of :py:attr:`DataFile.setas` is as a list, but it also has a string conversion available. You can also find which column
has been assigned as 'x', 'y' etc. by treating the :py:attr:`DataFile.setas` as a dictionary::

    d.column_headers=["One","Two","three","Four"]
    d.setas="xy.z"

    print list(d.setas) #['x','y','.','z']
    print d.setas #'xy.z'
    print d.setas['x'] # "One"
    print d.setas["#x"] # 0

Note that the :py:attr:`DataFile.setas` attribute supports reading keys that are either the single letter t get the name of the column or the letter
preceded by a # character to get the number of the column.

Alternatively, and equivalently, you can access the column indexes via attributes of :py:attr:`DataFile.setas`:

    d.setas.has_xcol # True
    d.setas.has_ucol # False
    d.setas.ycol # [1]
    d.setas.xcol # 0

Implied Column Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not specify the column types via the setas attributes, then :py:class:`DataFile` will try to guess sensible columns assignments based on the number
of columns in your data file. These default assignments are only done at the point at which the :py:attr:`DataFile.setas` attribute is consulted. The default
assignments are:

=================  ================
Number of Columns  Assignments
=================  ================
2                  x, y
3                  x, y, e
4                  x, d, y, e
5                  x, y, u, v, w
6                  x, y, z, u, v, w
=================  ================

Swapping and Rotating Column Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, if the :py:attr:`DataFile.setas` attribute has been set with *x*, *y* (and *z*) columns then these assignments can be
swapped around by the **invert** operator **~**. This either swaps *x* and *y* with eir associated errorbars for 2-D datasets, or rotates
*x* to *y*, *y* to *z* and *z* to *x* )again with their associated errors bars.::

    d.setas="xye"
    print d.setas
    >>> ['x','y','e']
    e=~d
    print e.setas
    >>> ['y','x','d']

Printing the Complete :py:class:`DataFile`
------------------------------------------

If the optional *tabulate* package is installed, then a pretty formatted representation of the :py:class:`DataFile` can be generated using:

    print(repr(d))

This will give something like::

    ================================  =============  ============  ===========
    TDI Format 1.5                    Temperature    Resistance     Column 2
    index                                  0              1             2
    ================================  =============  ============  ===========
    Stoner.class{String}= b"'Data'"   291.6          4.7878        0.04687595
    Measurement Title{String}= b"''"  291.6          4.78736       1.125022
    Timestamp{String}= b"''"          291.6          4.78788       2.187542
    User{String}= b"''"               291.6          4.78758       3.250062
    TDI Format{Double Float}= b'1.5'  291.6          4.78782       4.312583
    Loaded as{String}= b"'DataFile'"  291.6          4.7878        5.375103
                                      291.6          4.78748       6.453249
                                      291.6          4.7878        7.515769
                                      291.6          4.78789       8.57829

If more columns exist in the :py:class:`DataFile` then the *repr* method attempts to pick 'interesting' columns. Thealgorithm will prioritise showing columns
that have been assigned a meaning with the :py:attr:`DataFile.setas` attribute. If there are space for further columns, then the last column will be shown
and other columns that follow from any that are marked in :py:attr:`DataFile.setas`. If no columns are marked as interesting, then the first n-2 columns and
the last column will be shown.::

    ====================  =====================  =====================  ====================  ===================  =============  ==============
    TDI Format 1.5        Magnetic Field (Oe)       Moment (emu)        M. Std. Err. (emu)     Transport Action           ....          Map 16
    index                        3 (x)                  4 (y)                    5                     6                                  71
    ====================  =====================  =====================  ====================  ===================  =============  ==============
    Stoner.class{String}  0.990880310535431      -5.83640043200129e-06  8.83351337362955e-09  1.0                                 nan
    = b"'Data'"           0.965473115444183      -5.82915649064088e-06  1.23199616933718e-08  1.0                  ...            nan
    Title,{String}=       1.16873073577881       -5.82160118818014e-06  8.92093033864879e-09  1.0                  ...            nan
    b'None'               1.13061988353729       -5.8201045325249e-06   1.07142611311115e-08  1.0                  ...            nan
    Fileopentime{String}  1.33387744426727       -5.83922456812945e-06  1.02539653165381e-08  1.0                  ...            nan
    = b"'3540392668.062   1.49902403354645       -5.81961870971478e-06  1.04490646832536e-08  1.0                  ...            nan

The table header lists the column titles, numerical indices for each column and the assignment in the :py:attr:`DataFile.setas` attribute.

If the file has more than 256 rowns, then the first 128 rows and last 128 rows will be shown with a row of *...* to show the split.

Many of the methods in the Stoner package return a copy of the current :class:`Stoner.Data` object and in ipython consoles and jupyter notebooks
these will get printed out using the table formats above. This may be more than is required, in which case you can set options in the Stoner
package to control the output format.::

    from Stoner import Options
    Options.short_data_repr = True # or "short_repr" for short representations of all objects.
    print(repr(d))
    >>> TDI_Format_RT.txt(<class 'Stoner.Data'>) of shape (1676, 3) (xy.) and 6 items of metadata


Working with columns of data
-----------------------------

This is all very well, but often you want to examine a particular column of data
or a particular row::

  d.column(0)
  d.column('Temperature')
  d.column(['Temperature',0])

In the first example, the first column of numeric data will be returned. In the
second example, the column headers will first be checked for one labeled exactly
*Temperature* and then if no column is found, the column headers will be
searched using *Temperature* as a regular expression. This would then
match *Temperature (K)* or *Sample Temperature*.  The third
example results in a 2 dimensional numpy array containing two columns in the
order that they appear in the list (ie not the order that they are in the data
file). For completeness, the :py:meth:`DataFile.column` method also allows one to
pass slices to select columns and should do the expected thing.

There are a couple of convenient short cuts. Firstly the *floormod* operator //
is an alias for the :py:meth:`DataFile.column` method and secondly for working
with cases where the column headers are not the same as the names of any of the attributes
of the :py:class:`DataFile` object::

  d//"Temperature"
  d.Temperature
  d.column('Temperature')

all return the same data.

Whenever the Stoner package needs to refer to a column of data, you cn specify it in a number of ways:-

 1) As an integer where the first column on the left is index 0
 2) As a string. if the string matches a column header exactly then the index of that column is returned.
      If the string fails to match any column header it is compiled as a regular expression and then that
      is tried as a match. If multiple columns match then only the first is returned.
 3) As a regular expression directly - this is similar to the case above with a string, but allows you to pass a pre-compiled regular
      expression in directly with any extra options (like case insensitivity flags) set.
 4) As a slice object (ee.g. ``0:10:2``) this is expanded to a list of integers.
 5) As a list of any of the above, in which case the column finding routine is called recursively in turn for each element of the list and
      the final result would be to use a list of column indices.

For example::
    import re
    col=re.compile('^temp',re.IGNORECASE)
    d.column(col)


Working with complete rows of data
----------------------------------

Rows don't have labels, so are accessed directly by number::

  d[1]
  d[1:4]

The second example uses a slice to pull out more than one row. This syntax also
supports the full slice syntax which allows one to, for example, decimate the
rows, or directly pull out the last fews rows in the file.

Special Magic When Working with Subsets of Data
-----------------------------------------------

As mentioned above, the data in a :py:class:`DataFile` is a special siubclass of numpy's Masked Array - :py:class:`DataArray`.
A DataArray understands that columns can have names and can be assigned to hold specific types of data - x,y,z values etc. In
fact, the logic used for the column names and setas attribute in a :py:class:`DataFile` is actually supplied by the
:py:class:`DataArray`. When you index a DataFile or it's data, the resulting data remembers it's column names and assignments
and these can be used directly::

    r=d[1:4]
    print r.x,r.y

In addition to the column assignments, :py:class:`DataArray` also keeps a track of the row numbers and makes them available via
the *i* attribute.::

    d.data.i # [0,1,2,3...,len(d)]
    r=d[10]
    r.i # 10
    r.column_headers

You can reset the row numbers by assigning a value to the *i* attribute.

A single column of data also gains a *.name* attribute that matches its column_header::

    c=d[:,0]
    c.name == c.column_headers[0] #True

Manipulating the metadata
-------------------------

What happens if you use a string and not a number in the above examples ?::

  d['User']

in this case, it is assumed that you meant the metadata with key *User*.
To get a list of possible keys in the metadata, you can do::

  d.dir()
  d.dir('Option\:.*')

In the first case, all of the keys will be returned in a list. In the second,
only keys matching the pattern will be returned -- all keys containing
*Option:*. For compatibility with normal opython semantics: :py:meth:`DataFile.keys` is
synonymous with :py:meth:`DataFile.dir`.

If the string you supply to get the metadata item does not exactly match an item of
metadata, then it is interpreted as a regular expression to try and match against all the
items of metadata. In this case, rather than returning a single item, all of the
matching metadata is returned as a dictionary. PAssing a compiled regular epxression
as the item name also has the same effect - this is useful if the regular expression
you want to match is also an exact match to one particular metadata name.

We mentioned above that the metadata also keeps a note of the expected type of
the data. You can get at the metadata type for a particular key like this::

  d.metadata.type('User')

to get a dictionary of all of the types associated with each key you could do::

  dict(zip(d.dir(),d.metadata.type(d.dir())))

but an easier way would be to use the :py:attr:`typeHintedDict.types` attribute::

   d.metadata.types

More on Indexing the data
-------------------------

There are a number o other forms of indexing supported for :py:class:`DataFile`
objects.::

  d[10,0]
  d[0:10,0]
  d[10,'Temp']
  d[0:10,['Voltage','Temp']

The first variant just returns the data in the 11th row, first column (remember
indexing starts at 0). The second variant returns the first 10 values in the
first column. The third variant demonstrates that columns can be indexed by
string as well as number, and the last variant demonstrates indexing
multiplerows and columns -- in this case the first 10 values of the Voltage and
Temp columns.

You might think of the data as being a list of records, where each column is a
field in the record. Numpy supports this type of structured record view of data
and the :py:class:`DataFile` object provides the :py:attr:`DataFile.records`
attribute to d this. This read-only attribute is just providing an alternative
view of the same data.::

   d.records

Finally the :py:attr:`DataFile.dict_records` atrtibute does the same thing, but presetns the data as an array of dictionaries, where the
keys are the column names and each dictionary represents a single row.

Selecting Individual rows and columns of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many of the function in the Stoner module index columns by searching the column
headings. If one wishes to find the numeric index of a column then the
:py:meth:`DataFile.find_col` method can be used::

   index=d.find_col(1)
   index=d.find_col('Temperature')
   index=d.find_col('Temp.*')
   index=d.find_col('1')
   index=d.find_col(1:10:2)
   index=d.find_col(['Temperature',2,'Resistance'])
   index=d.find_col(re.compile(r"^[A-Z]"))

:py:meth:`DataFile.find_col` takes a number of different forms. If the argument
is an integer then it returns (trivially) the same integer, a string argument is
first checked to see if it exactly matches one of the column headers in which
case the number of the matching column heading is returned. If no exact match is
found then a regular expression search is carried out on the column headings. In
both cases, only the first match is returned. If the string still doesn't match, then
the string is checked to see if it can be cast to an integer, in which case the integer value is used.

The final three examples given above
both return a list of indices, firstly using a slice construct - in this case
the result is trivially the same as the slice itself, and in the second example by
passing a list of column headers to look for. The final example uses a compiled
regular expression. Unlike passing a string which contains a regular expression,
passing a compiled regular expression will return a list of all columns that
match. This distinction allows you to use a unique partial string to match just
one column - but if you really want all possible columns that would match the
pattern, then you can compile the regular expression and pass that instead.

This is the function that is used internally by :py:meth:`DataFile.column`,
:py:meth:`DataFile.search` etc and for this reason the trivial integer and slice
forms are implemented to allow these other functions to work with multiple
columns.

Sometimes you may want to iterate over all of the rows or columns in a data set.
This can be done quite easily::

  for row in d.rows():
  	print row

  for column in d.columns():
  	print column
  ......

If there is no mask set, then the first example could also have been written more compactly as::

  for row in d:
  	print row
  ......

.. note::

    :py:meth:`DataFile.rows` and :py:meth:`DataFile.columns` both take an optional parameter *not_masked*. If this is True then these iterator
    methods will skip over any rows/columns with masked out data values. When iterating over the :py:class:`DataFile` instance directly the
    masked rows are skipped over.

Searching, sectioning and filtering the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Searching
~~~~~~~~~

In many cases you do not know which rows in the data file are of interest - in
this case you want to search the data.::

  d.search('Temperature',4.2,accuracy=0.01)
  d.search('Temperature',4.2,['Temperature','Resistance'])
  d.search('Temperature',lambda x,y: x>10 and x<100)
  d.search('Temperature',lambda x,y: x>10 and
                x<1000 and y[1]<1000,['Temperature','Resistance'])

The general form is ``DataFile.search(<search column>,<search term>[,<listof return columns>])``

The first example will return all the rows where the value of the
*Tenperature* column is 4.2. The second example is the same, but only
returns the values from the *Temperature*, and *Resistance*
columns. The rules for selecting the columns are the same as for the
DataFile.column method above -- strings are matched against column headers and
integers select column by number.

The third and fourth examples above demonstrate the use of a function as the
search value. This allows quite complex search criteria to be used. The function
passed to the search routine should take two parameters -- a floating point
number and a numpy array of floating point numbers and should return either
**True** or **False**. The function is evaluated for each row in the
data file and is passed the value corresponding to the search column as the
first parameter while the second parameter contains a list of all of the values
in the row to be returned. If the search function returns True, then the row is
returned, otherwise it isn't. In thr last example, the final parameter can
either be a list of columns or a single column. The rules for indexing columns
are the same as used for the :py:meth:`DataFile.find_col` method.

The 'accuracy' keyword parameter sets the level of accuracy to accept when testing
equality or ranges (i.e. when the value parameter is a float or a tuple) - this avoids
the problem of rounding errors with floating point arithmetic. The default is accuracy is 0.0.

Filtering
~~~~~~~~~

Sometimes you may want not to get the rows of data that you are looking for as a
separate array, but merely mark them for inclusion (or exclusion) from subsequent
operations. This is where the masked array (see ':ref:`maskeddata`) comes into its own.
To select which rows of data have been masked off, use the :py:meth:`DataFile.filter` method.::

 d.filter(lambda r:r[0]>5)
 d.filter(lambda r:r[0]>5,['Temp'])

With just a single argument, the filter method takes a complete row at a time and passes it
to the first argument, expecting to get a boolean response (or list olf booleans equal in length
to the number of columns). With a second argument as in the second example, you can specify which
columns are passed to the filtering function in what order. The second argument must be a list
of things which can be used to index a column (ie strings, integers, regular expressions).

Selecting
~~~~~~~~~

A very powerful way to get at just the dat rows you want is to make use of the :py:meth:`DataFile.select` method.
This offers a simple way to query which rows have columns matching some criteria.::

 d.select(Temp=250)
 d.select(Temp__ge=150)
 d.select(T1__lt=4,T2__lt=5).select(Res__between=(100,200))

The general form is to provide keyword arguments that are something that can be used to index a column, followed by a double
underscore, followed by and operator. Where more than one keyword argument is supplied, the results of testing each row are logically
ORed. The result of chaining together two separate calls to select will, however, logically AND the two tests. So, in the examples above,
the, first line will assume an implicit equality test and give only those rows with a column *Temp* equal to 250. The second line gives an
explicit greater than or equal to test for the same column. The third line will select first those rows that have column T1 less than 4.2 *or*
column T2 less than 5 and then from those select those rows which have a column Res between 100 and 200. The full list of operators is given in
:py:meth:`Stoner.Folders.baseFolder.select`.

Sectioning
~~~~~~~~~~

Another option is to construct a new `DataFile` object from a section of the data - this is
particularly useful where the `DataFile` represents data correspondi ng to a set of (x,y,z)
points. For this case the :py:,eth:`DataFile.section` method can be used::

    d.setas="x..y.z."
    slab=d.section(x=5.2)
    line=d.section(x=4.7,z=2)
    thick_slab=d.section(z=(5.0,6.0))
    arbitrary=d.section(r=lambda x,y,z:3*x-2*y+z-4==0)

After the x, y, z data columns are identified, the :py:meth:`DataFile.section` method works with
'x', 'y' and 'z' keyword arguments which ar then used to search for matching data rows (the arguments to
these keyword arguments follow the same rules as the :py:meth:`DataFile.search` method).

A final way of searching data is to look for the closest row to a given value. For this the eponymous method may be used::

    r=d.closest(10.3,xcol="Search Col")
    r=d.closest(10.3)

If the *xcol* parameter is not supplied, the value from the :py:attr:`DataFile.setas` attribute is used. Since the returned row
is an instance of the :py:class:`DataArray` that has been taken from the original data, it will know what row number it was and
will make that available via it's *i* attribute.

Find out more about the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another question you might want to ask is, what are all the unique
values of data in a given column (or set of columns). The Python numpy
package has a function to do this and we have a direct pass through
from the DataFile object for this::

   d.unique('Temp')
   d.unique(column,return_index=False, return_inverse=False)

The two optional keywords cause the numpy routine to return the
indices of the unique and all non-unique values in the array. The
column is specified in the same way as the :py:meth:`DataFile.column`
method does.

Copying Data
^^^^^^^^^^^^

One of the characteristics of Python that can confuse those used to other
programming languages is that assignments and argument passing is by reference
and not by value. This can lead to unexpected results as you can end up modifying variables
you were not expecting ! To help with creating genuine copies of data Python provides the copy module.
Whilst this works with DataFile objects, for convenience, the :py:attr:`DataFile.clone` attribute is
provided to make a deep copy of a DataFile object.

.. note::
   This is an attribute not a method, so there are no brackets here !

::

    t=d.clone


Modifying Data
==============

Appending data
--------------

The simplest way to modify some data might be to append some columns or rows.
The Stoner module redefines two standard operators, ``+`` and ``&`` to
have special meanings::

  a=Stoner.DataFile('some_new_data.txt')
  add_rows=d+a
  add_columns=d&a

In these example, *a* is a second DataFile object that contains some
data. In the first example, a new DataFile object is created where the contents
of *a* are added as new rows after the data in *d*. Any metadata
that is in *a* and not in *d* are added to the metadata as well.
There is a requirement, however, that the column headers of *d* and
*a* are the same -- ie that the two DataFile objects appear to represent
similar data.

In the second example, the data in *a* is added as new columns after the
data from *d*. In this case, there is a requirement that the two DataFile
objects have the same number of rows.

These operators are not limited just to DataFile objects, you can also add numpy
arrays to the DataFile object to append additional data.::

  import numpy as np
  x=np.array([1,2,3])
  new_data=d+x
  y=np.array([1,2,3],[11,12,13],[21,22,23],[31,32,33]])
  new_data=d+y
  z={"X":1.0,"Y":2.1,"Z":7.5}
  new_data=d+z
  new_data=d+[x,y,z]
  column=d.column[0]
  new_data=d&column

In the first example above, we add a single row of data to *d*. This
assumes that the number of elements in the array matches the number of columns
in the data file. The second example is similar but this time appends a 2
dimensional numpy array to the data. The third example demonstrates adding data from a dictionary.
In this case the keys of the dictionary are used to determine which column the values are added
to. If their columns that don't match one of the dictionary keys, then a ``NaN`` is inserted. If there
are keys that don't match columns labels, then new columns are added to the data set, filled with ``NaN``.
In the fourth example, each element in the list is added in turn to *d*. A similar effect would be achieved by doing
``new_data=d+x+y+z``.

The last example appends a numpy array as
a column to *d*. In this case the requirement is that the numpy array has
the same or fewer rows of data as *d*.


Working with Columns of Data
-----------------------------

Changing Individual Columns of Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:attr:`DataFile.data` attribute is not simply a 2D numpy array, but a special subclass :py:class:`DataArray`, but still
can be directly modified like any other numpy array like class might be. If, however, the :py:attr:`DataFile.setas` attribute has
been used to identify columns as containing x,y,z,u,v,w,d,e or f type data, then the correspondign attributes can be written
to as well as read to directly modify the data without having to keep track any further of which column(s) is indexed.
Thus the following will work::

    d.setas="x..y..z"
    d.x=d.x-5.0*d.y/d.z
    d.y=d.z**2
    d.z=np.ones(len(d))

When writing to the column attriobutes you must supply a numpy array with the correct number of elements (although DataFile will
try to spot and correct if the array needs to be transposed first). If you specify more than one column has a particular type
then you should supply a 2D array with the corresponding number of columns of data setting the attribute.

In order to preserve the behaviour that allows you to set the column assignments by setting the attribute to an index type, the
:py:class:`DataFile` checks to see if you are setting something that might be a column index or a numpy array. Thus the following
also works::

    d.x="Temp" # Set the Temp column to be x data
    d.x=np.linspace(0,300,len(d)) # And set the column to contain a linear array of values from 0 to 300.

You cannot set the p,q, or r attributes like this as they are calculated on the fly from the cartesian coordinates.
On the otherhand you can do an efficient conversion to polar coordinates with::

    d.setas="xyz"
    (d.x,d.y,d.z)=(d.p,d.q,d.r)

Rearranging Columns of Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is useful to rearrange columns of data. :py:class:`DataFile` offers a couple of methods to help with this.::

   d.swap_column('Resistance','Temperature')
   d.swap_column('Resistance','Temperature',headers_too=False,setas_too=False)
   d.swap_column([(0,1),('Temp','Volt'),(2,'Curr')])
   d.reorder_columns([1,3,'Volt','Temp'])
   d.reorder_columns([1,3,'Volt','Temp'],header_too=False,setas_too=False)

The :py:meth:`DataFile.swap_column` method takes either a either a tuple (or just a pair of arguments) of column names, indices or a list of such
tuples and swaps the columns accordingly, whilst the :py:meth:`DataFile.reorder_columns` method takes a
list of column labels or indices and constructs a new data matrix out of those columns in the new order.
The ``headers_too=False`` options, as the name suggests, cause the column headers not be rearranged.

.. note::
    The addition of the setas_too keyword to swap the column assignments around as well is new in 0.5rc1

Renaming Columns of Data
^^^^^^^^^^^^^^^^^^^^^^^^

As a convenience, :py:class:`DataFile` also offers a useful method to rename data columns::

   d.rename('old_name','new_name')
   d.rename(0,'new_name')

Alternatively,of course, one could just edit the column_headers attribute.

Inserting Columns of Data
^^^^^^^^^^^^^^^^^^^^^^^^^

The append columns operator **&** will only add columns to the end of a
dataset. If you want to add a column of data in the middle of the data set then
you should use the :py:meth:`DataFile.add_column` method.::

  d.add_column(numpy.array(range(100)),header='Column Header')
  d.add_column(numpy.array(range(100)),header='Column Header',index=Index)
  d.add_column(lambda x: x[0]-x[1],header='Column Header',func_args=None)

The first example simply adds a column of data to the end of the dataset and
sets the new column headers. The second variant  inserts the new column before
column *Index*. *Index* follows the same rules as for the
:py:meth:`DataFile.column` method. In the third example, the new column data is
generated by applying the specified function. The function is passed s dingle
row as a 1D numpy array and any of the keyword, argument pairs passed in a
dictionary to the optional *func_args* argument.

The :py:meth:`DataFile.add_column` method returns a copy of the DataFile object
itself as well as modifying the object. This is to allow the method to be chained
up with other methods for more compact code writing.

Deleting Rows of Data
---------------------

Removing complete rows of data is achieved using the :py:meth:`DataFile.del_rows`
method.::

  d.del_rows(10)
  d.del_rows(1:100:2)
  d.del_rows('X Col',value)
  d.del_rows('X Col',lambda x,y:x>300)
  d.del_rows('X Col',(100,200))
  d.del_rows(;X Col',(100,200),invert=True)


The first variant will delete row 10 from the data set (where the first row will
be row 0). You can also supply a list or slice (as in the second example) to
:py:meth:`DataFile.del_rows` to delete multiple rows.

If you do not know in advance which row to delete, then the remiaining
variants provide more advanced options. The third variant searches for and
deletes all rows in which the specified column contains *value*. The
fourth variant selects which ros to delete by calling a user supplied function
for each row. The user supplied function is the same in form and definitition as
that used for the :py:meth:`DataFile.search` method::

    def user_func(x_val,row_as_array):
        return True or False

The final two variants above, use a tuple to select the data. The final example makes
use of the *invert* keyword argument to reverse the sense used to selkect tows. In both cases
rows are deleted(kept for *invert* = True) if the specified column lies between the maximum and minimum
values of the tuple. The test is done inclusively. Any length two iterable object can be used
for specifying the bounds. Finally, if you call :py:meth:`DataFile.del_rows` with no arguments at all, then
it removes all rows where at least one column of data is masked out.::

    d.filter(lambda r:r[0]>50) # Mask all rows where the first column is greater than 50
    d.del_rows() # Then delete them.

For simple caases where the row to be deleted can be expressed as an integer or list of integers,
the subtration operator can be used.::

   e=d-2
   e=d-[1,2,3]
   d-=-1

The final form looks stranger than it is - it simply deletes the last row of data in place.

Deleting Columns of Data
------------------------

Deleting whole columns of data can be done by referring to a column by index or
column header - the indexing rules are the same as used for the
:py:meth:`DataFile.column` method.::

  d.del_column('Temperature')
  d.del_column(1)

Again, there is an operator that can be used to achieve the same effect, in this
case the modulus operator %.::

  e=d%"temperature"
  e=d%1
  d%=-1

Sorting Data
------------

Data can be sorted by one or more columns, specifying the columns as a number or
string for single columns or a list or tuple of strings or numbers for multiple
columns. Currently only ascending sorts are supported.::

  d.sort('Temp')
  d.sort(['Temp','Gate'])


.. _save:

Saving Data
-----------

Only saving data in the *TDI format* and as comma or tab deliminated formats is supported.

.. warning:
   The :py:class:`Stoner.FileFormats.CSVFile` comma or tab eliminated files discard all metadata
   about the measurement. You absolutely must not use this as your primary data format -- always
   keep the *TDI format* files as well.

For example::

  d.save()
  d.save(filename)
  d=Stoner.CSVFile(d)
  d.save()
  d.save(filename,'\t')

In the first case, the filename used tosave the data is determined from the
filename attribute of the DataFile object. This will have been set when the
filewas loaded from disc.

If the filename attribute has not been set eg if the DataFile object was
created from scratch, then the :py:meth:`DataFile.save` method will cause a dialogue
box to be raised so that the user can supply a filename.

In the second variant, the supplied filename is used and the filename attribute
is changed to match this ie ``d.filename`` will always return the last
filename used for a load or save operation.

The third is similar but convert the file to ``cvs`` format while the fourth also
specifies that the eliminator is a tab character.

Exporting Data to pandas
------------------------

The :py:meth:`Stoner.Data.to_pandas` method can be used to convert a :py:class:`Stoner.Data` object to
a *pandas.DataFrame*. The numerical data will be transferred directly, with the DataFrame columns being set up
as a two level index of column headers and column assignments. The Stoner library registers an additional
*metadata* extension attribute for DataFrames that provides thin sub-class wrapper around the same regular expression
based and type hinting dictionary that is used to store metadata in :py:attr:`Stoner.Data.metadata`.

The pandas.DataFrame produced by the :py:meth:`Stoner.Data.to_pandas` method is reversibly convertible back to an identical
:py:class:`Stoner.Data` object by passing the DataFrame into the constructor of the :py:class:`Stoner.Data` object.
