Selecting rows and masking measurements
***************************************

.. currentmodule:: Stoner

Use a search when you need values matching a condition. Use a selection when
you need another Data object. Use a mask when you want to exclude measurements
temporarily while keeping their original values.

These examples assume a Data object ``d`` with a column called ``Temperature``;
see :doc:`datafile` for a small example or :doc:`data_loading` to load your own.
Replace the column names and values with those in your measurement.

===============================  ===========================================
Task                             Starting point
===============================  ===========================================
Read matching values             ``d.search("Temperature", 4.2)``
Create a matching Data object     ``d.select(Temperature__ge=10)``
Temporarily exclude values        Set ``d.mask`` or use ``d.filter(...)``
Remove rows permanently           See :doc:`data_editing`
===============================  ===========================================

``search`` and ``select`` leave the original unchanged. Setting a mask or
calling ``filter`` changes the original object's exclusions. The sections below
explain these choices and their boundary rules.

.. contents:: On this page
   :local:
   :depth: 1

Search for matching values
==========================

In many cases you do not know which rows in the data file are of interest - in
this case you want to search the data::

  d.search('Temperature',4.2,accuracy=0.01)
  d.search('Temperature',4.2,['Temperature','Resistance'])
  d.search('Temperature',lambda x,y: x>10 and x<100)
  d.search('Temperature',lambda x,y: x>10 and
                x<1000 and y[1]<1000,['Temperature','Resistance'])

The general form is ``Data.search(<search column>,<search term>[,<listof return columns>])``

The first example will return all the rows where the value of the
*Temperature* column is 4.2. The second example is the same, but only
returns the values from the *Temperature*, and *Resistance*
columns. The rules for selecting the columns are the same as for the
:py:meth:`~Stoner.core.data.Data.column` method -- strings are matched against column headers and
integers select column by number.

The third and fourth examples above demonstrate the use of a function as the
search value. This allows quite complex search criteria to be used. The function
passed to the search routine should take two parameters -- a floating point
number and a numpy array of floating point numbers and should return either
**True** or **False**. The function is evaluated for each row in the
data file and is passed the value corresponding to the search column as the
first parameter while the second parameter contains a list of all of the values
in the row to be returned. If the search function returns True, then the row is
returned, otherwise it isn't. In the last example, the final parameter can
either be a list of columns or a single column. The rules for indexing columns
are the same as used for the :py:meth:`~Stoner.core.data.Data.find_col` method.

The 'accuracy' keyword parameter sets the level of accuracy to accept when testing
equality or ranges (i.e. when the value parameter is a float or a tuple) - this avoids
the problem of rounding errors with floating point arithmetic. The default is accuracy is 0.0.

Numeric NumPy arrays select discrete target values, including arrays with two
entries. Use a tuple for an interval: the lower endpoint is excluded and the upper
endpoint included, after expanding by ``accuracy``. Two-item numeric lists retain
their legacy interval meaning. For example::

    import numpy as np

    d.search('Temperature', np.array([4.2, 77.0]))  # Two discrete targets.
    d.search('Temperature', (4.2, 77.0))            # An interval.
    d.search('Temperature', [True, False])         # The first row only.

Boolean lists and arrays select rows positionally; short selectors are padded
with false. Scalar target comparisons use ``numpy.isclose``, with ``accuracy``
as its absolute tolerance and NumPy's default relative tolerance. Returned masked
arrays retain raw excluded values, dtype and fill value, including when several
columns are requested. Editing a search result does not change the measurement.

Create a Data object from matching rows
=======================================

A very powerful way to get at just the data rows you want is to make use of the :py:meth:`~Stoner.core.data.Data.select` method.
This offers a simple way to query which rows have columns matching some criteria::

 d.select(Temp=250)
 d.select(Temp__ge=150)
 d.select(T1__lt=4,T2__lt=5).select(Res__between=(100,200))

The general form is to provide keyword arguments that are something that can be used to index a column, followed by a double
underscore, followed by an operator. Where more than one keyword argument is supplied, the results of testing each row are logically
ORed. The result of chaining together two separate calls to select will, however, logically AND the two tests. So, in the examples above,
the, first line will assume an implicit equality test and give only those rows with a column *Temp* equal to 250. The second line gives an
explicit greater than or equal to test for the same column. The third line will select first those rows that have column T1 less than 4 *or*
column T2 less than 5 and then from those select those rows which have a column Res between 100 and 200. The full list of operators is given in
:py:meth:`Stoner.core.data.Data.select`.

.. _maskeddata:

Exclude measurements with a mask
================================

Masked data arrays differ from normal data arrays in that they include an option to mask or hide individual data elements.
This can be useful to temporarily discount parts of your data when, for example, fitting a curve or calculating a mean
value or plotting some data. One could, of course, simply ignore the masking option and use the data as is,
however, masking does have a number of practical uses.

The data mask can be accessed via the :py:attr:`~Stoner.core.data.Data.mask` attribute of :py:class:`~Stoner.core.data.Data`::

   import numpy.ma as ma
   print(d.mask)
   d.mask=False
   d.mask=ma.nomask
   d.mask = d.to_numpy(masked=False) < 0  # Mask individual negative values.
   d.mask=lambda x: x[0]<50
   d.mask=lambda x:[y<50 for y in x]

The first line is simply the import statement for the numpy masked arrays in order to get the *nomask*
symbol. The second line will simply print the current mask. The next two examples will unmask all the data
i.e. make the values visible and usable. The next example illustrates using a numpy array of booleans to
set the mask - every element in the mask array that evaluates as a boolean True will be masked and every
False value unmasked. The final two examples illustrate an extension that setting the
:py:class:`~Stoner.core.data.Data` mask attribute allows. If you pass a callable object to the mask attribute it will
be executed, passing each row of the data array to the user supplied function as a numpy array. The user
supplied function can then either return a single boolean value -- in which case it will be used to mask
the entire row -- or a list of boolean values to mask individual cells in the current row.

A masked measurement is hidden rather than deleted. Array exports retain its
mask; text output may use a fill value (often ``1e20``) in place of the hidden
measurement. Do not rely on a text save to preserve both the original value and
its mask.

.. warning::
   This is somewhat dangerous behaviour. Be very careful to remove a mask before saving data if there is any
   chance that you will need the masked data values again later !

.. note::
    ``Data.data`` returns the stored pandas DataFrame of raw values, including
    excluded measurements. ``d.mask`` remains the persistent exclusion interface.
    Use ``d.to_numpy()`` for masked numerical exports or ``d.asarray()`` for a
    detached read-only snapshot. Pandas calculations do not apply exclusions.

Set exclusions with a filtering function
========================================

Sometimes you may want not to get the rows of data that you are looking for as a
separate array, but merely mark them for inclusion (or exclusion) from subsequent
operations. This is where the masked array (see :ref:`maskeddata`) comes into its own.
To select which rows of data have been masked off, use the :py:meth:`~Stoner.core.data.Data.filter` method::

 d.filter(lambda r:r[0]>5)
 d.filter(lambda r:r[0]>5,['Temp'])

With just a single argument, the filter method takes a complete row at a time and passes it
to the first argument, expecting to get a Boolean response (or list of Booleans equal in length
to the number of columns). With a second argument as in the second example, you can specify which
columns are passed to the filtering function in what order. The second argument must be a list
of things which can be used to index a column (ie strings, integers, regular expressions).

Advanced: sections and closest points
=====================================

Another option is to construct a new `Data` object from a section of the data - this is
particularly useful where the `Data` represents data corresponding to a set of (x,y,z)
points. For this case the :py:meth:`~Stoner.core.data.Data.section` method can be used::

    d.setas="x..y.z."
    slab=d.section(x=5.2)
    line=d.section(x=4.7,z=2)
    thick_slab=d.section(z=(5.0,6.0))
    arbitrary=d.section(r=lambda x,y,z:3*x-2*y+z-4==0)

After the x, y, z data columns are identified, the :py:meth:`~Stoner.core.data.Data.section` method works with
'x', 'y' and 'z' keyword arguments which are then used to search for matching data rows (the arguments to
these keyword arguments follow the same rules as the :py:meth:`~Stoner.core.data.Data.search` method).

A final way of searching data is to look for the closest row to a given value. For this the eponymous method may be used::

    r=d.closest(10.3,xcol="Search Col")
    r=d.closest(10.3)

If the *xcol* parameter is not supplied, the value from the :py:attr:`~Stoner.core.data.Data.setas` attribute is used. Since the returned row
is a detached NumPy masked array, its explicitly supplied *i* annotation records the selected row number.

Advanced: find distinct values
==============================

:py:meth:`~Stoner.core.data.Data.unique` returns distinct values through NumPy's
masked-array interface, retaining the selected dtype and fill value. It leaves
the source unchanged and returns detached, writable result arrays::

   d.unique('Temp')
   d.unique(column,return_index=False, return_inverse=False)

   values, first, inverse = d.unique('Temp', return_index=True, return_inverse=True)

``return_index=True`` adds the first occurrence positions;
``return_inverse=True`` adds the mapping from each input element to its result
group. With both options the result is ``(values, first, inverse)``; either option
can be used independently. Column selection follows
:py:meth:`~Stoner.core.data.Data.column`, including first-occurrence resolution of
duplicate names. A scalar column selector gives one-dimensional input; a list
containing one column keeps its two-dimensional shape. With several columns, values are read left to right across each row and then
compared individually. This does not find distinct complete rows. The first-occurrence positions count through that combined sequence; the
inverse result has the same shape as the selection.

Ordinary finite values are sorted. Masked measurements form a masked result
group: their hidden values are not listed as distinct observations and are not
deleted from the original. Do not interpret the hidden number under a masked
result as a measured category. NaNs, complex NaNs and mixed masks/NaNs retain NumPy's masked-array behaviour;
mixed cases can produce multiple masked entries. Stoner keeps those NumPy rules rather than combining all missing values into
one group. Inside an editing block, reads through ``d`` still see the original
values until the block completes successfully.

Pandas ``unique`` preserves encounter order, does not apply Stoner exclusions and
does not supply this index/inverse interface. Use it directly on ``d.data`` when
those raw pandas semantics are wanted. ``Data.unique`` accepts neither ``where`` nor arbitrary pandas keywords.

Next steps
==========

Continue with :doc:`data_editing` to change values, add columns or sort rows.
Return to :doc:`datafile` for the full learning path.
