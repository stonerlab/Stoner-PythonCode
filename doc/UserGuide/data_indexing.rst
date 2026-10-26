Reading rows, cells and slices
******************************

.. currentmodule:: Stoner

Indexing means selecting values using square brackets. The first position is
zero: ``d[0]`` reads the first row, and ``d[0, 1]`` reads its second cell.
A slice such as ``0:3`` includes positions 0, 1 and 2, stopping before position 3.

These examples use the measurement ``d`` from :doc:`datafile` or one loaded with
:doc:`data_loading`. Use :doc:`data_columns` when you need to select columns by
name or assign their roles.

Choose the syntax according to what you need:

===============================  ===========================================
Expression                       Result
===============================  ===========================================
``d[0]``                         First row of numerical values
``d[:3]``                        First three rows of numerical values
``d[:3, "Temperature"]``         First three values in a named column
``d["Sample"]``                  Metadata item called Sample
``d.head(3)``                    Independent Data object for the first rows
===============================  ===========================================

Numerical indexing reads values; a string used alone reads metadata. To read a
whole numerical column, use ``d.column("Temperature")`` instead.

.. contents:: On this page
   :local:
   :depth: 1

Read rows
=========

Rows don't have labels, so are accessed directly by number::

  d[1]
  d[1:4]

The second example uses a slice to pull out more than one row. This syntax also
supports the full slice syntax which allows one to, for example, decimate the
rows, or directly pull out the last few rows in the file.

Read cells and named records
============================

There are a number of other forms of indexing supported for :py:class:`~Stoner.core.data.Data`
objects::

  d[10,0]
  d[0:10,0]
  d[10,'Temp']
  d[0:10,['Voltage','Temp']]

The first variant just returns the data in the 11th row, first column (remember
indexing starts at 0). The second variant returns the first 10 values in the
first column. The third variant demonstrates that columns can be indexed by
string as well as number, and the last variant demonstrates indexing
multiple rows and columns -- in this case the first 10 values of the Voltage and
Temp columns.

You might think of the data as being a list of records, where each column is a
field in the record. Numpy supports this type of structured record view of data
and the :py:class:`~Stoner.core.data.Data` object provides the :py:attr:`~Stoner.core.data.Data.records`
attribute to do this. This property returns a separate structured array. Changes to it do not update
the measurement; duplicate headers are made unique for the record field names::

   d.records

Finally the :py:attr:`~Stoner.core.data.Data.dict_records` attribute does the same thing, but presents the data as an array of dictionaries, where the
keys are the column names and each dictionary represents a single row.

Understand copies and read-only results
=======================================

Indexing, ``column()`` and ``asarray()`` return read-only NumPy masked arrays.
They are separate from the measurement: use them for calculations and inspection,
and assign through ``d[row, column]`` to change the original. Use ``to_numpy()``
when you need a writable array copy.

Some results also have descriptive attributes such as ``i`` (the selected row
number), ``column_headers`` and ``setas``. These describe the selection when it
was made. They do not track later edits, and further NumPy operations may lose
them. Keep using ``d.column_headers`` and ``d.setas`` for the current labels and
column assignments::

    row = d[10]
    print(row.i)  # 10
    temperature = row[d.find_col("Temperature")]
    values = d[:, 0]
    label = d.column_headers[0]

NumPy rows use numerical positions rather than column names. For named record
access, use ``d.records[index]["header"]``. To calculate angles, use ``d.q`` and
``d.p`` or NumPy functions. A single masked cell reads as ``numpy.ma.masked``;
test it with ``numpy.ma.is_masked(value)``.

``d.column_ids`` provides stable column identifiers, useful when displayed names
are repeated. ``d.export_storage()`` preserves the complete state for transfer
to another Data object.

Next steps
==========

Continue with :doc:`data_selecting` to find rows by value and exclude unwanted measurements.
Use :doc:`data_editing` when you want to change the measurement itself.
Return to :doc:`datafile` for the full learning path.
