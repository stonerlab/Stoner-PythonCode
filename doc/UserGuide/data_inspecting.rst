Inspecting measurements and metadata
************************************

.. currentmodule:: Stoner

After loading a measurement, check its size, column names, first few rows and
metadata before analysing it. Metadata means information about the measurement,
such as a sample name, temperature or instrument setting.

The examples use a Data object called ``d``. You can load one as described in
:doc:`data_loading` or use the example in :doc:`datafile`. Substitute column and
metadata names from your own measurement where needed.

Start with these checks::

    print(d.shape)                  # (number of rows, number of columns)
    print(d.column_headers)
    print(d.head())                 # First five rows, with metadata
    print(d.metadata)

.. contents:: On this page
   :local:
   :depth: 1

What a Data object contains
===========================

Each Data object contains a numerical table, column names and additional
information about the measurement. This additional information is called
*metadata*: named values such as the sample name or measurement temperature.
Metadata retains type information to distinguish numbers, text and other
supported values. Inspect these parts with::

  d.data
  d.column_headers
  d.metadata

``d.data`` is a pandas DataFrame containing the raw measurements. Stoner keeps
the mask and column assignments separately; ordinary pandas calculations do
not automatically use them. See :doc:`data_selecting` for masks,
:doc:`data_columns` for column roles, and :doc:`data_interchange` for pandas
and NumPy exports and editing contexts.

Previewing rows
===============

:py:meth:`~Stoner.core.data.Data.head` and :py:meth:`~Stoner.core.data.Data.tail`
return independent Stoner objects containing the first or last five rows by
default. They use pandas positional row selection while retaining raw excluded
measurements, masks, dtype, fill value, duplicate headers, column roles, stable
column IDs and typed metadata. Neither method changes the source::

    first = d.head(3)
    last = d.tail(3)
    without_last_two = d.head(-2)
    without_first_two = d.tail(-2)

Zero selects an empty table with the same columns. Positive counts larger than
the table select all rows; negative counts omit rows from the opposite end.
Counts must be integers rather than ``True`` or ``False``. Row numbering in
the returned object starts at zero. Exclusions remain flags and do not remove rows from a preview.
These methods accept neither ``inplace`` nor ``where``. For a pandas result use
``d.data.head()`` or ``d.data.tail()`` directly; those results follow pandas
copying rules and do not carry Stoner's separate mask or column assignments.

Only the selected rows are copied, along with their masks. Metadata is copied
independently too, so large metadata entries can still make a preview expensive.
Subclasses with their own copying behaviour retain that behaviour.

.. note::
    ``d.data.head()`` and ``d.data.tail()`` instead return pandas results. They
    contain raw measurements and do not carry Stoner's separate mask or column
    assignments. Use ``d.head()`` and ``d.tail()`` to keep that information.

Read and update metadata
========================

Use a string in square brackets to read or set a metadata item::

    d["Sample"] = "Example wire"
    print(d["Sample"])

To get a list of possible keys in the metadata, you can do::

  d.dir()
  d.dir(r'Option\:.*')

In the first case, all of the keys will be returned in a list. In the second,
only keys matching the pattern will be returned -- all keys containing
*Option:*. For compatibility with normal Python semantics: :py:meth:`~Stoner.core.data.Data.keys` is
synonymous with :py:meth:`~Stoner.core.data.Data.dir`.

If the string you supply to get the metadata item does not exactly match an item of
metadata, then it is interpreted as a regular expression to try and match against all the
items of metadata. In this case, rather than returning a single item, all of the
matching metadata is returned as a dictionary. Passing a compiled regular expression
as the item name also has the same effect - this is useful if the regular expression
you want to match is also an exact match to one particular metadata name.

We mentioned above that the metadata also keeps a note of the expected type of
the data. You can get at the metadata type for a particular key like this::

  d.metadata.type('Sample')

to get a dictionary of all of the types associated with each key you could do::

  dict(zip(d.dir(),d.metadata.type(d.dir())))

but an easier way is to use the :py:attr:`Stoner.core.base.TypeHintedDict.types` attribute::

   d.metadata.types

Keep an independent copy
========================

Assigning ``other = d`` gives the same object another name. Editing ``other``
therefore also changes what you see through ``d``. Use
:py:attr:`~Stoner.core.data.Data.clone` for an independent deep copy, including
the measurement's metadata. Python's :py:mod:`copy` module also works with Data
objects.

.. note::
   This is an attribute not a method, so there are no brackets here !

::

    t=d.clone

Control printed output
======================

If the optional *tabulate* package is installed, then a pretty formatted representation of the :py:class:`~Stoner.core.data.Data` can be generated using:

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
    Loaded as{String}= b"'Data'"  291.6          4.7878        5.375103
                                      291.6          4.78748       6.453249
                                      291.6          4.7878        7.515769
                                      291.6          4.78789       8.57829

If more columns exist in the :py:class:`~Stoner.core.data.Data` then the *repr* method attempts to pick 'interesting' columns. The algorithm will prioritise showing columns
that have been assigned a meaning with the :py:attr:`~Stoner.core.data.Data.setas` attribute. If there are space for further columns, then the last column will be shown
and other columns that follow from any that are marked in :py:attr:`~Stoner.core.data.Data.setas`. If no columns are marked as interesting, then the first n-2 columns and
the last column will be shown::

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

The table header lists the column titles, numerical indices for each column and the assignment in the :py:attr:`~Stoner.core.data.Data.setas` attribute.

If the file has more than 256 rows, then the first 128 rows and last 128 rows will be shown with a row of *...* to show the split.

Many methods in the Stoner package return the modified :class:`Stoner.core.data.Data` object and in IPython consoles and Jupyter notebooks
these will get printed out using the table formats above. This may be more than is required, in which case you can set options in the Stoner
package to control the output format::

    from Stoner import Options
    Options.short_data_repr = True # or "short_repr" for short representations of all objects.
    print(repr(d))
    >>> TDI_Format_RT.txt(<class 'Stoner.Data'>) of shape (1676, 3) (xy.) and 6 items of metadata

Next steps
==========

Continue with :doc:`data_columns` to identify your coordinates and uncertainties.
Return to :doc:`datafile` for the full learning path.
