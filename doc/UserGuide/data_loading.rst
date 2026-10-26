Loading and saving measurements
*******************************

.. currentmodule:: Stoner

Start here when your measurements are in a file. You will learn how to load
a file, choose a format when automatic recognition is not enough, and save a
processed result. If you already have arrays, skip to
`Creating a measurement from arrays`_ below.

Examples use illustrative filenames. Replace these with your own file paths;
a relative path is resolved from Python's current working directory. To try
Stoner without any files, use the small example in :doc:`datafile`.

.. contents:: On this page
   :local:
   :depth: 1

Load a measurement
==================


The first step in using the Stoner module is to load some data from a
measurement::

   from Stoner import Data
   d=Data('my_data.txt')

In this example we have loaded data from ``my_data.txt`` which should be in the
current directory.

``Data`` combines loading, editing, analysis, fitting and plotting in one
object. It recognises many common measurement formats automatically.
``Stoner.Data`` and :py:class:`Stoner.core.data.Data` name the same class;
``from Stoner import Data`` is the usual import.

The native file format for the Stoner package is known as the *TDI 1.5* format - a tab delimited text file
that stores arbitrary metadata and a single 2D data set. It closely matches the :py:class:`~Stoner.core.data.Data` class of the
:py:mod:`Stoner` module.

.. note::
    :py:class:`~Stoner.core.data.Data` will also read a related text format where the first column of the first line contains the string
    *TDI Format=Text 1.0* which are produced by some of the LabVIEW rigs used by the Device Materials Group in
    Cambridge.

Choosing a file format
======================

``Data`` uses registered loader functions to recognise measurement formats.
Normally, supplying a filename is enough. For a known format, give its registered
name with ``filetype``::

    from Stoner import Data

    d = Data.load("my_data.txt")
    v = Data.load("my_VSM_data.fld", filetype="VSMFile")
    c = Data.load("data.csv", filetype="CSVFile", header_line=0, data_line=1,
                  header_delim=",", data_delim=",")
    numbers = Data("numbers.txt", filetype="JustNumbers",
                   column_headers=["z", "I", "dI"], setas="xye")

``Data.load(...)`` creates and returns a new object. Calling ``d.load(...)`` also
returns a new object; it does not replace ``d``. In contrast, many analysis and
editing methods change their existing object and return that same object so
that you can chain calls. Use ``d.clone`` first when you need to keep the original.

Automatic loading tries suitable registered loaders in priority order. The
``"Loaded as"`` metadata entry records the successful loader. Recognition depends
on each loader rejecting files it cannot interpret, so inspect the headers and
values after loading an unfamiliar format. An explicit ``filetype`` selects that
loader without automatic guessing; use the registered name, rather than a
partial class name. ``auto_load=False`` without a ``filetype`` selects the
default TDI loader.

TDI 1.5 is the package's standard tab-delimited text format for a numerical table
and metadata. Other formats may store less information. For CSV layout options
used through initialisation files, see :doc:`fitting_configuration`.

Creating a measurement from arrays
==================================

You can create a Data object from arrays, column names, dictionaries or a pandas
DataFrame. The meaning of the inputs depends on their number and type, as shown
below. Keyword arguments such as ``column_headers`` and ``setas`` make your
intent clearer.

Single Argument Constructor
---------------------------

A single argument passed to :py:meth:`~Stoner.core.data.Data.__init__` is interpreted as follows:

-   A string is assumed to be a filename, and therefore a Data is created by loading a file.
-   A 2D numpy array is taken as the numeric data for the new Data
-   A list or other iterable of strings is assumed to represent the column headers
-   A list or other iterable of numpy arrays is assumed to represent a sequence of columns
-   A dictionary with string keys and numpy array values of equal length is taken as a set of columns whose
    header labels are the keys of the dictionaries.
-   A *pandas.DataFrame* is used to provide data, column headers and if it has a suitable multi-level column index,
    the :py:attr:`~Stoner.core.data.Data.setas` attribute.
-   Otherwise a dictionary is treated as the metadata for the new Data instance.

Two Argument Constructor
------------------------

When the second argument is a dictionary or a sequence of strings, each input
is interpreted using the rules above. For example, pass numerical columns first
and a metadata dictionary second; the examples below show this combination.

Many Argument Constructor
-------------------------

If there are more than two arguments and they are numpy arrays, then they are used as the columns of the new Data.

Keyword Arguments in Constructor
--------------------------------

After the positional arguments are dealt with, any keywords that match attributes of the Data are used to set the corresponding
attribute.

Examples
--------

.. code::

    # Load a file from disc, set the setas attribute and column headers
    d=Data("filename.txt",setas="xy", column_headers=["X-Data","Y-Data"])
    # Create a Data from a dictionary:
    d=Data({"Temperature":temp_array,"Resistance":res_data})
    # The same, but set metadata too
    d=Data({"Temperature":temp_array,"Resistance":res_data},{"User":"Fred","Sample":"X234_a","Field":2.4})
    # From a pandas DataFrame (replace ... with your table values)
    import pandas as pd
    df=pd.DataFrame(...)
    d=Data(df)

.. _save:

Save a measurement
==================

Saving chooses a registered writer from the filename suffix. Use ``as_loaded``
to request a particular registered format::

    d.save()  # Use the current filename, or ask for one if it is unset.
    d.save("processed.tdi", as_loaded="DataFile")
    d.save("processed.csv", as_loaded="CSVFile")
    d.save("processed.txt", as_loaded="CSVFile")

To reopen a known TDI file with its format specified explicitly, use::

    restored = Data.load("processed.tdi", filetype="DataFile")

Saving updates ``d.filename`` and returns ``d``. ``as_loaded=True`` requests the
format recorded in ``"Loaded as"``, where a suitable writer is available.
Not every readable format has a writer. See the format reference for supported
formats and their limitations.

.. warning::
    CSV and plain numerical text omit measurement metadata. Keep the original
    measurement file and use TDI when you need a text file with metadata.
    Text saving may also replace masked measurements with fill values rather
    than preserving the hidden values and mask. Use ``export_storage()`` for a
    lossless transfer between Stoner objects, and retain your original files.

Advanced: read TDI text and streams
===================================

In some circumstances you may have a string representation of a :py:class:`~Stoner.core.data.Data` object and want to
transform this into a proper :py:class:`~Stoner.core.data.Data` object. This might be, for example, from transmitting
the data over a network connection or receiving it from another program. In these situations the
*left shift operator* ``<<`` can be used::

   data=Stoner.Data() << string_of_data
   data=Stoner.Data() << iterable_object

The second example would allow any object that can be iterated (i.e. has a *next()* method that returns lines
of the data file, to be used as the source of the data. The :py:class:`Stoner.core.data.Data` constructor creates an empty object so
that the left shift operator calls the method in :py:class:`~Stoner.core.data.Data` to read the data in. It also
determines the type of the object ``data``. This also provides an alternative syntax for reading a file
from disk::

   with open("File on Disk.txt", encoding="utf-8") as stream:
       data = Stoner.Data() << stream

The operator accepts decoded TDI 1.0, 1.5 and 2.0 text, including text files,
``StringIO`` objects and generators of lines. It consumes streams from their
current position without closing them and returns a new object, leaving the
receiver unchanged. Binary streams must be decoded first.

TDI 2.0 input preserves Python literal metadata types from the measurement
writer, including explicit strings that resemble numbers or booleans. Missing
numerical cells are masked; rows containing only metadata do not add data rows.
The ``<<`` operator and the dedicated ``TDI_2_0`` filename loader share the
parser. Dotted metadata paths and list indices are reconstructed as nested
dictionaries and lists, preserving the existing filename-loader convention.
Metadata may be shorter or longer than the numerical data. Columns may also
have different lengths: shorter columns are padded with masked cells up to
the longest data column, including when one column is entirely empty.

Next steps
==========

Continue with :doc:`data_inspecting` to check the loaded values and metadata.
Return to :doc:`datafile` for the full learning path.
