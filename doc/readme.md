Introduction
============

The *Stoner* Python package is a set of utility classes for writing data
analysis code. It was written within the Condensed Matter Physics group
at the University of Leeds as a shared resource for quickly writing
simple programs to do things like fitting functions to data, extract
curve parameters and churn through large numbers of small text data
files.

For a general introduction, users are referred to the Users Guide, which
is part of the online documentation
\<http://pythonhosted.org/Stoner/\> along with the API Reference guide.
The github repository
\<http://www.github.com/gb119/Stoner-PythonCode/\> also contains some example scripts.

Getting this Code
==================

The \*Stoner\* package requires numpy \>=1.4, scipy \>=0.12, matplotlib \>=1.1, h5py and lmfit. Experimental code also makes use of
the Enthought Tools Suite packages.

At present it looks like lmfit does not install correctly with easy\_install, so use pip instead

.. code-block:: sh

   pip install lmfit


The easiest way to install this code is via seuptools' easy\_install

.. code-block:: sh

   easy\_install Stoner

This will install the Stoner package into your current Python environment. Since the package is under fairly
constant updates, you might want to follow the development with git. The source code, along with example scripts
and some sample data files can be obtained from the github repository: https://github.com/gb119/Stoner-PythonCode


Overview
========
The \*\*Stoner\*\* package provides two basic top-level classes that describe an individual file of experimental data and a
list (such as a directory tree on disc) of many experimental files. For our research, a typical single experimental data file
is essentially a single 2D table of floating point numbers with associated metadata, usually saved in some
ASCII text format. This seems to cover most experiments in the physical sciences, but it you need a more complex
format with more dimensions of data, we suggest you look elsewhere.

DataFile and Friends
--------------------

\*\*Stoner.Core.DataFile\*\* is the base class for representing individual experimental data sets.
It provides basic methods to examine and manipulate data, manage metadata and load and save data files.
It has a large number of sub classes - most of these are in Stoner.FileFormats and are used to handle the loading of specific
file formats. Two, however, contain additional functionality for writing analysis programs.

\*   \*\*Stoner.Analysis.AnalyseFile\*\* provides additional methods for curve-fitting, differentiating, smoothing and carrying out
        basic calculations on data.

\* \*\*Stoner.Plot.PlotFile\*\* provides additional routines for plotting data on 2D or 3D plots.

As mentioned above, there are subclasses of \*\*DataFile\*\* in the \*\*Stoner.FileFormats\*\* module that support
loading many of the common file formats used in our research.

DataFolder
----------

\*\*Stoner.Folders.DataFolder\*\* is a class for assisting with the work of processing lots of files in a common directory
structure. It provides methods to list. filter and group data according to filename patterns or metadata and then to execute
a function on each file or group of files.

The \*\*Stoner.HDF5\*\* module provides some experimental classes to manipulate \*DataFile\* and \*DataFolder\* objects within HDF5
format files. These are not a way to handle arbitary HDF5 files - the format is much to complex and flexible to make that
an easy task, rather it is a way to work with large numbers of experimental sets using just a single file which may be less
brutal to your computer's OS than having directory trees with millions of individual files.

Resources
==========

Included in the github repository
\<<http://www.github.com/gb119/Stoner-PythonCode/>\> are a (small)
collection of sample scripts for carrying out various operations and
some sample data files for testing the loading and processing of data.
There is also a user guide \<UserGuide/ugindex\> as part of this
documentation, along with a complete API reference \<Stoner\>

Contact and Licensing
=====================

The lead developer for this code is Dr Gavin Burnell
\<<g.burnell@leeds.ac.uk>\> <http://www.stoner.leeds.ac.uk/people/gb>.
The User Guide gives the current list of other contributors to the
project.

This code and the sample data are all (C) The University of Leeds
2008-2015 unless otherwise indficated in the source file. The contents
of this package are licensed under the terms of the GNU Public License
v3

Recent Changes
==============

Version 0.2.5
-------------

Add a MokeFile class for loading Leeds MOKE system files.

Version 0.2.4
-------------

Refactored the setas attribute, improvments to loading some file
formats, new Engineering formatting for plots (optional)

Version 0.2.0
-------------

Added the dependency on lmfit and depricated mpfit for doing bounded
least-squares fitting of complex data functions.
