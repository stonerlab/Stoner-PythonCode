.. image:: https://travis-ci.org/gb119/Stoner-PythonCode.svg?branch=master
   :target: https://travis-ci.org/gb119/Stoner-PythonCode

.. image:: https://coveralls.io/repos/github/gb119/Stoner-PythonCode/badge.svg?branch=master
   :target: https://coveralls.io/github/gb119/Stoner-PythonCode?branch=master
    
.. image:: https://landscape.io/github/gb119/Stoner-PythonCode/master/landscape.svg?style=flat
   :target: https://landscape.io/github/gb119/Stoner-PythonCode/master
   :alt: Code Health

.. image:: https://badge.fury.io/py/Stoner.svg
   :target: https://badge.fury.io/py/Stoner

.. image:: https://readthedocs.org/projects/stoner-pythoncode/badge/?version=latest
   :target: http://stoner-pythoncode.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://zenodo.org/badge/17265/gb119/Stoner-PythonCode.svg
   :target: https://zenodo.org/badge/latestdoi/17265/gb119/Stoner-PythonCode

.. image:: http://depsy.org/api/package/pypi/Stoner/badge.svg
   :target: http://depsy.org/package/python/Stoner

Introduction
============


The  *Stoner* Python package is a set of utility classes for writing data analysis code. It was written within
the Condensed Matter Physics group at the University of Leeds as a shared resource for quickly writing simple
programs to do things like fitting functions to data, extract curve parameters, churn through large numbers of
small text data files and work with certain types of scientific image files.

For a general introduction, users are referred to the Users Guide, which is part of the `online documentation`_ along with the
API Reference guide. The `github repository`_ also contains some example scripts.

Getting this Code
==================

The *Stoner* package requires numpy >=1.8, scipy >=0.14, matplotlib >=1.5, h5py, lmfit, and has a number of optional dependencies on blist, filemagic, npTDMS 
and numba.

Ananconda Python (and probably other scientific Python distributions) include nearly all of the dependencies, aprt from lmfit.
However, this can by installed with the usual tools such as *easy_install* or *pip*.

.. code-block:: sh

   pip install lmfit

The easiest way to install the Stoner package is via seuptools' easy_install

.. code-block:: sh

   pip install Stoner

This will install the Stoner package and any missing dependencies into your current Python environment. Since the package is under fairly
constant updates, you might want to follow the development with git. The source code, along with example scripts
and some sample data files can be obtained from the github repository: https://github.com/gb119/Stoner-PythonCode

The codebase is compatible with Python 2.7 and Python 3.5+, at present we still develop primarily in Python 2.7 but test with 3.5 and 3.6 as well.

Overview
========
The main part of the **Stoner** package provides two basic top-level classes that describe an individual file of experimental data and a
list (such as a directory tree on disc) of many experimental files. For our research, a typical single experimental data file
is essentially a single 2D table of floating point numbers with associated metadata, usually saved in some
ASCII text format. This seems to cover most experiments in the physical sciences, but it you need a more complex
format with more dimensions of data, we suggest you look elsewhere.

Data and Friends
----------------

**Stoner.Core.DataFile** is the base class for representing individual experimental data sets.
It provides basic methods to examine and manipulate data, manage metadata and load and save data files.
It has a large number of sub classes - most of these are in Stoner.FileFormats and are used to handle the loading of specific
file formats. 

There are also two mxin classes designed to work with DataFile to enable additional functionality for writing analysis programs.

*   **Stoner.Analysis.AnalysisMixin** provides additional methods for curve-fitting, differentiating, smoothing and carrying out
    basic calculations on data.

* **Stoner.plot.PlotMixin** provides additional routines for plotting data on 2D or 3D plots.

For rapid development of small scripts, we would recommend the **Stoner.Data** class which is a superclass of the above,
and provides a 'kitchen-sink' one stop shop for most of the package's functionality.

DataFolder
----------

**Stoner.Folders.DataFolder** is a class for assisting with the work of processing lots of files in a common directory
structure. It provides methods to list. filter and group data according to filename patterns or metadata and then to execute
a function on each file or group of files.

The **Stoner.HDF5** module provides some experimental classes to manipulate *DataFile* and *DataFolder* objects within HDF5
format files. These are not a way to handle arbitary HDF5 files - the format is much to complex and flexible to make that
an easy task, rather it is a way to work with large numbers of experimental sets using just a single file which may be less
brutal to your computer's OS than having directory trees with millions of individual files. The module also provides some classes to
support loading some other HDF5 flavoured files into a **DataFile**.

The **Stoner.Zip** module provides a similar set of classes to **Stoner.HDF5** but working with the ubiquitous zip compressed file format.

Image Subpackage
----------------

The **Stoner.Image** package is a new feature of recent versions of the package and provides dedicated classes for working with image data,
and in particular for analysing Kerr Microscope image files. It provides an **ImageFile** class that is functionally similar to **DataFile**
except that the numerical data is understood to represent image data and additional methods are incorporated to facilitate processing.

Resources
==========

Included in the `github repository`_  are a (small) collection of sample scripts
for carrying out various operations and some sample data files for testing the loading and processing of data. There is also a
`User_Guide`_ as part of this documentation, along with a :doc:`complete API reference <Stoner>`

Contact and Licensing
=====================

The lead developer for this code is `Dr Gavin Burnell`_ <g.burnell@leeds.ac.uk>, but many current and former members of the CM Physics group have
contributed code, ideas and bug testing.

The User Guide gives the current list of other contributors to the project.

This code and the sample data are all (C) The University of Leeds 2008-2017 unless otherwise indficated in the source file.
The contents of this package are licensed under the terms of the GNU Public License v3

Recent Changes
==============

Current PyPi Version
--------------------

The current version of the package on PyPi will be the stable branch until the development branch enters beta testing, when we start
making beta packages available.


Development Version
-------------------

The development version will be on version 0.8. Presently nothing has been done on this.

Online documentation for all versions can be found on the ReadTheDocs pages `online documentation`_

Build Status
~~~~~~~~~~~~

Version 0.7 onwards are tested using the Travis-CI services with unit test coverage assessed by Coveralls. We currently test against
python 2.7 and 3.5 via Travis and internally test on Python 3.6. Overall code quality 
is measured by landscape.io. The current status is shown at the top of this readme.

Citing the Stoner Package
~~~~~~~~~~~~~~~~~~~~~~~~~

We maintain a digital object identifier (doi) for this package (linked to on the status bar at the top of this readme) and
encourage any users to cite this package via that doi.

Stable Versions
---------------

The current stable version is 0.7. Features of 0.7 include

    *   Replace older AnalyseFile and PlotFile with mixin based versions AnalysisMixin and PlotMixin
    *   Addition of Stoner.Image package to handle image analysis
    *   Refactor DataFolder to use Mixin classes
    *   DataFolder now defaults to using :py:class:`Stoner.Core.Data`
    *   DataFolder has an options to skip iterating over empty Data files
    *   Further improvements to :py:attr:`Stoner.Core.DataFile.setas` handline.

0.7 will continue to have bug fix releases and back-ports of anything very useful.

The old stable version was 0.6. This features 

    *   Some major changes in the architecture, 
    *   Switching from a numpy MaskedArray as the main data store to a custom sub-class that contains most of the logic 
        for indexing data by column name and designation.
    *   The metadata storage has also been switched to using blist.sortteddict for a fast, alphabetically ordered dictionary storage.
    *   Other underlying changes are a switch to using properties rather than straight attribute access.

0.6 now also makes use of filemagic to work out the mime type of files to be loaded to try and improve the resilience of the automatic
file format detection on platforms where this is supported and adds some extra methods to AnalyseFile for extrapolation.

No further relases will be made to 0.6.

0.6 and 0.7 should work on Python 2.7 and 3.5

.. _online documentation: http://stoner-pythoncode.readthedocs.io/en/latest/
.. _github repository: http://www.github.com/gb119/Stoner-PythonCode/
.. _Dr Gavin Burnell: http://www.stoner.leeds.ac.uk/people/gb
.. _User_Guide: http://stoner-pythoncode.readthedocs.io/en/latest/UserGuide/ugindex.html