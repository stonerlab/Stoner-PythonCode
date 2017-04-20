Introduction
============

The *Stoner* Python package is a set of utility classes for writing data analysis code. It was written within the Condensed Matter Physics group at the University of Leeds as a shared resource for quickly writing simple programs to do things like fitting functions to data, extract curve parameters and churn through large numbers of small text data files.

For a general introduction, users are referred to the Users Guide, which is part of the [online documentation](http://pythonhosted.org/Stoner/) along with the API Reference guide. The [github repository](http://www.github.com/gb119/Stoner-PythonCode/) also contains some example scripts.

Getting this Code
=================

The *Stoner* package requires numpy \>=1.8, scipy \>=0.14, matplotlib \>=1.5, h5py, lmfit, and has a number of optional dependencies on blist, filemagic, npTDMS and numba

Ananconda Python (and probably other scientific Python distributions) include nearly all of the dependencies, aprt from lmfit. However, this can by installed with the usual tools such as *easy\_install* or *pip*.

``` {.sourceCode .sh}
easy_install lmfit
```

The easiest way to install the Stoner package is via seuptools' easy\_install

``` {.sourceCode .sh}
easy_install Stoner
```

This will install the Stoner package and any missing dependencies into your current Python environment. Since the package is under fairly constant updates, you might want to follow the development with git. The source code, along with example scripts and some sample data files can be obtained from the github repository: <https://github.com/gb119/Stoner-PythonCode>

The codebase is largely compatible with Python 3.4, with the expception of the 3D vector map plots which make use of Enthought's *mayavi* package which is still only Python 2 compatible due to the underlying Vtk toolkit. Other issues of broken 3.4 code are bugs to be squashed.

Overview
========

The **Stoner** package provides two basic top-level classes that describe an individual file of experimental data and a list (such as a directory tree on disc) of many experimental files. For our research, a typical single experimental data file is essentially a single 2D table of floating point numbers with associated metadata, usually saved in some ASCII text format. This seems to cover most experiments in the physical sciences, but it you need a more complex format with more dimensions of data, we suggest you look elsewhere.

DataFile and Friends
--------------------

**Stoner.Core.DataFile** is the base class for representing individual experimental data sets. It provides basic methods to examine and manipulate data, manage metadata and load and save data files. It has a large number of sub classes - most of these are in Stoner.FileFormats and are used to handle the loading of specific file formats.

There are also two mxin classes designed to work with DataFile to enable additional functionality for writing analysis programs.

-   **Stoner.Analysis.AnalysisMixin** provides additional methods for curve-fitting, differentiating, smoothing and carrying out  
    basic calculations on data.

-   **Stoner.plot.PlotMixin** provides additional routines for plotting data on 2D or 3D plots.

For rapid development of small scripts, we would recommend the **Stoner.Data** class which is a superclass of the above, and provides a 'kitchen-sink' one stop shop for most of the package's functionality.

DataFolder
----------

**Stoner.Folders.DataFolder** is a class for assisting with the work of processing lots of files in a common directory structure. It provides methods to list. filter and group data according to filename patterns or metadata and then to execute a function on each file or group of files.

The **Stoner.HDF5** module provides some experimental classes to manipulate *DataFile* and *DataFolder* objects within HDF5 format files. These are not a way to handle arbitary HDF5 files - the format is much to complex and flexible to make that an easy task, rather it is a way to work with large numbers of experimental sets using just a single file which may be less brutal to your computer's OS than having directory trees with millions of individual files.

Resources
=========

Included in the [github repository](http://www.github.com/gb119/Stoner-PythonCode/) are a (small) collection of sample scripts for carrying out various operations and some sample data files for testing the loading and processing of data. There is also a User\_Guide as part of this documentation, along with a complete API reference \<Stoner\>

Contact and Licensing
=====================

The lead developer for this code is [Dr Gavin Burnell](http://www.stoner.leeds.ac.uk/people/gb) \<<g.burnell@leeds.ac.uk>\> . The User Guide gives the current list of other contributors to the project.

This code and the sample data are all (C) The University of Leeds 2008-2015 unless otherwise indficated in the source file. The contents of this package are licensed under the terms of the GNU Public License v3

Recent Changes
==============

Current PyPi Version
--------------------

The current version of the package on PyPi will be the stable branch until the development branch enters beta testing, when we start making beta packages available. The current version is:

[![image](https://badge.fury.io/py/Stoner.svg)](https://badge.fury.io/py/Stoner)

Development Version
-------------------

The current development version is 0.7. Features of 0.7 include

> -   Replace older AnalyseFile and PlotFile with mixin based versions AnalysisMixin and PlotMixin
> -   Addition of Stoner.Image package to handle image analysis
> -   Refactor DataFolder to use Mixin classes
> -   DataFolder now defaults to using :pyStoner.Core.Data
> -   DataFolder has an options to skip iterating over empty Data files
> -   Further improvements to :pyStoner.Core.DataFile.setas handline.

Online documentation for the development version can be found on the [githib repository pages](http://gb119.github.io/Stoner-PythonCode)

### Build Status

Travis CI is used to test the development branch to see if it passes the current unit tests and coveralls.io handles the unit test coverage reporting. The current status is:

[![image](https://travis-ci.org/gb119/Stoner-PythonCode.svg?branch=master)](https://travis-ci.org/gb119/Stoner-PythonCode)

[![image](https://coveralls.io/repos/github/gb119/Stoner-PythonCode/badge.svg?branch=master)](https://coveralls.io/github/gb119/Stoner-PythonCode?branch=master)

[![Code Health](https://landscape.io/github/gb119/Stoner-PythonCode/master/landscape.svg?style=flat)](https://landscape.io/github/gb119/Stoner-PythonCode/master)

### Citing the Stoner Package

You can cite the Stoner package via its doi:

[![image](https://zenodo.org/badge/17265/gb119/Stoner-PythonCode.svg)](https://zenodo.org/badge/latestdoi/17265/gb119/Stoner-PythonCode)

Stable Version
--------------

The current stable version is 0.6. This features some major changes in the architecture, switching from a numpy MaskedArray as the main data store to a custom sub-class that contains most of the logic for indexing data by column name and designation. The metadata storage has also been switched to using blist.sortteddict for a fast, alphabetically ordered dictionary storage. Other underlying changes are a switch to using properties rather than straight attribute access.

0.6 now also makes use of filemagic to work out the mime type of files to be loaded to try and improve the resilience of the automatic file format detection on platforms where this is supported and adds some extra methods to AnalyseFile for extrapolation.

0.6 should work on Python 2.7 and 3.5
