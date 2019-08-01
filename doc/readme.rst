.. image:: https://travis-ci.org/stonerlab/Stoner-PythonCode.svg?branch=master
   :target: https://travis-ci.org/stonerlab/Stoner-PythonCode

.. image:: https://coveralls.io/repos/github/stonerlab/Stoner-PythonCode/badge.svg?branch=master
    :target: https://coveralls.io/github/stonerlab/Stoner-PythonCode?branch=master

.. image:: https://api.codacy.com/project/badge/Grade/372f2f9227134fab9c6c2f2ba83962ba
    :target: https://www.codacy.com/app/stonerlab/Stoner-PythonCode?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=stonerlab/Stoner-PythonCode&amp;utm_campaign=Badge_Grade

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

.. image:: https://i.imgur.com/h4mWwM0.png
    :target: https://www.youtube.com/watch?v=uZ_yKs11W18
    :alt: Introduction and Installation Guide to Stoner Pythin Package
    :width: 320

The *Stoner* package requires numpy >=1.8, scipy >=0.14, matplotlib >=1.5, h5py, lmfit,Pillow  and has a number of optional dependencies
on blist, filemagic, npTDMS, imreg_dft and numba.

Ananconda Python (and probably other scientific Python distributions) include nearly all of the dependencies, and the remaining 
dependencies are collected together in the phygbu repositry on anaconda cloud. The easiest way to install the Stoner package is, 
therefore, to install the most recent Anaconda Python distribution (Python 3.7, 3.6, 3.5 or 2.7 should work) and then to install 
the Stoner package via:

.. code-block:: sh

    conda install -c phygbu Stoner

If you are not using Anaconda python, then pip should also work:

.. code-block:: sh

    pip install Stoner

This will install the Stoner package and any missing dependencies into your current Python environment. Since the package is under fairly
constant updates, you might want to follow the development with git. The source code, along with example scripts
and some sample data files can be obtained from the github repository: https://github.com/gb119/Stoner-PythonCode

The codebase is compatible with Python 2.7 and Python 3.5+, at present we still develop primarily in Python 3.6 and 3.7  but test with 
2.7 as well. *NB* Python 3.7 is only supported in version 0.9x onwards and is known to not work with version 0.8.x.

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
except that the numerical data is understood to represent image data and additional methods are incorporated to facilitate processing. The **ImageFolder**
and **ImageStack** classes provide similar functionality to **DataFolder** but with additional methods specific to handling collections of images. **ImageStack**
uses a 3D numpy array as it's primary image store which permits faster access (at the expense of a larger memory footprint) than the lazy loading ordered
dictionary of **ImageFolder**

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

The next development version will be on version 0.9. Features expected in 0.9 include:

    *   Refactoring of the Core module into a more granual core package with submodules
    *   Overhaul of the documentation and user guide
    *   Dropping support for the older Stoner.Image.stack.ImageStack class
    *   Droppping support for matplotlib<2.0
    *   Support for Python 3.7

Online documentation for all versions can be found on the ReadTheDocs pages `online documentation`_

Build Status
~~~~~~~~~~~~

Version 0.7 onwards are tested using the Travis-CI services with unit test coverage assessed by Coveralls. We currently test against
python 2.7 and 3.6 via Travis and internally test on Python 3.5 as well. Overall code quality
is measured by landscape.io. The current status is shown at the top of this readme.

Version 0.9 onwards, we test with Python 3.7, 3.6, 3.5 and 2.7. All versions are probably not compatible with Python 3.8, this will
feature in version 0.10 (probably)

Citing the Stoner Package
~~~~~~~~~~~~~~~~~~~~~~~~~

We maintain a digital object identifier (doi) for this package (linked to on the status bar at the top of this readme) and
encourage any users to cite this package via that doi.

Stable Versions
---------------

Version 0.8 is the current stable release. The main new features are

    *   Reworking of the ImageArray, ImageFile and ImageFolder with many updates and new features.
    *   New mixin based ImageStack2 that can manipulate a large number of images in a 3D numpy array
    *   Continued re-factoring of DataFolder using the mixin approach
    *   Further increases to unit-test coverage, bug fixes and refactoring of some parts of the code.
    *   _setas objects implement a more complete MutableMapping interface and also support +/- operators.
    *   conda packages now being prepared as the preferred package format

0.8.x will continue to have bug fix releases and back-ports of anything very useful until 0.9 is released.

The old stable version is 0.7.2. Features of 0.7.2 include

    *   Replace older AnalyseFile and PlotFile with mixin based versions AnalysisMixin and PlotMixin
    *   Addition of Stoner.Image package to handle image analysis
    *   Refactor DataFolder to use Mixin classes
    *   DataFolder now defaults to using :py:class:`Stoner.Core.Data`
    *   DataFolder has an options to skip iterating over empty Data files
    *   Further improvements to :py:attr:`Stoner.Core.DataFile.setas` handline.

No further relases will be made to 0.7.x.

0.6, 0.7 should work on Python 2.7 and 3.5
0.8 is also tested on Python 3.6

.. _online documentation: http://stoner-pythoncode.readthedocs.io/en/latest/
.. _github repository: http://www.github.com/stonerlab/Stoner-PythonCode/
.. _Dr Gavin Burnell: http://www.stoner.leeds.ac.uk/people/gb
.. _User_Guide: http://stoner-pythoncode.readthedocs.io/en/latest/UserGuide/ugindex.html