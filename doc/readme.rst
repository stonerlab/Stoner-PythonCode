.. image:: https://github.com/stonerlab/Stoner-PythonCode/actions/workflows/run-tests-action.yaml/badge.svg?branch=stable
    :target: https://github.com/stonerlab/Stoner-PythonCode/actions/workflows/run-tests-action.yaml

.. image:: https://coveralls.io/repos/github/stonerlab/Stoner-PythonCode/badge.svg?branch=master
    :target: https://coveralls.io/github/stonerlab/Stoner-PythonCode?branch=master

.. image:: https://app.codacy.com/project/badge/Grade/a9069a1567114a22b25d63fd4c50b228
    :target: https://app.codacy.com/gh/stonerlab/Stoner-PythonCode/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade

.. image:: https://badge.fury.io/py/Stoner.svg
   :target: https://badge.fury.io/py/Stoner

.. image:: https://anaconda.org/phygbu/stoner/badges/version.svg
   :target: https://anaconda.org/phygbu/stoner

.. image:: https://readthedocs.org/projects/stoner-pythoncode/badge/?version=latest
   :target: http://stoner-pythoncode.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://zenodo.org/badge/10057055.svg
   :target: https://zenodo.org/badge/latestdoi/10057055


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

The *Stoner* package requires h5py>=2.7.0, lmfit>=0.9.7, matplotlib>=2.0,numpy>=1.13, Pillow>=4.0,
scikit-image>=0.13.0 & scipy>=1.0.0 and also optional depends on  filemagic, npTDMS, imreg_dft and numba, fabio, hyperspy.

Ananconda Python (and probably other scientific Python distributions) include nearly all of the dependencies, and the remaining
dependencies are collected together in the **phygbu** repositry on anaconda cloud. The easiest way to install the Stoner package is,
therefore, to install the most recent Anaconda Python distribution.

Compatibility
--------------

Versions 0.9.x (stable branch) are compatible with Python 2.7, 3.5, 3.6 and 3.7. The latest 0.9.6 version is also compatible with Python 3.8
The current stable verstion (0.10, stable branch) is compatible with Python 3.6-3.9

Conda packages are prepared for the stable branch and when the development branch enters beta testing. Pip wheels are prepared for selected stable releases only.

Installation
------------

After installing the current Anaconda version, open a terminal (Mac/Linux) or Anaconda Prompt (Windows) an do:

.. code-block:: sh

    conda install -c phygbu -c conda-forge Stoner

If (and only if) you are not using Anaconda python, then pip should also work:

.. code-block:: sh

    pip install Stoner

.. warning::
    The conda packages are generally much better tested than the pip wheels, so we would recommend using
    conda where possible.

This will install the Stoner package and any missing dependencies into your current Python environment. Since the package is under fairly
constant updates, you might want to follow the development with git. The source code, along with example scripts
and some sample data files can be obtained from the github repository: https://github.com/stonerlab/Stoner-PythonCode

Overview
========

The main part of the **Stoner** package provides four top-level classes that describe:
    - an individual file of experimental data (**Stoner.Data**) - somewhat similar to a DataFrame,
    - an individual experimental image (**Stoner.ImageFile**),
    - a nested list (such as a directory tree on disc) of many experimental files (**Stoner.DataFolder**)
    - a nested list (such as a directory tree on disc) of many image files (**Stoner.ImageFolder**).

For our research, a typical single experimental data file is essentially a single 2D table of floating point
numbers with associated metadata, usually saved in some ASCII text format. This seems to cover most experiments
in the physical sciences, but it you need a more complex format with more dimensions of data, we suggest
you look elsewhere.

Increasingly we seem also to need process image files and so partnering the experimental measurement file classes,
we have a parallel set of classes for interacting with image data.

The general philosophy used in the package is to work with data by using methods that transform the data in place.
Additionally, many of the analysis methods pass a copy of the data as their return value, allowing a sequence of
operations to be chained together in one line.

This is a *data-centric* approach - we have some data and we do various operations on it to get to our result. In
contrasr, traditional functional programming thinks in terms of various functions into which you pass data.

.. note::
    This is rather similar to pandas DataFrames and the package provides methods to easily convert to and from
    DataFrames. Unlike a DataFrame, a **Stoner.Data** object maintains a dictionary of additional metadata
    attached to the dataset (e.g. of instrument settings, experimental ort environmental; conditions 
    when thedata was taken). To assist with exporting to pandas DataFrames, the package will add a custom
    attrobute handler to pandas DataFrames **DataFrame.metadata** to hold this additional data.
    
    Unlike Pandas, the **Stoner** package's default is to operate in-place and also to return the object
    from method calls to facilitate "chaining" of data methods into short single line pipelines. 

Data and Friends
----------------

**Stoner.Data** is the core class for representing individual experimental data sets.
It is actually composed of several mixin classes that provide different functionality, with methods
to examine and manipulate data, manage metadata, load and save data files, plot results and carry out various analysis tasks.
It has a large number of sub classes - most of these are in Stoner.formats and are used to handle the loading of specific
file formats.

ImageFile
---------

**Stoner.ImageFile** is the top-level class for managing image data. It is the equivalent of **Stoner.Data** and maintains
metadta and comes with a number of methods to manipulate image data. The image data is stored internally as a masked numpy
array and where possible the masking is taken into account when carrying out image analysis tasks. Through some abuse of
the Python class system, functions in the scpy.ndimage and scikit-image modules are mapped into methods of the ImageFile
class allowing a very rich set of operations on the data sets. The default IO methods handle tiff and png images and can
store the metadata of the ImageFile within those file formats.

DataFolder
----------

**Stoner.DataFolder** is a class for assisting with the work of processing lots of files in a common directory
structure. It provides methods to list. filter and group data according to filename patterns or metadata and then to execute
a function on each file or group of files and then collect metadata from each file in turn. A key feature of DataFolder is
its ability to work with the collated metadata from the individual files that are held in the DataFolder.
In combination with its ability to walk through a complete heirarchy of groups of
**Data** objects, the handling of the common metadata provides powerful tools for quickly writing data reduction scripts.

ImageFolder
-----------

**Stoner.ImageFolder** is the equivalent of DataFolder but for images (although technically a DataFolder can contain ImageFile
objects, the ImageFolder class offers additional Image specific functionality). There is a subclass of ImageFolder,
**Stoner.Image.ImageStack** that uses a 3D numpy array as it's primary image store which permits faster access
(at the expense of a larger memory footprint) than the lazy loading ordered dictionary of **ImageFolder**

Other Modules and Classes
-------------------------

The **Stoner.HDF5** module provides some additional classes to manipulate *Data* and *DataFolder* objects within HDF5
format files. HDF5 is a common chouse for storing data from large scale facilties, although providing a way to handle
arbitary HDF5 files is beyond the scope of this package at this time - the format is much too complex and flexible to make that
an easy task. Rather it provides a way to work with large numbers of experimental sets using just a single file which may be less
brutal to your computer's OS than having directory trees with millions of individual files.

The module also provides some classes to support loading some particular HDF5 flavoured files into **Data** and **ImageFile**
objects.

The **Stoner.Zip** module provides a similar set of classes to **Stoner.HDF5** but working with the ubiquitous zip compressed file format.

Resources
==========

Included in the `github repository`_  are a (small) collection of sample scripts
for carrying out various operations and some sample data files for testing the loading and processing of data. There is also a
`User_Guide`_ as part of this documentation, along with a :doc:`complete API reference <Stoner>`

Contact and Licensing
=====================

The lead developer for this code is `Dr Gavin Burnell`_ <g.burnell@leeds.ac.uk>, but many current and former members of
the CM Physics group have contributed code, ideas and bug testing.

The User Guide gives the current list of other contributors to the project.

This code and the sample data are all (C) The University of Leeds 2008-2021 unless otherwise indficated in the source
file. The contents of this package are licensed under the terms of the GNU Public License v3

Recent Changes
==============

Current PyPi Version
--------------------

The current version of the package on PyPi will be the stable branch until the development branch enters beta testing, when we start
making beta packages available.

Development Version
-------------------

The current development version is hosted in the master branch of the repository and will become version 0.11.

At the moment the development version is maily broen....

Build Status
~~~~~~~~~~~~

Version 0.7-0.9 were tested using the Travis-CI services with unit test coverage assessed by Coveralls.

Version 0.9 was tested with Python 2.7, 3.5, 3.6 using the standard unittest module.

Version 0.10 is tested using **pytest** with Python 3.7-3.11 using a github action.


Citing the Stoner Package
~~~~~~~~~~~~~~~~~~~~~~~~~

We maintain a digital object identifier (doi) for this package (linked to on the status bar at the top of this readme) and
encourage any users to cite this package via that doi.

Stable Versions
---------------


New Features in 0.10 include:

    *   Support for Python 3.10 and 3.11
    *   Refactor Stoner.Core.DataFile to move functionality to mixin classes
    *   Start implementing PEP484 Type hinting
    *   Support pathlib for paths
    *   Switch from Tk based dialogs to Qt5 ones
    *   Refactoring the **baseFolder** class so that sub-groups are stored in an attribute that is an instance of a custom
        dictionary with methods to prune and filter in the virtual tree of sub-folders.
    *   Refactoring of the **ImageArray** and **ImageFile** so that binding of external functions as methods is done at
        class definition time rather than at runtime with overly complex __getattr__ methods. The longer term goal is to
        depricate the use of ImageArray in favour of just using ImageFile.
    *   Introduce interactive selection of boxes, lines and mask regions for interactive Matplotlib backends.
    *   Fix some long standing bugs which could lead to shared metadata dictionaries and race conditions

Online documentation for all versions can be found on the ReadTheDocs pages `online documentation`_

Version 0.9 is the old stable version. This is the last version to support Python 2 and 3<3.6. Features of this release are:

    *   Refactoring of the package into a more granual core, plot, formats, folders packages with submodules
    *   Overhaul of the documentation and user guide
    *   Dropping support for the older Stoner.Image.stack.ImageStack class
    *   Droppping support for matplotlib<2.0
    *   Support for Python 3.7 (and 3.8 from 0.9.6)
    *   Unit tests now > 80% coverage across the package.

Version 0.9.8 was the final version of the 0.9 branch

Version 0.8 is the very old stable release. The main new features were:

    *   Reworking of the ImageArray, ImageFile and ImageFolder with many updates and new features.
    *   New mixin based ImageStack2 that can manipulate a large number of images in a 3D numpy array
    *   Continued re-factoring of DataFolder using the mixin approach
    *   Further increases to unit-test coverage, bug fixes and refactoring of some parts of the code.
    *   _setas objects implement a more complete MutableMapping interface and also support +/- operators.
    *   conda packages now being prepared as the preferred package format

0.8.2 was the final release of the 0.8.0 branch

The ancient stable version is 0.7.2. Features of 0.7.2 include

    *   Replace older AnalyseFile and PlotFile with mixin based versions AnalysisMixin and PlotMixin
    *   Addition of Stoner.Image package to handle image analysis
    *   Refactor DataFolder to use Mixin classes
    *   DataFolder now defaults to using :py:class:`Stoner.Core.Data`
    *   DataFolder has an options to skip iterating over empty Data files
    *   Further improvements to :py:attr:`Stoner.Core.DataFile.setas` handline.

No further relases will be made to 0.7.x - 0.9.x

Versions 0.6.x and earlier are now pre-historic!

.. _online documentation: http://stoner-pythoncode.readthedocs.io/en/stable/
.. _github repository: http://www.github.com/stonerlab/Stoner-PythonCode/
.. _Dr Gavin Burnell: http://www.stoner.leeds.ac.uk/people/gb
.. _User_Guide: http://stoner-pythoncode.readthedocs.io/en/latest/UserGuide/ugindex.html
