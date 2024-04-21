*******************************
Installation of Stoner Package
*******************************

Introduction
============

This manual provides a user guide and reference for the Stoner python package.
The Stoner python package provides a set of python classes and functions for
reading, manipulating and plotting data acquired with the lab equipment in the
Condensed Matter Physics Group at the University of Leeds.

Getting the Stoner Package
--------------------------

We recommend the use of Ananconda Python for use with the Stoner package and provide
conda packages for Stoner and its non-standard dependencies. To install the Stoner package
in an Anaconda Python environment, simply do::

    conda install -c phygbu Stoner

If you are not using Ananconda python, then we also make Python wheels available that may be installed
using pip::

    pip install Stoner

The advantage of getting the package this way is that it is installed into your Python path properly.
The disadvantage is that you don't get this user guide and the version may not be the most
up to date (although given the fragile and continuously being broken state of the code that may be
a good thing !).

Getting the Latest Development Code
-----------------------------------


.. note::

   These instructions are for members of the University of Leeds Condensed Matter Physics Group. External users are recommended to
   download the source from GitHub


The source code for the Stoner python module is kept on github using the git
revision control tool. A nightly development release of the code is available for copying and
use in ``\\stonerlab\data\software\python\PythonCode\``.

The Stoner Package currently depends on a number of other modules. These are installed on the lab
machines that have Python installed. Primarily these are Numpy, SciPy and Matplotlib.  The easiest way to get a Python
installation with all the necessary dependencies for the Stoner Package is to install the *Enthought Python Distribution*,
Canopy*. Installers for Windows, MacOS and Linux are kept in ``\\stonerlab\data\software\Python``

The canonical source of the latest version is the master branch of the github archive. If you have git installed on your
machine then::

    pip install git+https://github.com/gb119/Stoner-PythonCode.git

should install the current master branch. Otherwise download the zip file from the github site and do::

    python setup.py install

to insatall it locally.

Using the Development Version of the Stoner Package
===================================================

.. note::
   You only need to follow this section if you are working with the
   version installed from the github repository. If you have installed
   the Stoner Package with the easy_install command given above, then you
   can disregard this section.

The easiest way to use the development version of the Stoner Package is to add the
path to the directory containing the files from github to your PYTHONPATH environment
variable. This can be done on Macs and Linux by doing.

.. code-block:: sh

  cd <path to PythonCode directory>
  export PYTHONPATH=`pwd`:$PYTHONPATH

On a windows machine the easiest way is to create a permanent entry to the
folder in the system environment variables. Go to Control Panel -> System ->
Advanced Tab -> click on Environment button and then add or edit an entry to the
system variable PYTHONPATH.

One this has been done, the Stoner module may be loaded from python command
line::

   import Stoner

or::

   from Stoner import *

Documentation
=============

These pages provide a user guide to the Stoner package and its various modules and classes.
It is not a reference to the library but instead aims to explain the various operations that
are possible and provide short examples of use. For the API reference for the library, please
see the `Module Api Documentation <Stoner>`. There is also a single sided cheat sheet that
summarises the examples in this user guide in the github repository.

.. warning::
   The code is still under active development to fix bugs and add features. Generally things don't
   get deliberately broken, but accidents happen, so if something stops working, please either fix and
   commit the code or tell Gavin.

Package Options
===============

The Stoner package supports a set of package level options to control preferences such as whether to show objects using 'rich'
representations such as png or html output formats.::

    from Stoner import Options
    Options.short_repr=True

The **options** object supports reading and setting options through attribute assignment. Deleting an Option attribute
resets the Option back to the corresponding default value. **dir(Options)** will give a list of all possible package options.

