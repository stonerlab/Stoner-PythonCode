**************************
:mod:`Stoner` Package
**************************

.. module:: Stoner

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Data
    DataFolder

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: Data DataFolder

Core Classes
============

.. automodapi:: Stoner.Core
    :inherited-members:

Analysis Module
===============

.. automodapi:: Stoner.Analysis
    :no-inheritance-diagram:

Plot Module and Sub-packages
============================

.. automodapi:: Stoner.plot
    :no-inheritance-diagram:

.. automodapi:: Stoner.plot.formats

.. automodapi:: Stoner.plot.utils
   :no-main-docstr:

Folders Module
==============
.. automodapi:: Stoner.Folders
    :inherited-members:

:mod:`FileFormats` Module
=========================

.. module:: Stoner.FileFormats

Generic Fomats
--------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    CSVFile
    SPCFile
    TDMSFile
    OVFFile
    EasyPlotFile

Instrument Formats
------------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    BigBlueFile
    FmokeFile
    QDFile
    RigakuFile
    VSMFile
    MokeFile
    XRDFile
    LSTemperatureFile

Facility Outputs
----------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    BNLFile
    MDAASCIIFile
    OpenGDAFile
    RasorFile
    SNSFile

.. automodapi:: Stoner.Fit
    :no-inheritance-diagram:


The following modules offer specialised file and foler formats.

HDF Support
-----------

.. automodapi:: Stoner.HDF5
   :no-main-docstr:

Zip File Support
----------------

.. automodapi:: Stoner.Zip
   :no-main-docstr:


Utility Functions
=================

.. automodapi:: Stoner.Util
   :no-main-docstr:

.. automodapi:: Stoner.tools
   :allowed-package-names: Stoner.tools

Image Subpackage
================

.. automodapi:: Stoner.Image

.. automodapi:: Stoner.Image.folders
   :no-main-docstr:

.. automodapi:: Stoner.Image.stack
   :no-main-docstr:

.. automodapi:: Stoner.Image.kerr
   :no-main-docstr:

.. automodapi:: Stoner.Image.imagefuncs
   :no-main-docstr:

.. automodapi:: Stoner.Image.util
   :allowed-package-names: Stoner.Image
