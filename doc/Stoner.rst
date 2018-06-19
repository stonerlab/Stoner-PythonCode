:mod:`Stoner` Package
=======================

.. module:: Stoner

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Data
    DataFolder

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: Data DataFolder

.. automodapi:: Stoner.Core
    :inherited-members:

.. automodapi:: Stoner.Analysis
    :no-inheritance-diagram:

.. automodapi:: Stoner.plot
    :no-inheritance-diagram:

.. automodapi:: Stoner.plot.formats

.. automodapi:: Stoner.plot.utils
   :no-main-docstr:


.. automodapi:: Stoner.Folders

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


.. automodapi:: Stoner.HDF5
   :no-main-docstr:


.. automodapi:: Stoner.Zip
   :no-main-docstr:



.. automodapi:: Stoner.Util
   :no-main-docstr:

.. automodapi:: Stoner.tools
   :allowed-package-names: Stoner.tools

Subpackages
===========

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
