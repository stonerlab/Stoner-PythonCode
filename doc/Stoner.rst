Stoner Package
###############

:mod:`Core` Module
==================

.. module:: Stoner.Core

.. autosummary::
    :toctree: classes
    :template: classdocs.rst

    typeHintedDict
    DataFile
    StonerLoadError

:mod:`Analysis` Module
=======================

.. module:: Stoner.Analysis


.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    AnalyseFile


:mod:`Plot` Module
==================

.. module:: Stoner.Plot


.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    PlotFile

:mod:`plotutils` Module
=======================

.. module:: Stoner.plotutils


.. autosummary::
    :toctree: functions
    :template: autosummary/module.rst


:mod:`Folders` Module
======================

.. module:: Stoner.Folders


.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    DataFolder

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
    QDSquidVSMFile
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

:mod:`PlotFormats` Module
==========================

.. automodapi:: Stoner.PlotFormats
    :no-heading:
    :no-inheritance-diagram:
    :headings: =-
    :skip: AutoLocator
    :skip: EngFormatter
    :skip: Formatter
    :skip: range


See also the stylesheets available in the stylelib directory.


:mod:`Fit` Module
=================

.. automodapi:: Stoner.Fit
    :no-heading:
    :no-inheritance-diagram:
    :headings: =-
    :skip: Linear
    :skip: PowerLaw
    :skip: Quadratic
    :skip: Model
    :skip: quadratic
    :skip: jit
    :skip: update_param_vals
    :skip: quad


The following modules offer specialised file and foler formats.

:mod:`HDF5` Module
=========================

.. module:: Stoner.HDF5

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    HDF5File
    HDF5Folder

:mod:`Zip` Module
=========================

.. module:: Stoner.Zip

.. autosummary:: 
   :toctree: classes
   :template: classdocs.rst

    ZipFile
    ZipFolder

:mod:`Util` Module
==================

.. module:: Stoner.Util

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Data

.. autosummary::
    :toctree: functions
    
    format_error
    hysteresis_correct
    ordinal
    split_up_down


:mod:`mpfit` Module
===================

.. automodule:: Stoner.mpfit

:mod:`nlfit` Module
===================

.. automodule:: Stoner.nlfit


:mod:`pyTDMS` Module
--------------------

.. automodule:: Stoner.pyTDMS


Subpackages
-----------

.. toctree::

    Stoner.gui

