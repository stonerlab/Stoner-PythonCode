==========================
:mod:`Stoner` Package
==========================

.. py:module:: Stoner

.. currentmodule:: Stoner

----------------
Primary Classes
----------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    core.data.Data
    folders.mixins.DataFolder
    Image.core.ImageFile
    Image.folders.ImageFolder

For collections stored as an image stack, use
:py:class:`~Stoner.Image.stack.ImageStack`.

Inheritance Diagrams
^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: Stoner.core.data.Data

.. inheritance-diagram:: Stoner.folders.mixins.DataFolder

.. inheritance-diagram:: Stoner.Image.core.ImageFile

.. inheritance-diagram:: Stoner.Image.folders.ImageFolder

.. inheritance-diagram:: Stoner.Image.stack.ImageStack


-----------------------
Numerical Data Objects
-----------------------

Core Package and Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. The skip lists below exclude implementation imports from module tables.
   Automodsumm's rendering and stub generation use different import filters;
   keep these lists explicit so intentional public re-exports remain visible.

.. automodapi:: Stoner.core.data
    :skip: ClassTester, DataFileInterfacesMixin, DataFileOperatorsMixin, DataFilePropertyMixin, DataOwner, DataStorage, StorageBridgeMixin, FileManager, Iterable, Mapping, MutableSequence, StonerLoadError, StonerSetasError, Tab_Delimited, TextWrapper, TypeHintedDict, metadataObject
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.base
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.setas
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.exceptions
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.utils
    :no-inheritance-diagram:
    :no-main-docstr:
    :headings: -~


Analysis Package
^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.analysis
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.analysis.fitting
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.analysis.utils
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.analysis.fitting.models
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.tools.formatting
    :no-inheritance-diagram:
    :headings: -~

Fitting Models
^^^^^^^^^^^^^^
.. toctree::
    :maxdepth: 2
    :name: models=toc

    Fitting Models <analysis-fitting>


Plot Package and Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.plot.accessor
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.plot
    :no-inheritance-diagram:
    :headings: -~
    :skip: SavedFigure, MultiPanel, StackVertical, DoubleYAxis, InsetPlot, CentredAxes, PlotLabeller, TexFormatter, TexEngFormatter

Stonerplots exports
-------------------

These are direct aliases of the stonerplots classes. The
:ref:`plotting-contexts` guide shows their use through ``d.plot``; upstream
signatures, keyword arguments and returned objects are retained.

.. py:class:: SavedFigure
    :module: Stoner.plot

    Save selected figures and apply temporary styles. Also ``d.plot.SavedFigure``.

.. py:class:: MultiPanel
    :module: Stoner.plot

    Arrange regular or irregular panels. Also ``d.plot.MultiPanel``.

.. py:class:: StackVertical
    :module: Stoner.plot

    Arrange stacked panels. Also ``d.plot.StackVertical``.

.. py:class:: DoubleYAxis
    :module: Stoner.plot

    Add an independent Y axis. Also ``d.plot.DoubleYAxis``.

.. py:class:: InsetPlot
    :module: Stoner.plot

    Add and position an inset. Also ``d.plot.InsetPlot``.

.. py:class:: CentredAxes
    :module: Stoner.plot

    Centre axes through an origin. Also ``d.plot.CentredAxes``.

.. py:class:: PlotLabeller
    :module: Stoner.plot

    Apply tick formatters and locators. Also ``d.plot.PlotLabeller``.

.. py:class:: TexFormatter
    :module: Stoner.plot

    Format ticks in mathematical notation. Also ``d.plot.TexFormatter``.

.. py:class:: TexEngFormatter
    :module: Stoner.plot

    Format ticks with engineering prefixes. Also ``d.plot.TexEngFormatter``.

.. automodapi:: Stoner.plot.utils
    :headings: -~

.. automodapi:: Stoner.plot.functions
    :no-inheritance-diagram:
    :no-main-docstr:
    :headings: -~

File Formats Package Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.formats
    :no-inheritance-diagram:
    :headings: -~

Data Classes
------------

.. automodapi:: Stoner.formats.data
    :no-inheritance-diagram:


Image Classes
-------------

.. automodapi:: Stoner.formats.image
    :no-inheritance-diagram:

.. automodapi:: Stoner.formats.image.hdf5
    :skip: deepcopy, get_filename, make_Data, register_loader
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.formats.decorators
    :skip: signature
    :no-inheritance-diagram:
    :headings: -~


-------------------------------------
Folders package - Collections Classes
-------------------------------------

Main Classes
^^^^^^^^^^^^
.. automodapi:: Stoner.folders
    :headings: -~


Folders Package and Submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.folders.core
    :no-inheritance-diagram:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.folders.mixins
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.folders.functions
    :skip: append, array, isiterable, np_any
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.folders.each
    :headings: -~

.. automodapi:: Stoner.folders.metadata
    :headings: -~

.. automodapi:: Stoner.folders.groups
    :headings: -~

.. automodapi:: Stoner.folders.zip
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.folders.hdf5
    :no-inheritance-diagram:
    :headings: -~

-----------------
Image Subpackage
-----------------

Main Image Classes
^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.Image
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.Image.folders
   :no-main-docstr:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.Image.stack
   :no-main-docstr:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.Image.attrs
   :no-main-docstr:
    :inherited-members:
    :headings: -~


Attocube SPM Scans
------------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Stoner.formats.attocube.AttocubeScan



Additional Image Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.Image.imagefuncs
    :no-inheritance-diagram:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.Image.util
    :no-inheritance-diagram:
    :allowed-package-names: Stoner.Image
    :inherited-members:
    :headings: -~


Kerr Image Handling
^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.Image.kerr
    :no-inheritance-diagram:
   :no-main-docstr:
   :inherited-members:
    :headings: -~
