==========================
:mod:`Stoner` Package
==========================

.. py:module:: Stoner

----------------
Primary Classes
----------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Stoner.core.data.Data
    Stoner.folders.mixins.DataFolder
    Stoner.Image.core.ImageFile
    Stoner.Image.folders.ImageFolder

Inheritance Diagrams
^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: Stoner.core.data.Data

.. inheritance-diagram:: Stoner.folders.mixins.DataFolder

.. inheritance-diagram:: Stoner.Image.core.ImageFile

.. inheritance-diagram:: Stoner.Image.folders.ImageFolder


-----------------------
Numerical Data Objects
-----------------------

Core Package and Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.core.data
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.base
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.core.array
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

.. automodapi:: Stoner.plot.core
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.plot
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.plot.formats
    :headings: -~

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
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.formats.decorators
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
