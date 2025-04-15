==========================
:mod:`Stoner` Package
==========================

----------------
Primary Classes
----------------

.. autosummary::
   :toctree: classes
   :template: classdocs.rst

    Stoner.Data
    Stoner.DataFolder
    Stoner.Image.ImageFile
    Stoner.Image.ImageFolder

Inheritance Diagrams
^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: Stoner.Data

.. inheritance-diagram:: Stoner.DataFolder

.. inheritance-diagram:: Stoner.ImageFile

.. inheritance-diagram:: Stoner.ImageFolder


-----------------------
Numerical Data Objects
-----------------------

Core Package and Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. automodapi:: Stoner.Analysis
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.analysis.fitting
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

.. automodapi:: Stoner.plot
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.plot.formats
    :headings: -~

.. automodapi:: Stoner.plot.utils
   :no-main-docstr:
    :headings: -~

File Formats Package Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Classes
------------

.. automodapi:: Stoner.formats.data
    :no-inheritance-diagram:


Image Classes
-------------

.. automodapi:: Stoner.formats.image
    :no-inheritance-diagram:


The following modules offer specialised file and foler formats.

HDF Support
^^^^^^^^^^^

.. automodapi:: Stoner.HDF5
    :no-inheritance-diagram:
   :no-main-docstr:
    :headings: -~

Zip File Support
^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.Zip
    :no-inheritance-diagram:
   :no-main-docstr:
    :headings: -~


Utility Functions
^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.Util
    :no-inheritance-diagram:
   :no-main-docstr:
    :headings: -~

.. automodapi:: Stoner.tools
   :allowed-package-names: Stoner.tools
    :no-inheritance-diagram:
    :headings: -~

-------------------------------------
Folders pacakge - Collections Classes
-------------------------------------

Main Classes
^^^^^^^^^^^^
.. automodapi:: Stoner.Folders
    :headings: -~


Folders Package and Submodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodapi:: Stoner.folders
    :no-inheritance-diagram:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.folders.core
    :no-inheritance-diagram:
    :inherited-members:
    :headings: -~

.. automodapi:: Stoner.folders.mixins
    :no-inheritance-diagram:
    :headings: -~

.. automodapi:: Stoner.folders.each
    :headings: -~

.. automodapi:: Stoner.folders.metadata
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
    :no-main-docstr:
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
