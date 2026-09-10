************************************************************
ConvertingScripts from Older Versions of the Stoner Package
************************************************************

.. currentmodule:: Stoner

The Stoner package has gradually undergone several changes as additional functions have been added to keep the codebase
manageable. This document summarises the main changes from previous versions.

Python Version Support
======================

    - Versions <= 0.5 supported Python 2.7 with some support for Python 3.x
    - Version 0.7 and 0.8 supported Python 2.7, 3.5 and 3.6
    - Version 0.9 supports Python 2.7, 3.5, 3.6 and 3.7. The final release of 0.9 also supports Python 3.8
    - Version 0.10 does *not* support Python 2.7 or 3.5. The supported versions of Python are 3.6, 3.7 and 3.8

Data versus DataFile, AnalysisFile and PlotFile
===============================================

Originally the package provided several subclasses of a **DataFile** class and the user was encouraged to make a superclass
of all of them. Thus code would have::

    import Stoner.Analysis as SA
    import Stoner.Plot as SP
    import Stoner.Core as SC

    class Data(SC.AnalysisFile, SP.PlotFile, SC.DataFile):
        pass

This should now simply be replaced with::

    from Stoner import Data

Stoner.PlotFormats and Stoner.Plot modules removed
==================================================

The **Stoner.Plot** and **Stoner.PlotFormats** modules were replaced with the :py:mod:`Stoner.plot` package. Code that did::

    from Stoner.PlotFormats import JTBStyle, TexFormatter

should now do::

    from Stoner.plot.formats import JTBPlotStyle, TexFormatter

Most of the functionality of the  **Stoner.Plot** module was in the **PlotFile** subclass and is now accessed via the :py:class:`~Stoner.core.data.Data`
class. A few additional functions that used to be in **Stoner.Plot** are now in :py:mod:`Stoner.plot.utils`.

Stoner.Folders Module deprecated
================================

The code around :py:class:`~Stoner.folders.mixins.DataFolder` was extensively rewritten and cleaned up in versions 0.9 onwards. Firstly, the old
**Stoner.Folders** module is replaced with the :py:mod:`Stoner.folders` package, and the :py:class:`~Stoner.folders.mixins.DataFolder` class is
now directly importable from the root :py:mod:`Stoner` package. Thus::

    from Stoner.Folders import DataFolder

becomes::

    from Stoner import DataFolder

The :py:mod:`Stoner.folders` package contains many modules and subpackages to implement generic folder-like objects, but for end
users of the package, the :py:class:`~Stoner.folders.mixins.DataFolder` class is probably all that is required to be imported.

Stoner.Fit Module deprecated
============================

The **Stoner.Fit** module contained a mixed bag of assorted fitting functions and **lmfit.Model** classes. These have been
reorganised into a series of sub-modules of the :py:mod:`Stoner.analysis` package. Although this makes the import lines
rather long, it groups the functions more logically by physics theme. Thus::

    from Stoner.Fit import blochGrueneisen

becomes::

    from Stoner.analysis.fitting.models.e_transport import blochGrueneisen

The **Stoner.Fit** module also contained various functions to make **lmfit.Model** classes from functions or from ini files -
these functions are now in the :py:mod:`Stoner.analysis.fitting.models` package directly.

Stoner.FileFormats Module deprecated
====================================

The **Stoner.FileFormats** module was getting a bit big and unwieldy with many different file formats being implemented. It has
now been superseded by the :py:mod:`Stoner.formats` package which groups the fileformats into different sub-modules.

Generally there is no particular reason to directly access the individual file format classes - the :py:class:`~Stoner.core.data.Data` class
can access all the subclasses loaded in memory and so::

    import Stoner.FileFormats

can simply be removed if the :py:class:`~Stoner.core.data.Data` is used.

Stoner.Image changes
====================

The original version of the :py:mod:`Stoner.Image` package provided a NumPy array subclass with metadata,
:py:class:`Stoner.Image.core.ImageArray`. For many purposes it is more useful to have a class that wraps the image data and
provides additional attributes and methods alongside it, avoiding name collisions with NumPy methods and attributes. For this reason, the
preferred class is :py:class:`Stoner.Image.core.ImageFile`, which is an analogue of :py:class:`Stoner.core.data.Data`. Generally the code can be
transferred directly so::

    from Stoner.Image import ImageArray

becomes::

    from Stoner import ImageFile

Like :py:class:`Stoner.Image.core.ImageArray`, :py:class:`Stoner.Image.core.ImageFile` can access key image-processing functions from :py:mod:`skimage`,
:py:mod:`scipy.ndimage` and the :py:mod:`Stoner.Image.imagefuncs` modules.

Deprecated Image functions
--------------------------

The following ImageArray/ImageFile methods should be swapped:

-   ``.box()`` - use :py:meth:`Stoner.Image.core.ImageArray.crop` instead
-   ``.crop_image()`` - use :py:meth:`Stoner.Image.core.ImageArray.crop` instead
-   ``.convert_float()`` - use :py:meth:`Stoner.Image.core.ImageArray.asfloat` instead
-   ``.convert_int()`` - use :py:meth:`Stoner.Image.core.ImageArray.asint` instead


Stoner.DataFolder/Stoner.ImageFolder changes
============================================

:py:class:`~Stoner.folders.mixins.DataFolder` and :py:class:`~Stoner.Image.folders.ImageFolder` are essentially similar objects for managing collections
of :py:class:`~Stoner.core.data.Data` and :py:class:`~Stoner.Image.core.ImageFile` respectively.

Earlier versions of the Stoner package exposed the ability to call methods of the stored :py:class:`~Stoner.core.data.Data`/
:py:class:`~Stoner.Image.core.ImageFile` instances by calling a corresponding method directly on the folder object. The problem with this
is that if there is a name collision with a method intended to work directly on the Folder, it's not clear what method is being
called. To remove this problem, and to make it a little more explicit when accessing a method of the stored instances, the
:py:attr:`~Stoner.folders.core.BaseFolder.each` attribute is now provided. Thus code like::

    fldr=DataFolder(".", pattern="*.txt", setas="xy")
    fldr.curve_fit(Linear)

now becomes::

    fldr=DataFolder(".", pattern="*.txt", setas="xy")
    fldr.each.curve_fit(Linear)

One of the frequently required operations is to loop through all the files in a folder and extract some of their metadata
and build that into a new table. This used to be done something like::

    result=Data()
    for data in fldr:
        result+=[data["thing_1"], data["thing_2"]]

The new :py:attr:`~Stoner.folders.core.BaseFolder.metadata` attribute and :py:meth:`Stoner.folders.metadata.MetadataProxy.slice` method
allow this to be done directly::

    result=fldr.metadata.slice(["thing_1","thing_2"], output="Data")


