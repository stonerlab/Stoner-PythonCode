====================
Stoner documentation
====================

Stoner helps you analyse experimental measurements: load instrument data,
keep metadata with your results, fit physical models, process images and
apply analyses to collections of files. It was developed in the Condensed
Matter Physics group at the University of Leeds.

Start here
==========

* :doc:`Install Stoner <UserGuide/install>` for supported Python versions,
  Conda and pip commands, and optional features such as OCR.
* :doc:`Load and inspect measurements <UserGuide/datafile>` to work with
  numerical data, metadata and column roles.
* :doc:`Plot results <UserGuide/plotfile>` and
  :doc:`fit curves <UserGuide/curve_fitting>` for common analysis workflows.
* :doc:`Process collections <UserGuide/datafolder>` or
  :doc:`work with images <UserGuide/image>` for larger datasets.
* :doc:`Browse the cookbook <UserGuide/cookbook>` for worked examples.

Main objects
============

:py:class:`~Stoner.core.data.Data` represents a numerical measurement with
metadata, masks and column roles. :py:class:`~Stoner.folders.mixins.DataFolder`
groups measurements for filtering and bulk analysis.

:py:class:`~Stoner.Image.core.ImageFile` represents an image with metadata and
processing methods. :py:class:`~Stoner.Image.folders.ImageFolder` groups images;
:py:class:`~Stoner.Image.stack.ImageStack` stores them together in an array.
All five classes can be imported directly from ``Stoner``.

Many analysis methods update the object in place and return it for chaining.
Consult the API reference for each method's return value and use ``clone``
when an independent copy is needed.

Guides and reference
====================

.. toctree::
   :maxdepth: 2

   User Guide <UserGuide/ugindex>
   Module API Reference <Stoner>
   Project Overview <readme>

For contributing, environment setup and documentation commands, see the
:doc:`developer's guide <UserGuide/developer>`. The
`source repository <https://github.com/stonerlab/Stoner-PythonCode>`_ contains
sample data, runnable examples and the maintenance plan.
