=================
Installing Stoner
=================

Stoner provides classes and functions for reading, manipulating, fitting and
plotting experimental data. This guide describes the maintained source;
requirements for older published releases may differ.

Requirements and packages
=========================

.. include:: ../../README.rst
   :start-after: .. installation-start
   :end-before: .. installation-end

Installing from source
======================

The maintained source is on the ``stable`` branch. To install it directly::

    python -m pip install git+https://github.com/stonerlab/Stoner-PythonCode.git@stable

For an editable checkout with test dependencies, follow the
:doc:`developer guide <developer>`.

Next steps
==========

Continue with :doc:`loading and examining data <datafile>`,
:doc:`plotting <plotfile>` or :doc:`curve fitting <curve_fitting>`.
The :doc:`API reference </Stoner>` describes individual methods and parameters.

Package options
===============

Package options control preferences such as rich HTML and image representations::

    from Stoner import Options
    Options.short_repr = True

Read and set options through attributes. Deleting an option attribute resets
it to its default; ``dir(Options)`` lists the available options.
