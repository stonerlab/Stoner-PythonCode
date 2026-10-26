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

To install the source from the ``stable`` branch directly::

    python -m pip install git+https://github.com/stonerlab/Stoner-PythonCode.git@stable

This guide also describes the development API on ``devel``, including the
0.12 storage and plotting changes. To try that version in a separate environment::

    python -m pip install git+https://github.com/stonerlab/Stoner-PythonCode.git@devel

Development versions may change before release. Use the documentation matching
your installed version when following examples.

For an editable checkout with test dependencies, follow the
:doc:`developer guide <developer>`.

Next steps
==========

Continue with :doc:`working with numerical data <datafile>`,
:doc:`plotting <plotfile>` or :doc:`curve fitting <curve_fitting>`.
The :doc:`API reference </Stoner>` describes individual methods and parameters.

Package options
===============

Package options control preferences such as rich HTML and image representations::

    from Stoner import Options
    Options.short_repr = True

Read and set options through attributes. Deleting an option attribute resets
it to its default; ``dir(Options)`` lists the available options.
