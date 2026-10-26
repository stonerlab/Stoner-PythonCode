Configuring fitting workflows with INI files
********************************************

.. currentmodule:: Stoner

An initialisation (INI) file stores named settings in sections. It is useful
when you want to repeat a fitting workflow with different files or parameter
values without editing the Python program each time.

Read :doc:`fitting_models` and :doc:`fitting_methods` first. This page is an
advanced worked example rather than a prerequisite for ordinary curve fitting.

.. contents:: On this page
   :local:
   :depth: 1

Configure the measurement and model
===================================

For writing general purpose fitting codes, it can be useful to drive the fitting code from a separate initialisation file so that users do not have to
edit the source code. :py:meth:`~Stoner.core.data.Data.lmfit` and :py:meth:`~Stoner.core.data.Data.odr` combined with :py:mod:`Stoner.analysis.fitting` provide some mechanisms to enable this.

Firstly, the initialisation file should take the form like so.

.. include:: ../../scripts/PCAR-New.ini
   :literal:

This initialisation file can be passed to :py:func:`Stoner.analysis.fitting.models.cfg_data_from_ini` which will use the information in the [Data]
section to read in the data file and identify the x and y columns.

The INI examples retain the CSV layout controls as reference. The CSVFile loader
accepts zero-based ``header_line`` and ``data_line`` row numbers and independent
``header_delim`` and ``data_delim`` separators. A ``header_line`` of ``None``
selects headerless input; ``None`` delimiters request automatic detection.
However, ``cfg_data_from_ini()`` currently does not forward these layout entries;
this is tracked for a future release in
`issue #41 <https://github.com/stonerlab/Stoner-PythonCode/issues/41>`_.
For custom layouts, pass the options directly to ``Data.load(...,
filetype="CSVFile", ...)``. The bundled PCAR examples use the CSV defaults:
headers on the first line and comma-separated data on the following line.
The historical ``start_line`` and ``separator`` entries correspond to the
current loader's ``data_line`` and ``header_delim``/``data_delim`` names.

The initialisation file can then be passed to :py:func:`Stoner.analysis.fitting.models.cfg_model_from_ini` which will use the configuration file to
setup the model and parameter hints. The configuration file should have one section for each parameter in the model. This function
returns the configured model and also a 2D array of values to feed as the starting values to :py:meth:`~Stoner.core.data.Data.lmfit`. Depending
on the presence and values of the *vary* and *step* keys, the code will either perform a single fitting attempt, or do a mapping of the
:math:`\\chi^2` goodness of fit.

Since both the :py:meth:`~Stoner.core.data.Data.odr` and :py:meth:`~Stoner.core.data.Data.differential_evolution` supports the same interface, either can be used as a
drop-in replacement as well. (Although it is interesting to note that in this example, they give quite different results - a matter of interest
for the physics!)

.. plot:: samples/odr_demo.py
    :include-source:
    :outname: odr_demo

Next steps
==========

Return to :doc:`curve_fitting` for the learning path or
:doc:`fitting_options` for output and metadata options.
