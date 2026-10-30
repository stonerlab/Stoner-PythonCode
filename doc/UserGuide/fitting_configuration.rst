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
edit the source code. :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit` and :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.odr` combined with :py:mod:`Stoner.analysis.fitting` provide some mechanisms to enable this.

Firstly, the initialisation file should take the form like so.

.. include:: ../../scripts/PCAR-New.ini
   :literal:

This initialisation file can be passed to :py:func:`Stoner.analysis.fitting.models.cfg_data_from_ini` which will use the information in the [Data]
section to read in the data file and identify the x and y columns.

CSV layout options
------------------

For ``type: CSVFile``, the INI helper forwards layout settings to the CSV loader.
Use zero-based row numbers: the first line is row 0. For example, this file has
two preamble lines, a header, one intervening line and then measurements::

    [Data]
    type: CSVFile
    x: V
    y: G
    e: dG
    header_line: 2
    data_line: 4
    header_delim: |
    data_delim: ;

Here the header might be ``V|G|dG`` and the first measurement ``-1;2;0.1``.
Omit ``e`` if there is no uncertainty column. Each omitted layout option retains
the loader's default: header row 0, data row 1 and comma delimiters. These are
the settings used by the bundled PCAR measurements.

For headerless measurements beginning on the first line, set ``header_line:
None`` and ``data_line: 0``, and select columns by their zero-based positions
(``x: 0``, ``y: 1`` and, optionally, ``e: 2``). Set either delimiter to ``None``
to detect it independently. A literal ``\t`` denotes a tab; use ``" "`` for a
space delimiter. Detection recognises tabs, spaces, commas and semicolons;
specify other delimiters such as ``|`` explicitly.
Layout entries are only applied to CSVFile, so retaining them
when switching to another loader does not change that loader's behaviour.

Compatibility with older INI files
----------------------------------

The older ``start_line`` and ``separator`` options remain supported::

    [Data]
    type: CSVFile
    x: V
    y: G
    header_line: 1
    start_line: 2
    separator: ,

When ``start_line`` is present without ``data_line``, both ``start_line`` and any
``header_line`` use the historical **one-based** convention. This example reads
the header from the first line and measurements from the second. ``header_line:
None`` still selects headerless input. An explicit ``data_line`` takes
precedence over ``start_line`` and selects modern zero-based INI row numbers;
when converting a legacy file, update its header row number at the same time.

``separator`` supplies both delimiters, but explicit ``header_delim`` and
``data_delim`` override it independently. Passing layout keywords directly to
``cfg_data_from_ini(...)`` overrides the corresponding normalised INI settings.
These Python keywords always use the native CSV loader's zero-based numbering::

    data = cfg_data_from_ini("fit.ini", filename="measurement.csv", data_line=4)

This fixes the INI forwarding defect described in
`issue #41 <https://github.com/stonerlab/Stoner-PythonCode/issues/41>`_. Direct
``Data.load(..., filetype="CSVFile", ...)`` remains available for loading a CSV
without a fitting configuration.

Configure and run the model
---------------------------

The initialisation file can then be passed to :py:func:`Stoner.analysis.fitting.models.cfg_model_from_ini` which will use the configuration file to
setup the model and parameter hints. The configuration file should have one section for each parameter in the model. This function
returns the configured model and also a 2D array of values to feed as the starting values to :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.lmfit`. Depending
on the presence and values of the *vary* and *step* keys, the code will either perform a single fitting attempt, or do a mapping of the
:math:`\\chi^2` goodness of fit.

Since both the :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.odr` and :py:meth:`~Stoner.analysis.fitting.accessor.DataFitter.differential_evolution` supports the same interface, either can be used as a
drop-in replacement as well. (Although it is interesting to note that in this example, they give quite different results - a matter of interest
for the physics!)

.. plot:: samples/odr_demo.py
    :include-source:
    :outname: odr_demo

Next steps
==========

Return to :doc:`curve_fitting` for the learning path or
:doc:`fitting_options` for output and metadata options.
