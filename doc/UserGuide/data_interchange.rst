Working with pandas and NumPy
*****************************

.. currentmodule:: Stoner

You do not need to use pandas or NumPy directly for most Stoner tasks. This page
is for calculations that need those libraries, or for transferring a measurement
between objects. Start with :doc:`data_indexing` and :doc:`data_editing` if you
are still learning how to read and change a Data object.

The examples use ``d`` from :doc:`datafile` or :doc:`data_loading`. A copy is
independent: changing it does not change ``d``. An editing context provides a
temporary working copy and applies your changes when the block succeeds.

.. contents:: On this page
   :local:
   :depth: 1

Choose an export or editing interface
=====================================

Most tasks can use Stoner's column, selection and analysis methods directly.
If you also use pandas or NumPy, choose the representation that fits your task:

* ``d.data`` is the stored pandas DataFrame. It contains the raw measurements,
  including masked values. Pandas calculations do not apply Stoner's mask.
* ``d.to_numpy()`` returns an independent, writable NumPy masked array. Changing
  it does not change ``d``. ``d.asarray()`` is an independent, read-only snapshot.
* ``d.to_pandas()`` returns an independent DataFrame. It warns when information
  such as masks or metadata is omitted. It is not a lossless backup.
* ``d.export_storage()`` preserves values, masks, column assignments and identities,
  fill value and typed metadata. ``Data.from_storage(...)`` restores that state.

To calculate with NumPy and write the result back, use an editing context::

    with d.edit_numpy() as values:
        values[:, 1] *= 2

The change is applied when the block finishes successfully. If the block raises
an exception, the draft changes are discarded. ``edit_pandas()`` works similarly::

    with d.edit_pandas() as frame:
        frame.iloc[:, 1] *= 2

Keep the number and order of rows and columns, labels and data type unchanged
inside these blocks. Use Stoner methods to add, remove, rename or sort columns
and rows. Do not edit ``d`` separately while an editing block is open.

For direct pandas cell edits through ``d.data``, keep the same restrictions and
obtain ``d.data`` again after changing the object with Stoner methods. Previously
saved references may no longer refer to its current data. Stoner detects some
unsupported structural changes and raises ``ValueError``, but cannot detect
every same-shaped rearrangement.

``Data.from_pandas(frame)`` imports a table with string column headers and a
standard positional row index. Use ``index="discard"`` to discard another index.
See :doc:`developer` for the full storage interchange rules.

By default, ``to_pandas()`` uses a two-level column index containing headers and
column assignments, and replaces masked entries with NaN. Use ``format="plain"``
for ordinary headers or ``masked="raw"`` to export hidden measurements too.
Neither choice preserves the separate mask or typed metadata. Mixed numerical
column types on import require an explicit common ``dtype``. A two-level
header/assignment index can also be imported with ``Data.from_pandas``.

Next steps
==========

Continue with :doc:`analysisfile`, :doc:`plotfile` or :doc:`curve_fitting`
for analysis workflows. Return to :doc:`datafile` for the full learning path.
