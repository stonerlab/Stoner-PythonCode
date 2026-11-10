Converting scripts from 0.10 or 0.11 to 0.12
********************************************

.. currentmodule:: Stoner

Use this guide to update a script written for Stoner 0.10 or 0.11 to the
0.12 API documented in this user guide. The changes are grouped by the work
needed in your script. For the history of individual releases, see
:doc:`../release-notes`.

If you are starting from 0.10, first check the imports below, then work through
the plotting, fitting and numerical-storage changes. Scripts written for 0.11
will usually already use the current imports, but still need those API changes.
You can update directly to 0.12 without installing each intervening release.

Beta compatibility policy
=========================

The documented API is locked from 0.12.0b1. Beta releases permit bug fixes,
documentation and small compatible additions. Release candidates also freeze
additions. Existing direct Data fitting methods and their ``xcol``/``ycol``
keywords, legacy ``ImageFile.imshow`` with its Figure return, and ``level_image``
remain supported throughout the 0.12 beta and RC series. No new deprecation
warnings or removal deadline are introduced for those interfaces. The migration
tables below distinguish interfaces already removed during alpha development
from compatibility calls that remain supported.

.. contents:: On this page
   :local:
   :depth: 1

Prepare the environment
=======================

Stoner 0.12 requires Python 3.12 or newer. Use a new environment if your
0.10 or 0.11 installation uses an older Python. Required dependencies now
include NumPy 2.0 or newer, pandas 2.2 or newer, xarray 2024.10 or newer,
and stonerplots 1.9.8 or newer. See :doc:`install` for installation instructions.

Keep copies of your original measurements and scripts while converting them.
Check numerical results, exclusions and metadata as well as whether the script
runs successfully.

Check imports, especially in 0.10 scripts
=========================================

The public classes are available from the package root in both older series
and in 0.12. If your script already uses these imports, keep them::

    from Stoner import Data, DataFolder, ImageFile, ImageFolder, ImageStack

Some 0.10 scripts instead use compatibility modules with capitalised names.
Replace those imports with the public classes or the corresponding packages:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Older import or approach
     - Replacement for 0.12
   * - ``from Stoner.Core import DataFile``
     - ``from Stoner import Data``
   * - ``from Stoner.Folders import DataFolder``
     - ``from Stoner import DataFolder``
   * - Importing ``Stoner.FileFormats`` to enable loaders
     - Remove the import; ``Data`` selects registered loaders automatically.

Use ``Data`` directly rather than building a class from numerical, analysis and
plotting mixins. Imports from ``Stoner.analysis.fitting.models`` remain the way
to select physical models; these subpackages organise models by their physics.
See :doc:`fitting_models`.

Format implementations now use registered loader functions rather than a
subclass for each format. For ordinary loading, prefer
``d = Data("measurement.txt")``. If automatic identification is insufficient,
use ``Data.load(..., filetype=...)`` with the registered format name. See
:doc:`data_loading` for file-loading examples and :doc:`developer` for custom
loader registration.

Move plotting to the plotting accessor
======================================

In 0.12, ``d.plot`` is a callable accessor. It still uses column roles and
uncertainties, but drawing returns a Matplotlib Axes. Display and saving are
explicit. For example, replace an older call such as
``d.plot_xy(xcol="Field", ycol="Signal", fmt="r-")`` with::

    ax = d.plot.line(x="Field", y="Signal", fmt="r-")
    ax.set_xlabel("Magnetic field (T)")
    ax.set_ylabel("Signal")
    ax.figure.savefig("measurement.png")
    d.plot.show()

For an automatic choice of plot type, use ``ax = d.plot()`` or
``ax = d.plot(x="Field", y="Signal")``. Keep ``d`` separately if your old
script chained Data operations from the plot's return value.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Previous usage
     - Replacement for 0.12
   * - ``d.plot(xcol=0, ycol=1)``
     - ``d.plot(x=0, y=1)``
   * - ``d.plot_xyz(...)``
     - ``d.plot.surface(...)``; use the named accessor method for other fields.
   * - ``d.annotate_fit(model, ...)``
     - ``d.plot.annotate_fit(model, ...)``
   * - ``d.fig`` or ``d.figure``
     - ``ax.figure`` from the returned Axes.
   * - ``d.xlabel = "Field"``
     - ``ax.set_xlabel("Field")`` or ``d.plot.xlabel = "Field"``.
   * - Plot templates or ``template=...``
     - stonerplots styles and layout contexts through ``d.plot``.
   * - ``showfig``, ``show_plot`` or ``save_filename``
     - Explicit display and saving as in the example above.

The old top-level plotting methods and broad Axes/Figure properties on Data
have been removed. The ordinary accessor rejects old plotting options such as
``figure``, ``template``, ``plotter``, ``multiple`` and the ``*col`` keywords.

Use styles around drawing rather than assigning a template::

    with d.plot.style("stoner"):
        ax = d.plot()

The former JTB template is not an alias for a stonerplots style. Layout helpers
include ``d.plot.OffsetPlots``, ``DoubleXAxis``, ``DoubleYAxis`` and
``DoubleXYAxis``. See :doc:`plot_migration` for the full call-by-call conversion
table and :doc:`plot_recipes` for translated style and layout examples.

Use the fitting accessor
========================

Fitting in 0.12 belongs to ``d.fit``. Replace, for example,
``d.lmfit(Linear, xcol="Field", ycol="Signal")`` with
``d.fit.lmfit(Linear, x="Field", y="Signal")``. The registered shortcut
``d.fit.linear(x="Field", y="Signal")`` selects the linear model for you.

Calling ``d.fit(model, ...)`` defaults to lmfit. Choose another algorithm
explicitly through ``d.fit.curve_fit``, ``d.fit.odr`` or
``d.fit.differential_evolution``. Polynomial fitting is ``d.fit.polyfit``.
Accessor coordinate keywords are ``x`` and ``y``; other fit options and
algorithm-specific return conventions are retained.

Direct fitting methods remain available without new deprecation warnings;
their removal release has not been selected. Unlike the removed plotting
methods, these calls do not have to be converted immediately. Use the accessor
when updating a script so that it follows the current guide.

For folder-wide fits, replace ``folder.each.curve_fit(Linear, ...)`` with
``folder.each.fit.curve_fit(Linear, ...)``. See :doc:`curve_fitting`,
:doc:`fitting_registry` and :doc:`fitting_folders` for examples, model registration
and output choices.

Replace direct NumPy storage assumptions
========================================

The main storage change from 0.10/0.11 is that Data tables now use pandas,
while images and stacks use xarray. The ``DataArray``, ``ImageArray`` and
``KerrArray`` subclasses have been removed. Work with ``Data``, ``ImageFile``
and ``KerrImageFile`` as the owners of measurements, masks and metadata.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Interface in 0.12
     - Meaning
   * - ``d.data``
     - Stored pandas DataFrame of raw measurements.
   * - ``im.image`` or ``im.data``
     - Native xarray Dataset containing intensities, exclusions and coordinates.
   * - ``stack.stack``
     - Native xarray Dataset including frame coordinates and valid image extents.
   * - ``obj.to_numpy()``
     - Detached, writable NumPy masked array; edits do not update the owner.
   * - ``d.asarray()`` or ``im.imarray``
     - Detached, read-only numerical snapshot.
   * - ``obj.export_storage()``
     - Lossless state export including exclusions and typed metadata.

For example, replace positional array access such as ``d.data[:, 1]`` with
``d.column(1)`` or ``d.to_numpy()[:, 1]``. Native pandas indexing uses
``d.data.iloc[:, 1]`` and follows pandas rules, including its handling of labels
and alignment.

Native pandas and xarray calculations do not automatically apply Stoner's
exclusions. Use Stoner methods or masked NumPy exports when excluded values
must be ignored. NumPy results are ordinary arrays and no longer carry Stoner
row or column annotations through slicing and arithmetic.

For a calculation that should update the original measurement, use an editing
block rather than modifying an exported array::

    with d.edit_numpy() as values:
        values[:, 1] *= 2

Changes are committed only when the block finishes successfully.
``d.edit_pandas()`` and image/stack ``edit_xarray()`` provide corresponding
native interfaces. Keep shape, data type, labels and ordering unchanged inside
editing blocks; use Stoner methods for structural changes.

Obtain native references again after changing an owner with Stoner methods.
Saved DataFrames or Datasets may no longer represent its current storage.
Do not edit the owner separately while an editing block is open, or rely on
borrowed frame/crop Datasets writing back to the original image or stack.
See :doc:`data_interchange` and :doc:`image_interchange` for the practical rules.

``d.head(n)`` and ``d.tail(n)`` now return independent Data objects, retaining
excluded measurements, masks, column roles and metadata. Zero gives an empty
table; negative counts omit rows from the opposite end. ``d.sort(...)`` still
sorts in place and returns ``d``; ``d.unique(...)`` retains NumPy's sorted masked
results rather than pandas' distinct-value semantics.

Check column and row selectors
==============================

Column names can now resolve to all matches rather than silently choosing one.
This includes repeated exact names. Matching tries exact equality, then glob
or literal substring matching, then regular expressions. Use a compiled regular
expression when you specifically want regex matching.

If your code needs exactly one column, use an unambiguous name or deliberately
choose the first result with ``d.find_col(selector, first=True)``. Scalar
consumers reject ambiguous selectors; methods supporting multiple columns use
all matches. ``force_list=True`` always returns a list. See :doc:`data_columns`.

For row searches, Boolean lists and short Boolean arrays select positions.
Two-value numeric NumPy arrays select two discrete targets. Use a tuple for an
interval; two-item numeric lists retain their older interval meaning. Search
results preserve exclusions when returning multiple columns.

``where`` selects observations for ``d.fit.curve_fit``, ``im.mean`` and
``stack.average`` without changing the persistent mask. True means include.
Unsupported numerical transforms reject this keyword rather than silently
ignoring it. To pass a callback's own argument named ``where``, use
``d.apply(callback, _extra={"where": value})``. ``d.select(where=value)``
continues to query a column named ``where``. See :doc:`data_selecting`,
:doc:`fitting_simple` and :doc:`image_regions`.

Update image scripts
====================

Replace direct construction of the removed array subclasses with owner classes::

    from Stoner import ImageFile
    from Stoner.Image.kerr import KerrImageFile

Call processing methods on those objects. ``im.image`` now exposes a Dataset,
so use ``im.to_numpy()`` for a writable numerical copy or ``im.edit_numpy()``
to commit pixel and mask changes. Ordinary rectangular slicing returns an
independent ImageFile. The ``crop`` method has separate copying controls;
see :doc:`image_regions` before relying on shared pixels.

For scripts retaining older image-method names, replace ``box`` and
``crop_image`` with ``crop``, ``convert_float`` with ``asfloat``, and
``convert_int`` with ``asint``.

The new ``im.plot()`` accessor returns Axes, consistently with Data plotting.
The separate legacy ``im.imshow()`` method retains its Figure return. See
:doc:`image_plotting` before changing code that uses the return value.

ImageFolder retains the folder interface; ImageStack now stores pixels, masks
and valid frame extents together in an xarray Dataset. Padding for differently
sized frames stays excluded. See :doc:`image_collections` for stack editing.
The optional ``im.roi`` and ``im.fit`` accessors add region selection and
background fitting; see :doc:`image_roi` and :doc:`image_fitting` when adopting
those workflows.

Update folder operations
========================

Use ``each`` to apply an operation to stored measurements. If an older script
calls ``folder.curve_fit(Linear)``, replace it with
``folder.each.fit.curve_fit(Linear)``. Operations that organise the collection,
such as ``folder.group(...)`` and ``folder.sort(...)``, remain on the folder.

Each bulk call returns a list of results, rather than a folder for chaining.
A returned Data object replaces its corresponding member; other results are
collected in the list. Failed calls leave an exception and traceback in that
member's result entry. Check these results before using a batch summary.

Instead of manually looping to build a table of metadata, use::

    result = folder.metadata.slice(["Temperature", "Resistance"], output="Data")

This operates on direct members by default. Use ``recurse=True`` to include
members throughout a grouped folder. See :doc:`folder_each`,
:doc:`folder_metadata` and :doc:`folder_grouping` for the current workflows.
