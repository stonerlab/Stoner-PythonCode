Histograms and statistical plots
********************************

.. currentmodule:: Stoner

Use a histogram to see how often values fall in each range, a box plot to
summarise their spread, or a density curve to estimate their distribution.
For example, with the measurement in :doc:`plotfile`::

    d.plot.hist(y="Signal", bins=5)
    d.plot.show()

``y`` identifies the column of samples. A hexagonal binning plot instead
groups pairs of X/Y values into coloured cells, which is useful for dense
scatter data. The detailed examples below use other table layouts; replace
their column selectors with those in your own measurement.

.. contents:: On this page
   :local:
   :depth: 1

.. _plotting-statistics:

Distributions and hexagonal binning
===================================

``hist``, ``box``, ``kde`` and ``hexbin`` use Matplotlib directly, with SciPy's
``gaussian_kde`` for density estimation. Each returns its primary Axes, respects
``ax=`` and current styling, and does not show/save. These methods apply Stoner's masks independently of pandas plotting settings.

For ``hist``, ``box`` and ``kde``, ``y`` selects sample columns rather than XY
pairs. Omitting it selects assigned Y columns in the current ``startx`` group;
if none exist, all real numeric columns are selected. A single-column table
therefore works. ``no_guess=True`` disables that numeric fallback. Explicit
selectors use Stoner's normal positional/header/pattern/slice rules and preserve
selection order, including duplicate-header behaviour::

    ax = d.plot.hist(y=[1, 3], bins=30, histtype="step")
    ax = d.plot(kind="hist", y=[1, 3], bins=30, histtype="step")
    ax = d.plot.box(y=[1, 3], orientation="horizontal")
    ax = d.plot.kde(y=1, bw_method="silverman", ind=300)

Each sample column removes its own excluded/non-finite values independently.
Masks in unselected X/error columns do not remove distribution samples;
no separate X coordinate is read for alignment. Every selected series must retain
samples, otherwise the call raises before creating axes. Integer samples are filtered without inserting
NaNs into the measurement table. All methods leave values, exclusions, metadata,
column assignments unchanged.

``hist`` forwards native options such as ``bins``, ``range``, ``density``,
``cumulative``, ``histtype`` and ``stacked``. Selected series share bin edges.
``stacked=True`` adds histogram values; use ``StackVertical`` to arrange panels.
For histograms and KDEs, ``weights`` is one column selector or an array with one
entry per original row. Invalid weights combine with each series' exclusions,
preserving alignment independently. Histograms retain native support for signed
weights. KDE weights must be non-negative with positive total weight.

``box`` passes independent sample arrays to Matplotlib; it has no weights or
group-by forwarding. ``label`` supplies category labels and ``orientation`` is
``"vertical"`` or ``"horizontal"``. Native controls such as ``whis``,
``showfliers``, ``notch``, ``positions`` and artist-property dictionaries remain
available. The accessor handles the native orientation/tick-label keyword names
across the supported Matplotlib versions.

``kde`` requires at least two distinct samples with positive weight per series.
``bw_method`` uses SciPy's factor/name/callable convention; None uses Scott's
rule. ``ind`` is an integer of at least two points or a finite evaluation vector.
By default each series is evaluated at 1000 points extending half its observed
range beyond each end; zero-weight samples do not determine that range. Gaussian
kernels extend beyond the measured range, including physical bounds. Remaining
styling arguments go to ``Axes.plot``.

Distribution ``label`` values default to column labels; explicit ``None``
suppresses legend/category labels. Histograms and KDEs update their legend by
default; ``legend=False`` disables that update. ``xlabel``, ``ylabel`` and
``title`` retain source/metadata interpolation and existing text unless explicitly
overridden; ``None`` clears an annotation. Unsupported ``backend``, ``by``,
``subplots``, ``layout`` and ``secondary_y`` options raise an error; use the
stonerplots contexts for layout and axes management.

``hexbin`` uses one X and one Y column, with the ordinary scientific coordinate
defaults. ``C=None`` counts jointly valid points; a column selector or source-row
array supplies values to aggregate with ``reduce_C_function`` (mean by default).
A point is used only when its X, Y and optional C values are unmasked and finite. Assigned uncertainties and
unrelated masks are ignored. Native ``gridsize``, ``mincnt``, ``extent``,
``bins``, ``cmap``, ``norm`` and ``xscale``/``yscale`` controls are supported,
including logarithmic binning. ``colorbar=True`` adds a colour scale without
replacing the primary recent target::

    ax = d.plot.hexbin(x=0, y=1, gridsize=25, mincnt=1)
    ax = d.plot.hexbin(x=0, y=1, C=3, reduce_C_function=sum, colorbar=False)

.. plot:: samples/statistical_plots.py
    :include-source:
    :outname: statistics

Next steps
==========

See :doc:`plot_formatting` for labels and saving, or :doc:`plot_layouts`
to compare distributions in separate panels. Return to :doc:`plotfile`.
