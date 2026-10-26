Labels, limits and saving plots
*******************************

.. currentmodule:: Stoner

Once you have drawn a measurement, use ``d.plot`` to change its labels and
limits. For example, after the recipe in :doc:`plotfile`::

    d.plot.xlabel = "Time / s"
    d.plot.ylabel = "Signal"
    d.plot.title = "My measurement"
    d.plot.ylim = (0, 5)
    d.plot.figure.savefig("measurement.png")

``savefig`` writes the picture, including labels and styling. To save the
underlying numerical measurement, use ``d.save(...)`` instead; see
:doc:`data_loading`.

.. contents:: On this page
   :local:
   :depth: 1

Convenient plot properties
==========================

Once a plot exists, public properties on ``d.plot`` forward to Matplotlib's
``get_...`` and ``set_...`` methods. The accessor's most recent live Axes takes
priority, followed by its Figure. Existing accessor attributes and methods,
including ``labels``, ``axes`` and ``figure``, retain their usual meaning::

    ax = d.plot()
    d.plot.xlabel = "Field / T"
    d.plot.xlim = (-1, 1)
    d.plot.dpi = 150                 # Figure property
    assert d.plot.xlabel == "Field / T"
    print(d.plot.xlim)               # Readable current limits

Reading a forwarded property returns a callable value proxy. Calling without
arguments returns the ordinary getter value; calling with arguments invokes the
setter and returns its native result. Assignment passes one value, without
unpacking tuples or dictionaries. Calls support the full Matplotlib signatures::

    label = d.plot.xlabel()          # Ordinary Python string
    d.plot.title("Run 1", loc="left", color="red")
    d.plot.xlim(left=-0.5, right=0.5)
    d.plot.size_inches(7, 4, forward=False)

Use explicit ``.get(...)`` and ``.set(...)`` calls when a getter needs arguments,
or a setter needs none. Keyword arguments on the shorthand call always select
the setter::

    left_title = d.plot.title.get(loc="left")
    d.plot.xticks.set([0.25, 0.75], labels=["Quarter", "Three quarters"], minor=True)
    renderer = ax.figure.canvas.get_renderer()
    bounds = d.plot.tightbbox.get(renderer)

The proxy supports representation, equality, formatting, indexing, iteration and
NumPy conversion. Use ``.get(...)`` or a no-argument call for an ordinary value
when doing arithmetic, checking its type or taking a snapshot. A saved proxy
continues to resolve against its owner's most recent plotting targets.

Getters and setters resolve independently. Read-only or write-only Matplotlib
properties remain usable for their supported operation, and native argument and
validation errors propagate. Unknown names and properties without a live target
raise ``AttributeError``. Lookup, assignment and completion never create a Figure
or Axes, select a pyplot target or fall back to an unrelated current figure.
After ``d.plot.new_figure()``, Figure properties work even before adding Axes.
These conveniences belong to ``d.plot``; ``d.xlabel`` remains unsupported.

Understanding Figures and Axes
==============================

A *Figure* is the whole picture, while an *Axes* is one plotting area
within it. The value returned by ``ax = d.plot()`` is that plotting area.
Use ``ax`` for Matplotlib formatting methods and ``ax.figure`` for the
whole picture. One Figure can contain several Axes.

The ``Data.plot`` property returns a callable
:py:class:`~Stoner.plot.accessor.DataPlotter`: calling it draws a plot,
while its methods and properties provide further plotting operations.
Its full API is documented in the :doc:`module reference <../Stoner>`.

Choose a plotting target and format it
======================================

Ordinary plots use explicit ``ax=`` or Matplotlib's current compatible axes. They
create a figure only when needed; a new Data object does not imply a new figure.
An incompatible implicit projection creates a suitable figure, while an
incompatible explicit axes raises an error::

    ax = d.plot()
    other.plot.line(x=0, y=1, ax=ax)
    figure = d.plot.new_figure(figsize=(7, 4))
    ax = d.plot.line()

``d.plot.ax``, ``d.plot.figure`` and ``d.plot.axes`` inspect recent live targets
without selecting or creating a figure. Obtain the Figure from ``ax.figure``.
Copy, deepcopy, clone, construction from another Data, and pickle round trips
preserve label configuration while detaching live figures/axes.

Drawing inherits current Matplotlib settings (``rcParams``) and existing axes styling. It does not reset
styles or apply a legacy template. Use the returned Axes for formatting::

    ax = d.plot(fmt="o", xlabel="Field / T", ylabel="Signal", title="Run 1")
    ax.set_xscale("log")
    ax.set_ylim(0, 10)
    ax.legend(loc="upper left")

``xscale`` and ``yscale`` are also accessor arguments. ``d.plot.labels`` provides
mutable column-label overrides, defaulting to column headers. Plain text labels,
titles and styling strings retain metadata ``{name}`` interpolation; strings
containing ``$`` retain their mathematical markup.

Omitted annotations fill empty labels/titles; subsequent traces do not overwrite
existing text. Explicit values override it; ``None`` clears it. ``label=None``
suppresses that series' legend entry and ``legend=False`` stops automatic legend
updates. User formatters and locators are left alone.

Next steps
==========

Use :doc:`plot_layouts` for several panels or a temporary style.
Return to :doc:`plotfile` to choose another plotting task.
