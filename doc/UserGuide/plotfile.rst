Plot your first measurement
***************************

Create a small measurement and draw it with ``d.plot()``::

    import numpy as np
    from Stoner import Data

    time = np.array([0.0, 1.0, 2.0, 3.0])
    signal = np.array([0.0, 1.0, 2.5, 4.0])
    d = Data(time, signal, column_headers=["Time / s", "Signal"], setas="xy")
    d.plot()
    d.plot.show()

You should see a line joining four points, with labels taken from the column
names. ``setas="xy"`` tells Stoner which column is the horizontal coordinate
and which is the signal. In a notebook, putting ``d.plot()`` at the end of a
cell normally displays the result; in a script, use ``d.plot.show()`` as above.

Plot your own data
==================

Replace the example measurement with a file from your experiment::

    d = Data("my_measurement.txt", setas="xy")
    d.plot()
    d.plot.show()

Use ``setas="xye"`` for columns containing a coordinate, signal and signal
uncertainty. Stoner then draws error bars automatically. See :doc:`data_columns`
if your columns have a different order, or :doc:`data_loading` for loading options.

Change a label or save the picture
==================================

After drawing, you can change the title and save a picture::

    d.plot.title = "My measurement"
    d.plot.figure.savefig("measurement.png")

This writes the picture to Python's current working directory. It does not save
the numerical measurement; use :doc:`data_loading` for that.

Choose your next task
=====================

.. toctree::
   :maxdepth: 1

   Lines, scatter and uncertainty plots <plot_lines>
   Labels, limits and saving plots <plot_formatting>
   Histograms and statistical plots <plot_statistics>
   Images, surfaces and vector fields <plot_fields>
   Panels, styles and independent scales <plot_layouts>
   Plot collections of measurements <plot_folders>
   Worked layout and style recipes <plot_recipes>
   Update older plotting scripts <plot_migration>

The formatting guide explains Figures and Axes when you need that terminology.
For full method signatures, use the :doc:`module reference <../Stoner>`.
