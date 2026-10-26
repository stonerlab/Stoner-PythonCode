Working with numerical data
***************************

A Data object keeps a table of measurements together with column names,
information about the experiment (metadata), and flags for excluded values
(a mask). These short guides introduce one task at a time. You only need basic
Python variables, function calls and lists to get started.

Try a small measurement
=======================

Run this example in a Python console or notebook. It creates a measurement
without downloading a file::

    import numpy as np
    from Stoner import Data

    temperature = np.array([4.2, 10.0, 77.0, 300.0])
    resistance = np.array([1.0, 1.2, 2.4, 5.0])
    uncertainty = np.array([0.05, 0.05, 0.10, 0.20])
    d = Data(temperature, resistance, uncertainty,
             column_headers=["Temperature", "Resistance", "Uncertainty"],
             setas="xye")
    d["Sample"] = "Example wire"  # Metadata describes the experiment.

    print(d.shape)                # Four rows and three columns: (4, 3).
    print(d.head(2))              # Preview the first two rows.
    print(d.column("Temperature"))

``setas="xye"`` identifies the coordinate, signal and signal uncertainty.
The example uses ``np.array(...)`` to create numerical arrays; each array
becomes one column. Keep using this object as ``d`` while reading the guides,
or replace it with ``d = Data("my_measurement.txt")`` to load your own file.

Choose your next task
=====================

If you are new to Stoner, read the first three pages in order. Later pages
cover particular tasks and can be read when you need them.

.. toctree::
   :maxdepth: 1

   Load and save measurements <data_loading>
   Inspect measurements and metadata <data_inspecting>
   Choose columns and assign roles <data_columns>
   Read rows, cells and slices <data_indexing>
   Select rows and mask measurements <data_selecting>
   Edit and combine measurements <data_editing>
   Work with pandas and NumPy <data_interchange>

Two habits will help you avoid surprises:

* Row and column positions start at zero. A slice such as ``d[:3]`` reads the
  first three rows and stops before row 3.
* Many editing and analysis methods change ``d`` itself. Use ``working = d.clone``
  to keep an independent copy before experimenting.

When you are ready, continue with :doc:`plotfile`, :doc:`curve_fitting` or
:doc:`analysisfile`. For developer and storage details, see :doc:`developer`.
