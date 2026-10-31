Working with Lots of Files
**************************

A DataFolder holds a collection of measurements so that you can select files,
organise them into groups and apply the same analysis to each one. It can list
files on disc and load them when needed, or hold Data objects created in Python.
These short guides introduce one task at a time.

Try a small collection
======================

Run this example in a Python console or notebook. It creates two measurements
without reading any files::

    import numpy as np
    from Stoner import Data, DataFolder

    f = DataFolder(readlist=False)
    field = np.array([-1.0, 0.0, 1.0])
    for temperature, resistance in [(4.2, [1.1, 1.0, 1.1]),
                                    (77.0, [2.6, 2.4, 2.6])]:
        d = Data(field, np.array(resistance),
                 column_headers=["Field", "Resistance"], setas="xy")
        d.filename = f"measurement_{temperature:g}K.dat"
        d["Temperature"] = temperature
        f += d

    print(len(f))                    # Two measurements.
    print(f[0].column_headers)        # Field and Resistance.
    print(f.metadata["Temperature"])  # One temperature per measurement.
    spans = f.each.span("Resistance") # One (minimum, maximum) pair per file.
    print(spans)

Keep using this collection as ``f``, or replace it with
``f = DataFolder("my_measurements", pattern="*.dat")`` to list your own files.
The Data objects are the measurements introduced in :doc:`datafile`.

Choose your next task
=====================

Read the first two pages to get started, then choose a guide for your task.

.. toctree::
   :maxdepth: 1

   Build folders and load files <folder_loading>
   Inspect, sort and select members <folder_selecting>
   Apply operations to every file <folder_each>
   Read and collect metadata <folder_metadata>
   Group folders and manage their hierarchy <folder_grouping>
   Combine measurements and process groups <folder_reducing>

Many operations change the collection or its stored measurements. Use
``working = f.clone`` before experimenting when you need an independent copy.
The ``each`` helper returns a list of results, so check it for individual
failures when processing a batch.

For paginated plots, see :doc:`plot_folders`; for bulk curve fitting, see
:doc:`fitting_folders`. Most folder operations also work with ImageFolder,
holding ImageFile objects instead of Data objects. See :doc:`image_collections`
for image folders and stacks, including their shared pixel storage.

For collections stored in archives, the package also provides
:class:`~Stoner.folders.zip.ZipFolder` and
:class:`~Stoner.folders.hdf5.HDF5Folder`.
