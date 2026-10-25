.. _User_Guide:

#####################################
 The Stoner Python Package User Guide
#####################################

.. rubric:: C.S. Allen, M. Newman, R. Temple, S. Morely  and G. Burnell


Start with :doc:`install`, then :doc:`datafile` for loading a measurement,
choosing columns and working with metadata. The plotting and fitting chapters
build on those basics. Use the folder and image chapters when working with
collections or pictures, and the cookbook for short practical recipes.

Examples use Python 3. Replace illustrative filenames and column names with
those in your own measurements. Unless an example says otherwise, ``d`` or ``a``
is a Data object. Methods often change that object: use ``d.clone`` to keep an
independent copy before experimenting.

This guide describes the current development API. If an older script behaves
differently, consult :doc:`upgrade`. The :doc:`developer` chapter contains the
implementation and contribution details; it is not needed for everyday use.

.. toctree::
   :maxdepth: 3

   Introduction and Installation <install>
   Loading and Examining Data <datafile>
   Plotting Data <plotfile>
   Curve Fitting Data <curve_fitting>
   Analysing and Processing Data <analysisfile>
   Working with Lots of Data Files <datafolder>
   Working with Images <image>
   Cookbook <cookbook>
   Converting Scripts from Older Versions <upgrade>
   Developing and Extending <developer>
   Credits <credits>

