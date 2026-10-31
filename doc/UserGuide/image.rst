Working with Images
*******************

An ImageFile keeps a two-dimensional picture together with information about
the measurement (metadata) and flags for pixels to exclude (a mask). These
short guides introduce one task at a time.

Try a small image
=================

Run this example in a Python console or notebook. It creates a small intensity
image without downloading a file::

    import numpy as np
    from Stoner import ImageFile

    pixels = np.arange(12, dtype=float).reshape(3, 4)
    im = ImageFile(pixels)
    im["Sample"] = "Example film"

    print(im.shape)        # Three rows and four columns: (3, 4).
    print(im["Sample"])    # Metadata describes the measurement.
    print(im.mean())       # Mean intensity: 5.5.
    section = im[:, 1:3]   # An independent image of the middle two columns.
    ax = im.plot(cmap="viridis")
    im.plot.show()

Keep using this object as ``im``, or replace it with
``im = ImageFile("my_image.png")`` to load your own image.

Choose your next task
=====================

Read the first two pages to get started, then choose a guide for your task.

.. toctree::
   :maxdepth: 1

   Load and save images <image_loading>
   Inspect images, pixels and metadata <image_inspecting>
   Plot images and histograms <image_plotting>
   Select image regions <image_roi>
   Measure, crop and align images <image_regions>
   Filter images and analyse spatial frequencies <image_processing>
   Fit image backgrounds <image_fitting>
   Work with NumPy and xarray images <image_interchange>
   Work with image folders and stacks <image_collections>

Many processing methods change ``im`` itself. Use ``working = im.clone`` to
keep an independent copy. A mask excludes pixels from calculations without
deleting their values; see :doc:`image_roi` for how masks and regions interact.

For collection operations shared with numerical data, continue with
:doc:`datafolder`. For storage and extension details, see :doc:`developer`.
