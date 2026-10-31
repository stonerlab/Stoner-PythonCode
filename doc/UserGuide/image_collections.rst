Work with image folders and stacks
**********************************

.. currentmodule:: Stoner.Image

For listing, filtering, grouping and applying methods to collections,
start with :doc:`datafolder`. The same folder interface works with images.

Choose an ImageFolder for a collection of separate ImageFile objects, or an
ImageStack to store pixels and masks together. For example, to list TIFF images::

    from Stoner import ImageFolder

    images = ImageFolder("my_images", pattern="*.tif")
    first = images[0]  # Load the first ImageFile when it is needed.

Just as :class:`Stoner.folders.mixins.DataFolder` allows you to efficiently process lots of separate :class:`Stoner.core.data.Data` files, :class:`ImageFolder` does the same for lots
of :class:`ImageFile` objects. It is based on the same parent :class:`Stoner.folders.core.BaseFolder` class, so has similar abilities to iterate, form
subfolders and so on. In addition, an :class:`ImageFolder` has attributes and methods for working with multiple images.

Due to the potentially large amount of data involved in processing images it is good to take advantage of native numpy's speed wherever possible. To this end
:class:`Stoner.Image.ImageStack` is now available. This works very similarly to ImageFolder but owns an xarray Dataset with frame, y and x dimensions and valid image extents.
For example::

    from Stoner import ImageStack

    imst = ImageStack('pathtomyfolder', pattern='*.tif')
    imst = imst['subfolder']  # Take advantage of folder grouping
    imst.each.translate((5, 3))  # Translate all images

The ``imarray`` property returns an independent, read-only array indexed as
``[frame, row, column]``. Use edit_numpy() for committed edits, or call functions on each ImageFile
(including the scikit-image and scipy library).

``imst.stack`` is the stored xarray Dataset. Its ``intensity`` and ``excluded``
variables hold pixel values and masks. ``valid_height`` and ``valid_width`` give
each frame's actual size; padding outside that size must stay masked. Frame
coordinates identify the stored images.

For everyday edits, use the ImageFile for a frame, or an editing block::

    with imst.edit_numpy() as pixels:
        pixels[0, 0, 0] = 12  # First frame, first row, first column.

Direct xarray cell edits are available for experienced users, but must retain
the dimensions, coordinates, data type, variables and valid sizes. Assigning a
whole Dataset requires the same structure and copies its input. Use storage
imports for a different structure; see :doc:`developer`.

Obtain the Dataset again after changing an image or stack with Stoner methods:
an old reference may no longer contain the current data. To edit a frame or
crop, assign through that ImageFile or its editing block. Do not rely on edits
to a saved frame/crop Dataset reaching the original.

For selected-pixel stack averages, see :doc:`image_regions`.
