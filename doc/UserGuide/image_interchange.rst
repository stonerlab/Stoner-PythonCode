Work with NumPy and xarray images
*********************************

.. currentmodule:: Stoner.Image

Use the image ``im`` from :doc:`image`. Choose a detached export for
calculations, or an editing block to commit changes to the image.

``ImageFile`` owns an xarray Dataset containing intensities, exclusions and
coordinates. ``image`` and ``data`` return the native Dataset; ``imarray`` returns
a detached read-only masked array and ``to_numpy()`` a writable detached export. Use NumPy routines
directly with an image when only its values are needed. Use ``to_numpy()`` when
the exclusion mask is needed, and ``export_storage()`` to preserve calibration
and typed metadata as well.

The old ``ImageArray`` and ``KerrArray`` subclasses have been removed. Construct
``ImageFile`` or ``KerrImageFile`` and call scientific methods on those objects.
NumPy calculations on these copies do not carry Stoner metadata into their
results and do not update the original image.
Write through image indexing, replace ``image`` as a whole, or use an editing
block to apply numerical results::

    with im.edit_numpy() as pixels:
        pixels[0, 0] = 12
        pixels.mask[1, 2] = True

The changes are applied only when the block finishes successfully. Keep its
shape and data type unchanged. ``edit_xarray()`` provides a similar block for
experienced xarray users and also preserves coordinates. Do not edit the
original image separately while one of these blocks is open.
