**************************
Working with Images
**************************
.. currentmodule:: Stoner.Image.core

Introduction
============

The :mod:`Stoner.Image` package provides a means to carry out image processing functions in a smilar way that :mod:`Stoner.Core` and :class:`Stoner.Data` and
:class:`Stoner.DataFolder` do. The :mod:`Stomner.Image.core` module contains the key classes for achieving this.

:class:`ImageArray`: A numpy array like class
=============================================

Somewhat analogous to :class:`Stoner.Core.DataArray`, the :class:`ImageArray` is a specialised subclass of :class:`numpy.ma.MaskedArray` with additional
attributes to assist in manipulating image data. The main additional feature of an :class:`ImageArray` are described int he following sections.

Loading an Image
----------------

The :class:`ImageArray` constructor supports takling a string argument which is interpreted as a filename of an image format recognised by PIL. The resulting
image data is used to form the contents of the :class:`ImageArray`.

   from Stoner.Image import ImageArray
   im = ImageArray('my_image.png')

In addiiton, :class:`1ImageArray` supports metadata about the image. Where this can be stored in the file, e.g. in png and tiff images, this is read in
automatically. Like :class:`Stoner.Data`, the metadata as a :class:`Sonter.Core.typeHintedDict` dictionary. This metadata can be set directly in the
construction of the :class:`ImageArray`::

   im2 = ImageArray(np.arange(12).reshape(3,4), metadata={'myarray':1})

ImageArray inherits from (ultimately) :class:`numpy.ndarray` and can be used in much the same way as a normal array. On top
of this are added specific image functions

Examining and manipulating the ImageArray
-----------------------------------------

Local functions and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typical start functions might be to convert the image into floating point form which is more precise than
integer format and crop the image::
  
  im = im.asfloat()
  im = im.crop_image(1,8,40,200) #(xmin, xmax, ymin, ymax)
  copyofim = im.clone #behaviour similar to data file

#et a metadata item::

  im['mymeta'] = 5
  
Then view parts of the array::

	im[:,10:50] 
	im.metadata.keys()
	im.imshow()
	
Note that as with numpy arrays im[:,10:50] is a view onto the same memory space, so if you change stuff with this it will 
change in the ImageArray.

Further functions
^^^^^^^^^^^^^^^^^

Further functions that could be useful:
  
  - im.threshold_minmax(0.2,0.8)
      Returns a binary image
  - im.plot_histogram() 
      Plot a histogram of the pixel intensities
  - im.level_image() 
      Flatten a skewed image background
  - im.subtract_image(otherim)
      Subtract another image and enhance contrast
  - im.align(otherim)
      Translate image to line up with other im
  
Scikit-Image & scipy.ndimage Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`ImageArray` also features a special feature that lets you use a variety of image processing function from :mod:`scipy.ndimage` and the
scikit image library as if they were built in methods. For example::

    im.threshold_otsu(nbins=300) #pass through to skimage.filters.threshold_otsu
    
:class:`ImageFile`
===================

Whilst :class:`ImageArray` represents an image with a subclass of a :class:`numpy.ndimage`, :class:`ImageFile` acts more like :class:`Stoner.Data`, ands indeed
shares a common ancestor: :class:`Stoner.Core.metadataObject`. This provides the same metadata faciltities as the general :class:`Stoner.Data`, but uses a
:class:`ImageArray` to store the data in. This is accessed via either :attr:`ImageFile.image` or :attr:`ImageFile.data`.

For maximum convenience, :class:`ImageFile` willautomatically pass through to its :attr:`Imagefile.image` :class:`ImageArray` any method calls that it can't
handle and that :class:`ImageArray` can - thus one gets direct access to the comple Scikit-Image and :mod:`scipy.ndimage` tools. The return value for such
calls is handled a bit carefully:

    #. If the return value is an :class:`ImageArray` that has the same size as the original, or if the *_* keyword argument is set to **True** then the original
        :class:`ImageArray` is replaced with the returned result.
    #. If the return value is anything else then it is simply past back to the calling program.
    
In this way, many operations can be carried out 'in-place' on a :class:`ImageFile`.

ImageFile Representation
------------------------

By default, the representation of an ImageFile is just a short textual description, however if the *short_repr& and *short_img_repr* options
are both set to False and a graphical console is in use with an ipython kernel, then th special _repr_png_ method will show a picture of the
contents of the ImageFile instead.::

    i = Stopner.Image.ImageFile("kermit.png")
    i
    >>> kermit.png(<class 'Stoner.Image.core.ImageFile'>) of shape (479, 359) (uint16) and 53 items of metadata
    from Stoner import set_option
    set_option("short_repr",False)
    set_option("shoft_img_repr",False)
    i
    >>> 
    
.. image:: ../../sample-data/kermit.png

Alternatively the :meth:`ImageArray.imshow` method (accessible to :class:`ImagerFile`) will show the image data in a matplotlib window.

Working with Lots of Images: :class:`ImageFolder`
==================================================

Just as :class:`Stoner.DataFolder` allows you to efficiently process lots of separate :class:`Stoner.Data` files, :class:`ImageFolder` does the same for lots
of :class:`ImageFile` files. It is based on the same parent :class:`Stoner.Fodlers.baseFolder` class - so jas similar abilities to itterate, form into 
sub-folders and so on. In addition, an :class:`Imagefolder` has additional attributes and methods for working with multiple images.



