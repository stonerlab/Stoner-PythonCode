**************************
Working with Images
**************************
.. currentmodule:: Stoner.Image.core

Introduction
============

The :mod:`Stoner.Image` package provides a means to carry out image processing functions in a smilar way that :mod:`Stoner.Core` and :class:`Stoner.Data` and
:class:`Stoner.DataFolder` do. The :mod:`Stomner.Image.core` module contains the key classes for achieving this.

:class:`ImageFile`
===================
This class inherits from :class:`Stoner.Core.metadataObject` and is analagous to :class:`Stoner.Data` but focussed on handling and manipulating grey scale
images. The data is stored internally as an :class:`ImageArray` attribute which inherits itself from numpy.ndarray type.
:class:`ImageFile` provides the same metadata faciltities as the general :class:`Stoner.Data` and also contains load routines that allow it to extract
certain metadata stored in image files.

Loading an Image
----------------

The :class:`ImageFile` constructor supports taking a string argument which is interpreted as a filename of an image format recognised by PIL. The resulting
image data is used to form the contents of the :attr:`ImageFile.image` which holds the image data.

   from Stoner.Image import ImageFile
   im = ImageFile('my_image.png')

Like :class:`Stoner.Data` :class:`ImageFile` supports image metadata. Where this can be stored in the file, e.g. in png and tiff images, this is read in
automatically. This metadata is stored as a :class:`Stoner.Core.typeHintedDict` dictionary. This metadata can be set directly in the
construction of the :class:`ImageFile`::

   im = ImageArray(np.arange(12).reshape(3,4), metadata={'myarray':1})

Examining and manipulating the ImageArray
-----------------------------------------

Local functions and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typical start functions might be to convert the image into floating point form which is more precise than
integer format and crop the image::

  im.asfloat()
  im.crop(1,8,40,200) #(xmin, xmax, ymin, ymax)
  copyofim = im.clone #behaviour similar to data file

#get a metadata item::

  im['mymeta'] = 5

Then view parts of the array::

	im[:,10:50]
	im.metadata.keys()
	im.view()

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

While many local analysis functions have been added to ImageFile one of the big benefits is that function proxy methods have been added to make the entire
scikit-image library and scipy routines available. Function requests will first search local image functions and secondly look up any function from the external
libraries. The proxy will pass the :attr:`ImageFile.image` attribute as the first argument to any external call to a scikit-image or scipy function. The return value for such
calls is handled a bit carefully:

    #. If the return value is a 2d numpy.ndarray like type that has the same size as the original, or if the *_* keyword argument is set to **True** then the original
        :attr:`ImageFile.image` is replaced with the returned result.
    #. If the return value is anything else then it is simply passed back to the calling program.

In this way, many operations can be carried out 'in-place' on a :class:`ImageFile`. For example::

ImageFile Representation
------------------------

By default, the representation of an ImageFile is just a short textual description, however if the *short_repr& and *short_img_repr* options
are both set to False and a graphical console is in use with an ipython kernel, then th special _repr_png_ method will show a picture of the
contents of the ImageFile instead.::

    i = Stopner.Image.ImageFile("kermit.png")
    i
    >>> kermit.png(<class 'Stoner.Image.core.ImageFile'>) of shape (479, 359) (uint16) and 53 items of metadata
    from Stoner import Options
    Options.short_repr=False
    Options.shoft_img_repr=False
    i
    >>>

.. image:: ../../sample-data/kermit.png

Alternatively the :meth:`ImageArray.imshow` method (accessible to :class:`ImagerFile`) will show the image data in a matplotlib window.

:class:`ImageArray`: A numpy array like class
=============================================

Somewhat analogous to :class:`Stoner.Core.DataArray`, the :class:`ImageArray` is a specialised subclass of :class:`numpy.ma.MaskedArray` used to
store the image data in ImageFile. The numpy.ndarray like data can be accessed at any point via either :attr:`ImageFile.image` or :attr:`ImageFile.data`
and will be accepted by functions that take an numpy.ndarray as an argument.

Working with Lots of Images: :class:`ImageFolder` and :class:`ImageStack2`
==========================================================================

Just as :class:`Stoner.DataFolder` allows you to efficiently process lots of separate :class:`Stoner.Data` files, :class:`ImageFolder` does the same for lots
of :class:`ImageFile` files. It is based on the same parent :class:`Stoner.Fodlers.baseFolder` class - so has similar abilities to iterate, form into
sub-folders and so on. In addition, an :class:`Imagefolder` has additional attributes and methods for working with multiple images.

Due to the potentially large amount of data involved in processing images it is good to take advantage of native numpy's speed wherever possible. To this end
:class:`Stoner.Image.ImageStack2` is now available. This works very similarly to ImageFolder but internally represents the image stack as a 3d numpy array.
For example::
	imst = ImageStack2('pathtomyfolder', pattern='*.tif') #directory is held in memory but images are not loaded yet
	imst = imst['subfolder'] #take advantage of :class:`DiskBasedFolder` grouping abilities
	imst.translate(5,3) #instantiate the stack and translate all images

You can request and manipulate this 3d array directly with the imarray property, alternatively you can ask for any function accepted by the underlying ImageFile
(including the scikit-image and scipy library).



