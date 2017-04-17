**************************
Working with Images
**************************


Loading an Image
====================

.. currentmodule:: Stoner.Image.core

Stoner now has classes available to work with data images. The class is inspired by skimage has pass through 
ability to most of the skimage functions. The images also have a metadata parameter similar to DataFile. To
load an image::

   from Stoner.Image import ImageArray
   im = ImageArray('my_image.png')
   im2 = ImageArray(np.arange(12).reshape(3,4), metadata={'myarray':1})

ImageArray is designed to act much like a 2d numpy array but with the extra metadata parameter.

Examining and manipulating the ImageArray
=========================================

Local functions and properties
------------------------------
Typical start functions might be to convert the image into floating point form which is more precise than
integer format and crop the image::
  
  im = im.convert_float()
  im = im.crop_image(box=(1,8,40,200)) #(xmin, xmax, ymin, ymax)
  copyofim = im.clone #behaviour similar to data file
Set a metadata item::

  im['mymeta'] = 5
  
Then view parts of the array::

	im[:,10:50] 
	im.metadata.keys()
	im.imshow()
	
Note that as with numpy arrays im[:,10:50] is a view onto the same memory space, so if you change stuff with this it will 
change in the ImageArray.

Further functions
------------------
Any functions called for that are not in the main class are searched for in imagefuncs and then passed through
to the skimage library.

Further functions that could be useful::
  
  im.threshold_minmax(0.2,0.8) #returns a binary image
  im.plot_histogram() #plot a histogram of the pixel intensities
  im.level_image() #flatten a skewed image
  im.subtract_image(otherim) #subtract another image and enhance contrast
  im.correct_drift(otherim) #translate image to line up with other im
  
skimage functions
-----------------
If we still haven't had a hit the function is passed through to the skimage library and our image
is given in as the first argument. For example::

  im.threshold_otsu(nbins=300) #pass through to skimage.filters.threshold_otsu

