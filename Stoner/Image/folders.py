# -*- coding: utf-8 -*-
"""Implements a baseFolder type structure for working with collections of images."""
__all__ = ["_generator", "ImageFolderMixin", "ImageFolder"]
from .core import ImageArray
from Stoner.Folders import DiskBasedFolder, baseFolder
from Stoner.compat import string_types
from Stoner.Image import ImageFile

from skimage.viewer import CollectionViewer
import numpy as np


def _load_ImageArray(f, **kargs):
    """Simple meothd to load an image array."""
    kargs.pop("img_num", None)
    return ImageArray(f, **kargs)


class _generator(object):

    """A helper class to iterator over ImageFolder yet remember it's own length."""

    def __init__(self, fldr):
        """Initialise the generator object.

        Args:
            fldr (ImageFolder): The folder that we iterate over.
        Returns:
            None.
        """
        self.fldr = fldr
        self.len = len(fldr)

    def __len__(self):
        """Get the length of the iterator

        Returns:
            int: Length of the generator object.

        """
        return self.len

    def __iter__(self):
        """Return an iterator object.

        Returns:
            __generator class: This is its own generator object.
        """
        self.ix = 0
        return self

    def __next__(self):
        """Iterator accessor.

        Raises:
            StopIteration: Done iterating through the folder.

        Returns:
            ret (2D array): Image data.
        """
        if self.ix < len(self):
            ret = self[self.ix]
            self.ix += 1
            return ret
        else:
            raise StopIteration("Finished iterating Folder.")

    def __getitem__(self, index):
        """Item accessor method

        Args:
            index (int): Image to return.

        Returns:
            ret (2D array): Image array data.
        """
        ret = self.fldr[index]
        if hasattr(ret, "image"):
            ret = ret.image
        return ret

    def next(self):
        """Iterate to next value."""
        return self.__next__()


class ImageFolderMixin(object):

    """Mixin to provide a folder object for images.

    ImageFolderMixin is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

        Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`) the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)

        extra_args (dict): Extra arguments to use when instantiatoing the contents of the folder from a file on disk.

        pattern (str or regexp): A filename globbing pattern that matches the contents of the folder. If a regular expression is provided then
            any named groups are used to construct additional metadata entryies from the filename. Default is *.* to match all files with an extension.

        read_means (bool): IF true, additional metatdata keys are added that return the mean value of each column of the data. This can hep in
            grouping files where one column of data contains a constant value for the experimental state. Default is False

        recursive (bool): Specifies whether to search recurisvely in a whole directory tree. Default is True.

        flatten (bool): Specify where to present subdirectories as spearate groups in the folder (False) or as a single group (True). Default is False.
            The :py:meth:`DiskBasedFolder.flatten` method has the equivalent effect and :py:meth:`DiskBasedFolder.unflatten` reverses it.

        directory (str): The root directory on disc for the folder - by default this is the current working directory.

        multifile (boo): Whether to select individual files manually that are not (necessarily) in  a common directory structure.

        readlist (bool): Whether to read the directory immediately on creation. Default is True
    """

    _defaults = {"type": ImageArray, "pattern": ["*.png", "*.tiff", "*.jpeg", "*.jpg", "*.tif"]}
    _no_defaults = ["flat"]

    def __init__(self, *args, **kargs):
        """nitialise the ImageFolder.

        Mostly a pass through to the :py:class:`Stoner.Folders.baseFolder` class.
        """
        super(ImageFolderMixin, self).__init__(*args, **kargs)

    @property
    def size(self):
        """Return the size of an individual image or False if not all images are the same size."""
        shape = self.images[0].shape
        for i in self.images:
            if i.shape != shape:
                return False
        return shape

    @property
    def images(self):
        """A generator that iterates over just the images in the Folder."""
        return _generator(self)

    def _getattr_proxy(self, item):
        """Override baseFolder proxy call to access a method of the ImageFile

        Args:
            item (string): Name of method of metadataObject class to be called

        Returns:
            Either a modifed copy of this objectFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        meth = getattr(self.instance, item, None)

        def _wrapper_(*args, **kargs):
            """Wraps a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None): specify to store the return value in the individual object's metadata

            Note:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item
            """
            retvals = []
            _return = kargs.pop("_return", None)
            for ix, f in enumerate(self):
                meth = getattr(f, item, None)
                ret = meth(*args, **kargs)  # overwriting array is handled by ImageFile proxy function
                retvals.append(ret)
                if item == "crop":
                    self[ix] = ret
                if _return is not None:
                    if isinstance(_return, bool) and _return:
                        _return = meth.__name__
                    self[ix][_return] = ret
            return retvals

        # Ok that's the wrapper function, now return  it for the user to mess around with.
        _wrapper_.__doc__ = meth.__doc__
        _wrapper_.__name__ = meth.__name__
        return _wrapper_

    def apply_all(self, *args, **kargs):
        """apply function to all images in the stack

        Args:
            func(string or callable):
                if string it must be a function reachable by ImageArray
            quiet(bool):
                if False print '.' for every iteration

        Note:
            Further args, kargs are passed through to the function
        """
        args = list(args)
        func = args.pop(0)
        quiet = kargs.pop("quiet", True)
        if isinstance(func, string_types):
            for i, im in enumerate(self):
                f = getattr(im, func)
                self[i] = f(*args, **kargs)
                if not quiet:
                    print(".")
        elif hasattr(func, "__call__"):
            for i, im in enumerate(self):
                self[i] = func(im, *args, **kargs)
            if not quiet:
                print(".")

    def average(self, weights=None, _box=None):
        """Get an array of average pixel values for the stack.

        Pass through to numpy average
        Returns:
            average(ImageArray):
                average values
        """
        if not self.size:
            raise RuntimeError("Cannot average Imagefolder if images have different sizes")
        stack = np.stack(self.images, axis=0)
        average = np.average(stack, axis=0, weights=weights)
        ret = average.view(ImageArray)
        ret.metadata = self.metadata.common_metadata
        return ImageFile(ret[ret._box(_box)])

    def loadgroup(self):
        """Load all files from this group into memory"""
        for _ in self:
            pass

    def as_stack(self):
        """Return a ImageStack of the images in the current group."""
        from Stoner.Image import ImageStack

        k = ImageStack(self)
        return k

    def mean(self, _box=None):
        """Calculate the mean value of all the images in the stack.

        Actually a synonym for self.average with not weights
        """
        return self.average(_box=_box)

    def stddev(self, weights=None):
        """Calculate weighted standard deviation for stack

        This is a biased standard deviation, may not be appropriate for small sample sizes
        """
        weights = np.ones(len(self)) if weights is None else weights
        avs = self.average(weights=weights)
        sumsqdev = np.zeros_like(avs)
        for ix, img in enumerate(self.images):
            sumsqdev += weights[ix] * (img - avs) ** 2
        sumsqdev = np.sqrt(sumsqdev) / np.sum(weights, axis=0)
        return sumsqdev.view(ImageArray)

    def stderr(self, weights=None):
        """Standard error in the stack average"""
        serr = self.stddev(weights=weights) / np.sqrt(len(self))
        return serr

    def view(self):
        """Create a matplotlib animated view of the contents.

        """
        cv = CollectionViewer(self.images)
        cv.show()
        return cv


class ImageFolder(ImageFolderMixin, DiskBasedFolder, baseFolder):

    """Folder object for images.

    ImageFolder is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

        Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`) the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)

        extra_args (dict): Extra arguments to use when instantiatoing the contents of the folder from a file on disk.

        pattern (str or regexp): A filename globbing pattern that matches the contents of the folder. If a regular expression is provided then
            any named groups are used to construct additional metadata entryies from the filename. Default is *.* to match all files with an extension.

        read_means (bool): IF true, additional metatdata keys are added that return the mean value of each column of the data. This can hep in
            grouping files where one column of data contains a constant value for the experimental state. Default is False

        recursive (bool): Specifies whether to search recurisvely in a whole directory tree. Default is True.

        flatten (bool): Specify where to present subdirectories as spearate groups in the folder (False) or as a single group (True). Default is False.
            The :py:meth:`DiskBasedFolder.flatten` method has the equivalent effect and :py:meth:`DiskBasedFolder.unflatten` reverses it.

        directory (str): The root directory on disc for the folder - by default this is the current working directory.

        multifile (boo): Whether to select individual files manually that are not (necessarily) in  a common directory structure.

        readlist (bool): Whether to read the directory immediately on creation. Default is True
    """

    pass
