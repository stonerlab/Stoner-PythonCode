# -*- coding: utf-8 -*-
"""Implements a baseFolder type structure for working with collections of images."""
__all__ = ["_generator", "ImageFolderMixin", "ImageFolder"]
from warnings import warn
from importlib import import_module
from os import path
from json import loads, dumps

from skimage.viewer import CollectionViewer
import numpy as np
from PIL.TiffImagePlugin import ImageFileDirectory_v2, Image

from .core import ImageArray
from ..Folders import DiskBasedFolderMixin, baseFolder
from ..compat import string_types, int_types
from . import ImageFile


def _load_ImageArray(f, **kargs):
    """Simple meothd to load an image array."""
    kargs.pop("img_num", None)
    return ImageArray(f, **kargs)


class _generator:

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
        """Get the length of the iterator.

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
        raise StopIteration("Finished iterating Folder.")

    def __getitem__(self, index):
        """Item accessor method.

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


class ImageFolderMixin:

    """Mixin to provide a folder object for images.

    ImageFolderMixin is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

    Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`):
            the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)
        extra_args (dict):
            Extra arguments to use when instantiatoing the contents of the folder from a file on disk.
        pattern (str or regexp):
            A filename globbing pattern that matches the contents of the folder. If a regular expression is
            provided then any named groups are used to construct additional metadata entryies from the filename.
            Default is *.* to match all files with an extension.
        read_means (bool):
            If true, additional metatdata keys are added that return the mean value of each column of the data.
            This can hep in grouping files where one column of data contains a constant value for the
            experimental state. Default is False
        recursive (bool):
            Specifies whether to search recurisvely in a whole directory tree. Default is True.
        flatten (bool):
            Specify where to present subdirectories as spearate groups in the folder (False) or as a single group
            (True). Default is False. The :py:meth:`DiskBasedFolderMixin.flatten` method has the equivalent effect and
            :py:meth:`DiskBasedFolderMixin.unflatten` reverses it.
        directory (str):
            The root directory on disc for the folder - by default this is the current working directory.
        multifile (boo):
            Whether to select individual files manually that are not (necessarily) in  a common directory structure.
        readlist (bool):
            Whether to read the directory immediately on creation. Default is True
    """

    _defaults = {"type": ImageArray, "pattern": ["*.png", "*.tiff", "*.jpeg", "*.jpg", "*.tif"]}
    _no_defaults = ["flat"]

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
        """Override baseFolder proxy call to access a method of the ImageFile.

        Args:
            item (string):
                Name of method of metadataObject class to be called

        Returns:
            Either a modifed copy of this objectFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        meth = getattr(self.instance, item, None)

        def _wrapper_(*args, **kargs):
            """Wraps a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None):
                    specify to store the return value in the individual object's metadata

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
                if getattr(meth, "changes_size", False):
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

    def align(self, *args, **kargs):
        """Align each image in the folder to the reference image.

        Args:
            ref (str, int, ImageFile, ImageArray or 2D array):
                The reference image to align to. If a string or an int, then this is used to lookup the corresponding
                member of the ImageFolder which is then used. ImageFiles, ImageArrays and 2D arrays are used directly
                as reference images.

        Keyword Arguments:
            method (str):
                The mthod is passed to the :py:class:`Stone.Image.ImageArray.align` method to control how the image
                alignment is done. By default the 'Scharr' method is used.
            box (int, float, tuple of ints or floats):
                Specifies a subset of the images to be used to calculate the alignment with.
            scale (int):
                Magnification factor to scale the image by before doing the alignment for better sub=pixel alignments.

        Returns:
            The aligned ImageFolder.
        """
        if len(args) == 1:
            ref = args[0]
        elif len(args) == 0:
            ref = 0
        else:
            raise ValueError(
                f"{self.__class__.__name__}.align only takes zero or one positional arguments not {len(args)}!"
            )
        # Get me reference data
        if isinstance(ref, (string_types, int_types)):
            ref_data = self.__getter__(ref, instantiate=True)
            if isinstance(ref_data, ImageFile):
                ref_data = ref_data.image
        elif isinstance(ref, ImageFile):
            ref_data = ref.image.view(ImageArray)
        elif isinstance(ref, np.ndarray) and ref.ndim == 2:
            ref_data = ref.view(ImageArray)
        else:
            try:
                ref_data = np.array(ref).view(ImageArray)
                if ref_data.ndim != 2:
                    raise TypeError()
            except Exception:
                raise TypeError(f"Cannot interpret {type(ref)} as reference image data.")
        # Call align on each object
        self.each.align(ref_data, **kargs)
        limits = self.metadata.slice("translation_limits", output="array")
        stack_limits = np.zeros(4)
        stack_limits[::2] = limits.max(axis=0)[::2]
        stack_limits[1::2] = limits.min(axis=0)[1::2]
        self.metadata["translation_limits"] = tuple(stack_limits)
        return self

    def apply_all(self, func, *args, **kargs):
        """Apply function to all images in the stack

        Args:
            func(string or callable):
                if string it must be a function reachable by ImageArray
            quiet(bool):
                if False print '.' for every iteration

        Note:
            Further args, kargs are passed through to the function
        """
        warn("apply_all is depricated and will be removed in a future version. Use ImageFolder.each() instead")
        return self.each(func, *args, **kargs)

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
        """Load all files from this group into memory."""
        for _ in self:
            pass

    def as_stack(self):
        """Return a ImageStack of the images in the current group."""
        stack = import_module("stack", ".")
        k = stack.ImageStack(self)
        return k

    @classmethod
    def from_tiff(cls, filename, **kargs):
        """Create a new ImageArray from a tiff file."""
        from PIL import Image

        self = cls(**kargs)
        with Image.open(filename, "r") as img:
            tags = img.tag_v2
            if 270 in tags:

                try:
                    userdata = loads(tags[270])
                    typ = userdata.get("type", cls.__name__)
                    mod = userdata.get("module", cls.__module__)
                    layout = userdata.get("layout", (0, {}))

                    mod = import_module(mod)
                    typ = getattr(mod, typ)
                    if not issubclass(typ, ImageFolderMixin):
                        raise TypeError(
                            f"Bad type in Tiff file {typ.__name__} is not a subclass of Stoner.ImageFolder"
                        )
                    metadata = userdata.get("metadata", [])
                except Exception:
                    metadata = []
            else:
                raise TypeError(f"Cannot load as an ImageFolder due to lack of description tag")
            imglist = []
            for ix, md in enumerate(metadata):
                img.seek(ix)
                image = np.asarray(img)
                if image.ndim == 3:
                    if image.shape[2] < 4:  # Need to add a dummy alpha channel
                        image = np.append(np.zeros_like(image[:, :, 0]), axis=2)
                    image = image.view(dtype=np.uint32).reshape(image.shape[:-1])

                if isinstance(self.type, np.ndarray):
                    image = image.view(self.type)
                else:
                    image = self.type(image)
                image.metadata.import_all(md)
                imglist.append(image)

            self._marshall(layout=layout, data=imglist)

        return self

    def mean(self, _box=None):
        """Calculate the mean value of all the images in the stack.

        Actually a synonym for self.average with not weights
        """
        return self.average(_box=_box)

    def stddev(self, weights=None):
        """Calculate weighted standard deviation for stack.

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
        """Standard error in the stack average."""
        serr = self.stddev(weights=weights) / np.sqrt(len(self))
        return serr

    def to_tiff(self, filename):
        """Save the ImageArray as a tiff image with metadata.

        Args:
            filename (str):
                Filename to save file as.

        Note:
            PIL can save in modes "L" (8bit unsigned int), "I" (32bit signed int),
            or "F" (32bit signed float). In general max info is preserved for "F"
            type so if forcetype is not specified then this is the default. For
            boolean type data mode "L" will suffice and this is chosen in all cases.
            The type name is added as a string to the metadata before saving.

        """
        metadata_export = []
        imlist = []
        for d in self._marshall():
            dtype = np.dtype(d.dtype).name  # string representation of dtype we can save
            d["ImageArray.dtype"] = dtype  # add the dtype to the metadata for saving.
            metadata_export.append(d.metadata.export_all())
            if d.dtype.kind == "b":  # boolean we're not going to lose data by saving as unsigned int
                imlist.append(Image.fromarray(d.image, mode="L"))
            else:
                try:
                    imlist.append(Image.fromarray(d.image))
                except TypeError:
                    imlist.append(Image.fromarray(d.image.astype("float32")))

        ifd = ImageFileDirectory_v2()
        ifd[270] = dumps(
            {
                "type": self.__class__.__name__,
                "module": self.__class__.__module__,
                "layout": self.layout,
                "metadata": metadata_export,
            }
        )
        ext = path.splitext(filename)[1]
        if ext in [".tif", ".tiff"]:  # ensure extension is preserved in save
            pass
        else:  # default to tiff
            ext = ".tiff"

        tiffname = path.splitext(filename)[0] + ext
        imlist[0].save(tiffname, save_all=True, append_images=imlist[1:], tiffinfo=ifd)
        return self

    def view(self):
        """Create a matplotlib animated view of the contents.

        """
        cv = CollectionViewer(self.images)
        cv.show()
        return cv


class ImageFolder(ImageFolderMixin, DiskBasedFolderMixin, baseFolder):

    """Folder object for images.

    ImageFolder is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

    Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`):
            the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)
        extra_args (dict):
            Extra arguments to use when instantiatoing the contents of the folder from a file on disk.
        pattern (str or regexp):
            A filename globbing pattern that matches the contents of the folder. If a regular expression is provided
            then any named groups are used to construct additional metadata entryies from the filename. Default is *.*
            to match all files with an extension.
        read_means (bool):
            If true, additional metatdata keys are added that return the mean value of each column of the data.
            This can hep in grouping files where one column of data contains a constant value for the experimental
            state. Default is False
        recursive (bool):
            Specifies whether to search recurisvely in a whole directory tree. Default is True.
        flatten (bool):
            Specify where to present subdirectories as spearate groups in the folder (False) or as a single group
            (True). Default is False. The :py:meth:`DiskBasedFolderMixin.flatten` method has the equivalent effect
            and :py:meth:`DiskBasedFolderMixin.unflatten` reverses it.
        directory (str):
            The root directory on disc for the folder - by default this is the current working directory.
        multifile (boo):
            Whether to select individual files manually that are not (necessarily) in  a common directory structure.
        readlist (bool):
            Whether to read the directory immediately on creation. Default is True
    """

    pass
