# -*- coding: utf-8 -*-
"""Implements a baseFolder type structure for working with collections of images."""
__all__ = ["ImageFolderMixin", "ImageFolder"]
from warnings import warn
from importlib import import_module
from os import path
from json import loads, dumps
from copy import deepcopy, copy

import numpy as np
from matplotlib.pyplot import figure, Figure, subplot, get_fignums
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from PIL import Image

from .core import ImageArray
from ..Folders import DiskBasedFolderMixin, baseFolder
from ..compat import string_types, int_types
from . import ImageFile


class ImageFolderMixin:
    """Mixin to provide a folder object for images.

    ImageFolderMixin is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

    Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`):
            the type ob object to store in the folder (defaults to :py:class:`Stoner.Cire.Data`)
        extra_args (dict):
            Extra arguments to use when instantiatoing the contents of the folder from a file on disk.
        pattern (str or regexp):
            A filename globbing pattern that matches the contents of the folder. If a regular expression is
            provided then any named groups are used to construct additional metadata entryies from the filename.
            Default is *.* to match all files with an extension.
        read_means (bool):
            If true, additional metadata keys are added that return the mean value of each column of the data.
            This can hep in grouping files where one column of data contains a constant value for the
            experimental state. Default is False
        recursive (bool):
            Specifies whether to search recursively in a whole directory tree. Default is True.
        flatten (bool):
            Specify where to present subdirectories as separate groups in the folder (False) or as a single group
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
        if len(self) > 0:
            shape = self[0].shape
        else:
            shape = tuple()
        for i in self:
            if i.shape != shape:
                return False
        return shape

    @property
    def images(self):
        """Iterate over just the images in the Folder."""
        for im in self:
            if not isinstance(im, np.ndarray):
                if hasattr(im, "image"):
                    im = im.image
                else:
                    raise TypeError(f"Cannot represent {type(im)} as an ImageArray.")
            yield im

    #########################################################################################################
    ##### Folder interface methods
    #########################################################################################################

    def __getter__(self, name, instantiate=True):
        """Ensure we set the title on the image.

        Parameters:
           name (key type):
               The canonical mapping key to get the dataObject. By default
               the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instantiate (bool):
                If True (default) then always return a metadataObject. If False, the __getter__ method may return a
                key that can be used by it later to actually get the metadataObject. If None, then will return
                whatever is held in the object cache, either instance
                or name.

        Returns:
            (metadataObject):
                The metadataObject

        Note:
            Mainly we call the parent method and then set the title if it's not already set.'
        """
        ret = super().__getter__(name, instantiate)
        if hasattr(ret, "_title") and ret._title is None:
            ret._title = name
        return ret

    def align(self, *args, **kargs):
        """Align each image in the folder to the reference image.

        Args:
            ref (str, int, ImageFile, ImageArray or 2D array):
                The reference image to align to. If a string or an int, then this is used to lookup the corresponding
                member of the ImageFolder which is then used. ImageFiles, ImageArrays and 2D arrays are used directly
                as reference images.

        Keyword Arguments:
            method (str):
                The method is passed to the :py:class:`Stone.Image.ImageArray.align` method to control how the image
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
                f"{type(self).__name__}.align only takes zero or one positional arguments not {len(args)}!"
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
            except (TypeError, ValueError) as err:
                raise TypeError(f"Cannot interpret {type(ref)} as reference image data.") from err
        # Call align on each object
        self.each.align(ref_data, **kargs)
        limits = self.metadata.slice("translation_limits", output="array")
        stack_limits = np.zeros(4)
        stack_limits[::2] = limits.max(axis=0)[::2]
        stack_limits[1::2] = limits.min(axis=0)[1::2]
        self.metadata["translation_limits"] = tuple(stack_limits)
        stack_limits[::2] = np.ceil(stack_limits)[::2]
        stack_limits[1::2] = np.floor(stack_limits)[1::2]
        self.metadata["align_box"] = tuple(stack_limits.astype(int))
        return self

    def apply_all(self, func, *args, **kargs):
        """Apply function to all images in the stack.

        Args:
            func(string or callable):
                if string it must be a function reachable by ImageArray
            quiet(bool):
                if False print '.' for every iteration

        Note:
            Further args, kargs are passed through to the function
        """
        warn("apply_all is deprecated and will be removed in a future version. Use ImageFolder.each() instead")
        return self.each(func, *args, **kargs)

    def average(self, weights=None, _box=False, _metadata="first"):
        """Get an array of average pixel values for the stack.

        Pass through to numpy average

        Keyword Arguments:
            _box (crop box):
                Specifies the region of the array to be averaged. Default - entire image
            _metadata (str):
                Specifies how to generate metadata for the averaged image.
                - "first": Just ise the first image's metadata
                - "common": Find the common metadata across all images
                - "none': no metadata from images.

        Returns:
            average(ImageArray):
                average values
        """
        if not self.size:
            raise RuntimeError("Cannot average Imagefolder if images have different sizes")
        if hasattr(self, "_stack"):
            stack = self._stack.view(np.ndarray)
            axis = -1
        else:
            stack = np.stack(list(self.images), axis=0)
            axis = 0
        average = np.average(stack, axis=axis, weights=weights)
        ret = average.view(ImageArray)
        if _metadata == "common":
            ret.metadata = self.metadata.common_metadata
        elif _metadata == "first":
            ret.metadata = deepcopy(self[0].metadata)
        return self._type(ret[ret._box(_box)])

    def loadgroup(self):
        """Load all files from this group into memory."""
        for _ in self:
            pass

    def as_stack(self):
        """Return a ImageStack of the images in the current group."""
        stack = import_module(".stack", "Stoner.Image")
        k = stack.ImageStack(self)
        return k

    @classmethod
    def from_tiff(cls, filename, **kargs):
        """Create a new ImageArray from a tiff file."""
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
                except (TypeError, ValueError, IOError):
                    metadata = []
            else:
                raise TypeError("Cannot load as an ImageFolder due to lack of description tag")
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

    def mask_select(self):
        """Run the ImageFile.mask.select() on each image."""
        sel = []
        for img in self:
            img.mask.select(_selection=sel)

    def mean(self, _box=False, _metadata="first"):
        """Calculate the mean value of all the images in the stack.

        Keyword Arguments:
            _box (crop box):
                Specifies the region of the array to be averaged. Default - entire image
            _metadata (str):
                Specifies how to generate metadata for the averaged image.
                - "first": Just ise the first image's metadata
                - "common": Find the common metadata across all images
                - "none': no metadata from images.

        Actually a synonym for self.average with not weights
        """
        return self.average(_box=_box, _metadata=_metadata)

    def montage(self, *args, **kargs):
        """Call the plot method for each metadataObject, but switching to a subplot each time.

        Args:
            args: Positional arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.
            kargs: Keyword arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.

        Keyword Arguments:
            plot_extra (callable(i,j,d)):
                A callable that can carry out additional processing per plot after the plot is done
            figsize(tuple(x,y)):
                Size of the figure to create
            dpi(float):
                dots per inch on the figure
            edgecolor,facecolor(matplotlib colour):
                figure edge and frame colours.
            frameon (bool):
                Turns figure frames on or off
            FigureClass(class):
                Passed to matplotlib figure call.
            plots_per_page(int):
                maximum number of plots per figure.

        Returns:
            A list of :py:class:`matplotlib.pyplot.Axes` instances.

        Notes:
            If the underlying type of the :py:class:`Stoner.Core.metadataObject` instances in the
            :py:class:`PlotFolder` lacks a **plot** method, then the instances are converted to
            :py:class:`Stoner.Core.Data`.

            Each plot is generated as sub-plot on a page. The number of rows and columns of subplots is computed
            from the aspect ratio of the figure and the number of files in the :py:class:`PlotFolder`.
        """
        plts = kargs.pop("plots_per_page", getattr(self, "plots_per_page", len(self)))
        plts = min(plts, len(self))

        plot_extra = kargs.pop("plot_extra", lambda i, j, d: None)

        fig_num = kargs.pop("figure", getattr(self, "_figure", None))
        if isinstance(fig_num, Figure):
            kargs.setdefault("figsize", tuple(fig_num.get_size_inches()))
            kargs.setdefault("facecolor", fig_num.get_facecolor())
            kargs.setdefault("edgecolor", fig_num.get_edgecolor())
            kargs.setdefault("frameon", fig_num.get_frameon())
            kargs.setdefault("FigureClass", fig_num.__class__)
            fig_num = fig_num.number

        fig_args = getattr(self, "_fig_args", [])
        fig_kargs = getattr(self, "_fig_kargs", {"layout": "constrained"})
        for arg in ("figsize", "dpi", "facecolor", "edgecolor", "frameon", "FigureClass"):
            if arg in kargs:
                fig_kargs[arg] = kargs.pop(arg)
        if fig_num is None:
            fig = figure(*fig_args, **fig_kargs)
        elif fig_num in get_fignums():
            fig = figure(fig_num)
        else:
            fig = figure(fig_num, **fig_kargs)
        w, h = fig.get_size_inches()
        plt_x = int(np.floor(np.sqrt(plts) * w / h))
        plt_y = int(np.ceil(plts / plt_x))

        kargs["figure"] = fig
        ret = []
        j = 0
        fignum = fig.number
        for i, d in enumerate(self):
            plt_kargs = copy(kargs)
            if i % plts == 0 and i != 0:
                fig = figure(*fig_args, **fig_kargs)
                fignum = fig.number
                j = 1
            else:
                j += 1
            fig = figure(fignum)
            ax = subplot(plt_y, plt_x, j)
            plt_kargs["figure"] = fig
            plt_kargs["ax"] = ax
            if "title" in kargs:
                if isinstance(kargs["title"], str):
                    plt_kargs["title"] = kargs["title"].format(**d)
                elif callable(kargs["title"]):
                    plt_kargs["title"] = kargs["title"](d)
            ret.append(d.imshow(*args, **plt_kargs))
            plot_extra(i, j, d)
        return ret

    def stddev(self, weights=None, _box=False, _metadata="first"):
        """Calculate weighted standard deviation for stack.

        Keyword Arguments:
            _box (crop box):
                Specifies the region of the array to be averaged. Default - entire image
            _metadata (str):
                Specifies how to generate metadata for the averaged image.
                - "first": Just ise the first image's metadata
                - "common": Find the common metadata across all images
                - "none': no metadata from images.

        This is a biased standard deviation, may not be appropriate for small sample sizes
        """
        if weights is None:  # shortcircuit
            if hasattr(self, "_stack"):
                sumsqdev = np.std(self._stack.view(np.ndarray), axis=-1)
            else:
                sumsqdev = np.stack(list(self.images), axis=0).std(axis=0)
        else:
            avs = self.average(weights=weights)
            if not isinstance(avs, np.ndarray) and hasattr(avs, "image"):
                avs = avs.image
            sumsqdev = np.zeros_like(avs)
            for ix, img in enumerate(self.images):
                sumsqdev += weights[ix] * (img - avs) ** 2
            sumsqdev = np.sqrt(sumsqdev) / np.sum(weights, axis=0)
        ret = sumsqdev.view(ImageArray)
        ret.metadata = self.metadata.common_metadata
        return self._type(ret[ret._box(_box)])

    def stderr(self, weights=None, _box=False, _metadata="first"):
        """Calculate standard error in the stack average.

        Keyword Arguments:
            _box (crop box):
                Specifies the region of the array to be averaged. Default - entire image
            _metadata (str):
                Specifies how to generate metadata for the averaged image.
                - "first": Just ise the first image's metadata
                - "common": Find the common metadata across all images
                - "none': no metadata from images.
        """
        serr = self.stddev(weights=weights, _box=_box, _metadata=_metadata) / np.sqrt(len(self))
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
                "type": type(self).__name__,
                "module": type(self).__module__,
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


class ImageFolder(ImageFolderMixin, DiskBasedFolderMixin, baseFolder):
    """Folder object for images.

    ImageFolder is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

    Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`):
            the type ob object to store in the folder (defaults to :py:class:`Stoner.Cire.Data`)
        extra_args (dict):
            Extra arguments to use when instantiatoing the contents of the folder from a file on disk.
        pattern (str or regexp):
            A filename globbing pattern that matches the contents of the folder. If a regular expression is provided
            then any named groups are used to construct additional metadata entryies from the filename. Default is *.*
            to match all files with an extension.
        read_means (bool):
            If true, additional metadata keys are added that return the mean value of each column of the data.
            This can hep in grouping files where one column of data contains a constant value for the experimental
            state. Default is False
        recursive (bool):
            Specifies whether to search recursively in a whole directory tree. Default is True.
        flatten (bool):
            Specify where to present subdirectories as separate groups in the folder (False) or as a single group
            (True). Default is False. The :py:meth:`DiskBasedFolderMixin.flatten` method has the equivalent effect
            and :py:meth:`DiskBasedFolderMixin.unflatten` reverses it.
        directory (str):
            The root directory on disc for the folder - by default this is the current working directory.
        multifile (boo):
            Whether to select individual files manually that are not (necessarily) in  a common directory structure.
        readlist (bool):
            Whether to read the directory immediately on creation. Default is True
    """
