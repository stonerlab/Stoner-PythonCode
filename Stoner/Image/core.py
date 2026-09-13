# -*- coding: utf-8 -*-
"""Implements core image handling classes for the :mod:`Stoner.Image` package."""

__all__ = ["ImageFile"]
import urllib
from copy import copy, deepcopy
from io import BytesIO as StreamIO
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import (
    color,
    exposure,
    feature,
    filters,
    graph,
    io,
    measure,
    morphology,
    restoration,
    segmentation,
    transform,
    util,
)

from ..compat import (  # Some things to help with Python2 and Python3 compatibility
    int_types,
    np_version,
    path_types,
    string_types,
)
from ..core.base import metadataObject
from ..core.exceptions import StonerLoadError, StonerUnrecognisedFormat
from ..tools import istuple, make_Data
from ..tools.classes import Options
from ..tools.decorators import (
    class_modifier,
    class_wrapper,
    image_file_adaptor,
    make_Image,
)
from ..tools.file import (
    URL_SCHEMES,
    auto_load_classes,
    file_dialog,
    get_file_name_type,
    get_filename,
    get_loader,
)
from . import imagefuncs
from .numerical import numerical_image, box
from .attrs import DrawProxy, MaskProxy

IMAGE_FILES = [("Tiff File", "*.tif;*.tiff"), ("PNG files", "*.png", "Numpy Files", "*.npy")]

if np_version.major == 1 and np_version.minor < 24:
    dtype_range = {
        np.bool_: (False, True),
        np.bool8: (False, True),
        np.uint8: (0, 255),
        np.uint16: (0, 65535),
        np.int8: (-128, 127),
        np.int16: (-32768, 32767),
        np.int64: (-(2**63), 2**63 - 1),
        np.uint64: (0, 2**64 - 1),
        np.int32: (-(2**31), 2**31 - 1),
        np.uint32: (0, 2**32 - 1),
        np.float16: (-1, 1),
        np.float32: (-1, 1),
        np.float64: (-1, 1),
    }
else:
    dtype_range = {
        np.bool_: (False, True),
        np.uint8: (0, 255),
        np.uint16: (0, 65535),
        np.int8: (-128, 127),
        np.int16: (-32768, 32767),
        np.int64: (-(2**63), 2**63 - 1),
        np.uint64: (0, 2**64 - 1),
        np.int32: (-(2**31), 2**31 - 1),
        np.uint32: (0, 2**32 - 1),
        np.float16: (-1, 1),
        np.float32: (-1, 1),
        np.float64: (-1, 1),
    }


def _add_core_(result, other):
    """Actually do result=result-other."""
    result_type = type(result)
    match other:
        case result_type() if result.shape == other.shape:
            result.image = result.image + other.image
        case np.ndarray() if result.shape == other.shape:
            result.image = result.image + other
        case int() | float():
            result.image = result.image + other
        case _:
            return NotImplemented
    return result


def _floor_div_core_(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageFile

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image // other
    return result


def _div_core_(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageFile

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image / other
    return result


def _sub_core_(result, other):
    """Actually do result=result-other."""
    result_type = type(result)
    match other:
        case result_type() if result.shape == other.shape:
            result.image = result.image - other.image
        case np.ndarray() if result.shape == other.shape:
            result.image = result.image - other
        case int() | float():
            result.image = result.image - other
        case _:
            return NotImplemented
    return result


def copy_into(source: "ImageFile", dest: "ImageFile") -> "ImageFile":
    """Copy the data associated with source to dest.

    Args:
        source(ImageFile): The ImageFile object to be copied from
        dest (ImageFile): The ImageFile objrct to be changed by receiving the copiued data.

    Returns:
        The modified *dest* ImageFile.

    Unlike copying or deepcopying a ImageFile, this function preserves the class of the destination and just
    overwrites the attributes that represent the data in the ImageFile.
    """
    dest.image = source.to_numpy()
    dest.metadata = source.metadata.copy()
    for k in source._public_attrs:
        if k not in {"_image", "_metadata"} and hasattr(source, k):
            setattr(dest, k, deepcopy(getattr(source, k)))
    return dest




@class_modifier(
    [
        color,
        exposure,
        feature,
        io,
        measure,
        filters,
        filters.rank,
        graph,
        util,
        restoration,
        morphology,
        segmentation,
        transform,
    ],
    adaptor=image_file_adaptor,
    alias=r"^skimage\.",
)
@class_modifier(
    [ndi],
    transpose=True,
    adaptor=image_file_adaptor,
)
@class_modifier(imagefuncs, overload=True, adaptor=image_file_adaptor)
@class_wrapper(target=np.ma.MaskedArray, exclude_below=metadataObject)
class ImageFile(metadataObject):
    """An Image file type that is analogous to :py:class:`~Stoner.core.data.Data`.

    Populated standalone images own an xarray Dataset of intensities and exclusions,
    with typed metadata kept separately. The image attribute supplies a detached
    :py:class:`numpy.ma.MaskedArray` compatibility snapshot for numerical reads.

    Attributes:
        image (:py:class:`numpy.ma.MaskedArray`):
            A detached read-only image snapshot. Assign the whole property to replace
            pixels, or use direct image indexing and editing contexts for updates.
        metadata (:py:class:`Stoner.core.RegexpDict`):
            A dictionary of metadata items associated with this image.
        filename (str):
            The name of the file from which this image was loaded.
        title (str):
            The title of the image (defaults to the filename).
        mask (:py:class:`Stoner.Image.attrs.MaskProxy`):
            A special object that allows manipulation of the image's mask - thius allows the
            user to selectively disable regions of the image from rpocessing functions.
        draw (:py:class:`Stoner.Image.attrs.DrawProxy`):
            A special object that allows the user to manipulate the image data by making use of
            :py:mod:`skimage.draw` functions as well as some additional drawing functions.
        clone (:py:class:`Stoner.ImageFile`):
            Return a duplicate copy of the current image - this allows subsequent methods to
            modify the cloned version rather than the original version.
        centre (tuple of (int,int)):
            The coordinates of the centre of the image.
        aspect (float):
            The aspect ratio (width/height) of the image.
        max_box (tuple (0,x-size,0-y-size)):
            The extent of the image size in a form suitable for use in defining a box.
        flip_h (ImageFile):
            Clone the current image and then flip it horizontally (left-right).
        flip_v (ImageFile):
            Clone the current image and then flip it vertically (top-bottom).
        CW (ImageFile):
            Clone the current image and then rotate it 90 degrees clockwise.
        CCW (ImageFile):
            Clone the current image and then rotate it 90 degrees counter-clockwise.
        T (ImageFile):
            Transpose the current image
        shape (tuple (int,int)):
            Return the current shape of the image (rows, columns)
        dtype (:py:class:`numpy.dtype`):
            The current dtype of the elements of the image data.

    Numerical methods use detached NumPy working arrays and explicitly commit
    results to the owner. Pixel assignment, masks and drawing update the owner.
    NumPy operations return ordinary arrays and do not carry Stoner metadata.

    Examples::

        imfile.asfloat()
        imfile.image = np.abs(imfile.image)
    """

    # pylint: disable=no-member
    filename = ""
    _protected_attrs = []
    _patterns = ["*.png", "*.tif", "*.jpeg", "*.jpg"]
    mime_type = ["image/png", "image/jpeg", "image/tiff", "application/octet-stream"]
    priority = 32

    def __init__(self, *args, **kwargs):
        """Construct image pixels and metadata from an array, file or gridded data.

        Populated standalone images transfer the prepared pixels into their xarray
        owner. A no-argument instance remains an empty loader placeholder until
        pixels are assigned; explicit zero-sized image inputs are rejected.

        There is one special case of creating an ImageFile from a :py:class:`Stoner.core.data.Data`. In this case the
        the DataFile is assumed to contain (x,y,z) data that should be converted to a map of
        z on a regular grid of x,y. The columns for the x,y,z data can be taken from the DataFile's
        :py:attr:`Stoner.core.data.Data.setas` attribute or overridden by providing xcol, ycol and zcol keyword
        arguments. A further *shape* keyword can spewcify the shape as a tuple or "unique" to use the unique values of
        x and y or if omitted asquare grid will be interpolated.

        """
        self._image = numerical_image()  # Ensire we have the image data in place
        super().__init__(*args, **kwargs)
        args = list(args)
        if args and isinstance(args[0], (str, Path)) and not Path(args[0]).is_file():
            raise StonerLoadError(f"Image file does not exist: {args[0]}")
        if args and isinstance(args[0], (list, tuple)):
            args[0] = np.asarray(args[0])
        if args and isinstance(args[0], np.ndarray) and args[0].ndim == 1:
            args[0] = np.atleast_2d(args[0])
        self.metadata.setdefault("Loaded from", "")
        if len(args) == 0:
            pass
        elif len(args) > 0 and isinstance(args[0], path_types):
            try:
                copy_into(self.__class__.load(args[0], **kwargs), self)
                self._public_attrs = {"title": str, "filename": str}
                return
            except StonerLoadError:
                args[0] = numerical_image(*args, **kwargs)
        if len(args) > 0 and isinstance(args[0], ImageFile):  # Fixing type
            self._image = args[0].image
            for k in args[0]._public_attrs:
                setattr(self, k, getattr(args[0], k, None))
        elif len(args) > 0 and isinstance(args[0], np.ndarray):  # Fixing type
            self._image = numerical_image(*args, **kwargs)
            if hasattr(args[0], "_public_attrs"):
                for k in args[0]._public_attrs:
                    setattr(self, k, getattr(args[0], k, None))
        elif len(args) > 0 and isinstance(
            args[0], make_Data(None)
        ):  # Support initing from a DataFile that defines x,y,z coordinates
            self._init_from_datafile(*args, **kwargs)
        self._public_attrs = {"title": str, "filename": str}

    ###################################################################################################################
    ############################# Properties #### #####################################################################

    @property
    def _repr_png_(self):
        return self._repr_png_private_


    @property
    def data(self):
        """Alias for image[:]. Equivalence to Stoner.data behaviour."""
        return self.image

    @data.setter
    def data(self, value):
        """Access the image data by data attribute."""
        self.image = value

    @property
    def draw(self):
        """Access the DrawProxy object for accessing the skimage draw sub module."""
        from .storage_bridge import owner_of, OwnerDraw
        return OwnerDraw(self) if owner_of(self) is not None else DrawProxy(self.image, self)

    @property
    def image(self):
        from .storage_bridge import owner_of, snapshot
        return snapshot(self) if owner_of(self) is not None else self.__dict__["_image"]

    @image.setter
    def image(self, value):
        from .storage_bridge import replace_image
        if self.__dict__.get("_image_initialising", False):
            self.__dict__["_image"] = numerical_image(value)
        else:
            replace_image(self, value)

    _box = box
    max_box = property(lambda self: (0, self.shape[1], 0, self.shape[0]))
    aspect = property(lambda self: float(self.shape[1]) / self.shape[0])
    centre = property(lambda self: (self.shape[0] / 2, self.shape[1] / 2))
    title = property(lambda self: self.__dict__.get("_image_attrs", {}).get("_title") or self.filename,
                     lambda self, value: self.__dict__.setdefault("_image_attrs", {}).update(_title=value))
    flip_h = property(lambda self: self._permutation("flip_h"))
    flip_v = property(lambda self: self._permutation("flip_v"))
    CW = property(lambda self: self._permutation("CW"))
    CCW = property(lambda self: self._permutation("CCW"))

    def _permutation(self, name):
        from .storage_transforms import permute
        return type(self).from_storage(permute(self.export_storage(), name))

    @property
    def mask(self):
        """Get the mask of the underlying IamgeArray."""
        return MaskProxy(self)

    @mask.setter
    def mask(self, value):
        """Set the underlying ImageFile's mask."""
        if isinstance(value, ImageFile):
            value = value.image
        if isinstance(value, MaskProxy):
            value = value._mask
        from .storage_bridge import owner_of
        owner = owner_of(self)
        if owner is not None:
            owner.mask[:] = value
        else:
            self.image.mask = value
    ###################################################################################################################
    ############################# Special methods #####################################################################



    def __getstate__(self):
        """Record state for pickling ImageFiles."""
        ret = copy(self.__dict__)
        ret.update({"metadata": self.metadata})
        return ret

    def __setstate__(self, state):
        """Write state for unpickling ImageFiles."""
        metadata = state.pop("metadata", {})
        self.__dict__.update(state)
        self.metadata.update(metadata)


    def __delattr__(self, name):
        """Handle the delete attribute code."""
        super().__delattr__(name)
        if name in self._public_attrs_real:
            del self._public_attrs_real[name]

    def __setattr__(self, n, v):
        """Handle setting attributes."""
        obj, _ = self._where_attr(n)
        if obj is None:  # This is a new attribute so note it for preserving
            obj = self
            if self._where_attr("_public_attrs_real")[0] is self:
                self._public_attrs = {n: type(v)}
        if obj is self:
            super().__setattr__(n, v)
        else:
            setattr(obj, n, v)

    def __add__(self, other):
        """Implement the subtract operator."""
        result = self.clone
        result = _add_core_(result, other)
        return result

    def __iadd__(self, other):
        """Implement the inplace subtract operator."""
        result = self
        result = _add_core_(result, other)
        return result

    def __floordiv__(self, other):
        """Implement a // operator to do XMCD calculations on a whole image."""
        if isinstance(other, ImageFile):
            if (
                hasattr(other, "polarization")
                and hasattr(self, "polarization")
                and getattr(self, "polarization") == getattr(other, "polarization")
            ):
                raise ValueError("Can only calculate and XMCD ratio from images of opposite polarization")
            if not (hasattr(other, "polarization") and hasattr(self, "polarization")) and Options().warnings:
                warn("Calculating XMCD ratio even though one or both image polarizations cannoty be determined.")
            if self.image.dtype != other.image.dtype:
                raise ValueError(
                    "Only ImageFiles with the same type of underlying image data can be used to calculate an"
                    + "XMCD ratio.Mismatch is {self.image.dtype} vs {other.image.dtype}"
                )
            if self.image.dtype.kind != "f":
                ret = self.clone.convert(float)
                other = other.clone.convert(float)
            else:
                ret = self.clone
            plus, minus = self, other
            polarization = getattr(self, "polarization", 1)
            ret.image = polarization * (plus.image - minus.image) / (plus.image + minus.image)
            return ret
        result = self
        return _floor_div_core_(result, other)

    def __truediv__(self, other):
        """Implement the divide operator."""
        result = self.clone
        result = _div_core_(result, other)
        return result

    def __itruediv__(self, other):
        """Implement the inplace divide operator."""
        result = self
        result = _div_core_(result, other)
        return result

    def __sub__(self, other):
        """Implement the subtract operator."""
        result = self.clone
        result = _sub_core_(result, other)
        return result

    def __isub__(self, other):
        """Implement the inplace subtract operator."""
        result = self
        result = _sub_core_(result, other)
        return result

    def __neg__(self):
        """Intelliegent negate function that handles unsigned integers."""
        ret = self.clone
        if self._image.dtype.kind == "u":
            for k in dtype_range:  # Have to manually look for dtype :-()
                if k == self._image.dtype:
                    break
            else:
                raise TypeError(f"Unrecognised unsigned type {self._image.dtype}, cannot negate sensibly !")
            high_val = dtype_range[k][1]
            ret.image = high_val - self.image
        else:
            ret.image = -self.image
        return ret

    def __invert__(self):
        """Equivalent to clockwise rotation."""
        return self.CW

    def __eq__(self, other):
        """Impleent and equality test."""
        if id(self) == id(other):
            ret = True  # short circuit for identity
        elif not isinstance(other, ImageFile):
            ret = False  # Shortcircuit for non equivalent types
        else:
            ret = self.metadata == other.metadata and np.all(self.image == other.image)
        return ret

    def __repr__(self):
        """Implement standard representation for text based consoles."""
        return (
            f"{self.filename}({type(self)}) of shape {self.shape} ({self.image.dtype}) and"
            + f" {len(self.metadata)} items of metadata"
        )

    ###################################################################################################################
    ############################# Private methods #####################################################################

    def _init_from_datafile(self, *args, **kwargs):
        """Initialise ImageFile from DataFile defining x,y,z coordinates.

        Args:
            args[0] (DataFile):
                A :py:class:`Stoner.core.data.Data` instance that defines x, y, z coordinates or has columns specified
                in keywords.

        Keyword Arguments:
            xcol (column index):
                Column in the DataFile that has the x-co-ordinate
            ycol (column index):
                Column in the data file that defines the y-cordinate
            zcol (column index):
                Column in the datafile that defines the intensity
        """
        data = make_Data(args[0])
        shape = kwargs.pop("shape", "unique")

        _ = data._col_args(**kwargs)
        data.setas(x=_.xcol, y=_.ycol, z=_.zcol)  # pylint: disable=not-callable
        if isinstance(shape, string_types) and shape == "unique":
            shape = (len(np.unique(data.x)), len(np.unique(data.y)))
        elif istuple(shape, int_types, int_types):
            pass
        else:
            shape = None
        X, Y, Z = data.griddata(_.xcol, _.ycol, _.zcol, shape=shape)
        self.image = Z.view(np.ma.MaskedArray)
        self.metadata = data.metadata.copy()
        self["x_vector"] = np.unique(X)
        self["y_vector"] = np.unique(Y)

    def _load(self, filename, *args, **kwargs):
        """Load an ImageFile by calling the ImageFile method instead."""
        self._image = numerical_image(filename, *args, **kwargs)
        for k in self._image._public_attrs:
            setattr(self, k, getattr(self._image, k, None))
        return self

    def _repr_png_private_(self):
        """Provide a display function for iPython/Jupyter."""
        fig = self.image.imshow(mask_color=self.mask.colour)
        plt.title(self.filename)
        data = StreamIO()
        fig.savefig(data, format="png")
        plt.close(fig)
        data.seek(0)
        ret = data.read()
        data.close()
        return ret

    def _where_attr(self, n):
        """Get the object that has the named attribute."""
        try:
            _ = super().__getattribute__(n)
            return self, _
        except AttributeError:
            try:
                _ = getattr(self._image, n)
                return self._image, _
            except AttributeError:
                return None, None

    ###################################################################################################################
    #############################  Public methods #####################################################################

    def get_filename(self, mode):
        """Force the user to choose a new filename using a system dialog box.

        Args:
            mode (string):
                The mode of file operation to be used when calling the dialog box

        Returns:
            str:
                The new filename

        Notes:
            The filename attribute of the current instance is updated by this method as well.
        """
        self.filename = file_dialog(mode, self.filename, type(self))
        return self.filename

    @classmethod
    def load(cls, *args, **kwargs):
        """Create a :py:class:`ImageFile` from file and guessing a better subclass if necessary.

        Args:
            filename (string or None):
                path to file to load

        Keyword Arguments:
            auto_load (bool):
                If True (default) then the load routine tries all the subclasses of :py:class:`ImageFile` in turn to
                load the file
            filetype (:py:class:`ImageFile`, str):
                If not none then tries using filetype as the loader.
            debug (bool):
                Turn on debugging when running autoload. Default *False*

        Returns:
            (ImageFile):
                A a new :py:data:`ImageFile` (or subclass thereof) instance

        Notes:
            If *filetupe* is a string, then it is first tried as an exact match to a subclass name, otherwise it
            is used as a partial match and the first class in priority order is that matches is used.

            Some subclasses can be found in the :py:mod:`Stoner.formats` package.

            Each subclass is scanned in turn for a priority that governs
            the order in which they are tried. Subclasses which can make an early positive determination that a
            file has the correct format can have higher priority levels. Classes should return a suitable exception
            if they fail to load the file.

            If no class can load a file successfully then a RunttimeError exception is raised.
        """
        filename, args, kwargs = get_filename(args, kwargs)
        filetype = kwargs.pop("filetype", None)
        debug = kwargs.pop("debug", False)
        auto_load = kwargs.pop("auto_load", filetype is None)
        if isinstance(filename, path_types) and urllib.parse.urlparse(str(filename)).scheme not in URL_SCHEMES:
            filename, filetype = get_file_name_type(filename, filetype, ImageFile)
        if filename is None or not filename:
            filename = file_dialog("r", filename, "ImageFile")
        elif not auto_load and not filetype:
            raise StonerLoadError("Cannot read data from non-path like filenames !")
        if auto_load:  # We're going to try every subclass we canA
            try:
                ret = auto_load_classes(filename, "Image", debug=debug, args=args, kwargs=kwargs)
            except StonerUnrecognisedFormat:
                ret = ImageFile()
                ret = ret._load(filename, *args, **kwargs)
                ret["Loaded as"] = filetype.__name__
        else:
            if isinstance(filetype, type) and issubclass(filetype, ImageFile):
                filetype = filetype.__name__
            elif isinstance(filetype, ImageFile):
                filetype = filetype.__class__.__name__
            if not isinstance(filetype, str):
                raise TypeError(f"Unable to work out how to load {filetype}")
            loader = get_loader(filetype)
            try:
                ret = loader(make_Image(), filename, *args, **kwargs)
                ret["Loaded as"] = filetype
            except StonerLoadError as err:
                raise ValueError(f"Unable to load {filename}") from err

        for k, i in kwargs.items():
            if not callable(getattr(ret, k, lambda x: False)):
                setattr(ret, k, i)
        ret._kwargs = kwargs
        return ret

    def save(self, filename=None, **kwargs):
        """Save the image into the file 'filename'.

        Args:
            filename (string, bool or None):
                Filename to save data as, if this is None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If filename is False then a file dialog is forced.

        Keyword Arguments:
            fmt (string or list):
                format to save data as. 'tif', 'png' or 'npy' or a list of them. If not included will guess from
                filename.

        Notes:
            Metadata will be preserved in .png and .tif format.

            fmt can be 'png', 'npy', 'tif', 'tiff'  or a list of more than one of those.
            tif is recommended since metadata is lost in .npy format but data is
            converted to integer format for png so that definition cannot be
            saved.
        """
        # catch before metadataObject tries to take over.
        self.image.save(filename, **kwargs)


from .storage_bridge import install as _install_image_storage

_install_image_storage(ImageFile)
