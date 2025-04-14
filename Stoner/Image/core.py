# -*- coding: utf-8 -*-
"""Implements core image handling classes for the :mod:`Stoner.Image` package."""
__all__ = ["ImageArray", "ImageFile"]
from pathlib import Path
import os
from collections.abc import Iterable
from copy import copy, deepcopy
import inspect
from importlib import import_module
from io import BytesIO as StreamIO
import urllib
from warnings import warn

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import (
    color,
    exposure,
    feature,
    io,
    measure,
    filters,
    graph,
    util,
    restoration,
    morphology,
    segmentation,
    transform,
)

from ..compat import np_version
from ..core.base import TypeHintedDict, metadataObject
from ..core.exceptions import StonerLoadError, StonerUnrecognisedFormat
from ..Core import DataFile
from ..tools import isTuple, make_Data
from ..tools.decorators import class_modifier, image_file_adaptor, class_wrapper, clones, make_Image
from ..compat import (
    string_types,
    get_filedialog,
    int_types,
    path_types,
)  # Some things to help with Python2 and Python3 compatibility
from .attrs import DrawProxy, MaskProxy
from .widgets import RegionSelect
from . import imagefuncs
from ..tools.classes import Options
from ..tools.file import (
    URL_SCHEMES,
    file_dialog,
    get_filename,
    get_loader,
    get_file_name_type,
    auto_load_classes,
)

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
            result.image += other.image
        case np.ndarray() if result.shape == other.shape:
            result.image += other
        case int() | float():
            result.image += other
        case _:
            return NotImplemented
    return result


def _floor_div_core_(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageArray

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image // other
    return result


def _div_core_(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageArray

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image / other
    return result


def _sub_core_(result, other):
    """Actually do result=result-other."""
    result_type = type(result)
    match other:
        case result_type() if result.shape == other.shape:
            result.image -= other.image
        case np.ndarray() if result.shape == other.shape:
            result.image -= other
        case int() | float():
            result.image -= other
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
    dest.image = source.image.clone
    for k in source._public_attrs:
        if hasattr(source, k):
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
    ]
)
@class_modifier([ndi], transpose=True)
@class_modifier(imagefuncs, overload=True)
class ImageArray(np.ma.MaskedArray, metadataObject):
    """A numpy array like class with a metadata parameter and pass through to skimage methods.

    ImageArray is for manipulating images stored as a 2d numpy array.
    It is built to be almost identical to a numpy array except for one extra
    parameter which is the metadata. This stores information about the image
    in a dictionary object for later retrieval.
    All standard numpy functions should work as normal and casting two types
    together should yield a ImageArray type (ie. ImageArray+np.ndarray=ImageArray)

    In addition any function from skimage should work and return a ImageArray.
    They can be called as eg. im=im.gaussian(sigma=2). Don't include the module
    name, just the function name (ie not filters.gaussian). Also omit the first
    image argument required by skimage.

    Attributes:
        metadata (:py:class:`Stoner.core.regexpDict`):
            A dictionary of metadata items associated with this image.
        filename (str):
            The name of the file from which this image was loaded.
        title (str):
            The title of the image (defaults to the filename).
        mask (:py:class:`numpy.ndarray of bool`):
            The underlying mask data of the image. Masked elements (i.e. where mask=True) are ignored for many
            image operations. Indexing them will return the mask fill value (typically NaN, ot -1 or -MAXINT)
            draw (:py:class:`Stoner.Image.attrs.DrawProxy`):
            A special object that allows the user to manipulate the image data by making use of
            :py:mod:`skimage.draw` functions as well as some additional drawing functions.
        clone (:py:class:`Stoner.ImageArry`):
            Return a duplicate copy of the current image - this allows subsequent methods to
            modify the cloned version rather than the original version.
        centre (tuple of (float,float)):
            The coordinates of the centre of the image.
        aspect (float):
            The aspect ratio (width/height) of the image.
        max_box (tuple (0,x-size,0-y-size)):
            The extent of the image size in a form suitable for use in defining a box.
        flip_h (:py:class:`ImageArray`):
            Clone the current image and then flip it horizontally (left-right).
        flip_v (:py:class:`ImageArray`):
            Clone the current image and then flip it vertically (top-bottom).
        CW (:py:class:`ImageArray`):
            Clone the current image and then rotate it 90 degrees clockwise.
        CCW (:py:class:`ImageArray`):
            Clone the current image and then rotate it 90 degrees counter-clockwise.
        T (:py:class:`ImageArray`):
            Transpose the current image
        shape (tuple (int,int)):
            Return the current shape of the image (rows, columns)
        dtype (:py:class:`numpy.dtype`):
            The current dtype of the elements of the image data.


    For clarity it should be noted that any function will not alter the current
    instance, it will clone it first then return the clone after performing the
    function on it.

    Note:
         For arrays the indexing is (row, column). However the normal way to index
         an image would be to do (horizontal, vert), which is the opposite.
         In ImageArray the coordinate system is chosen similar to skimage. y points
         down x points right and the origin is in the top left corner of the image.
         When indexing the array therefore you need to give it (y,x) coordinates
         for (row, column).::

              ----> x (column)
             |
             |
             v
             y (row)

         eg I want the 4th pixel in the horizontal direction and the 10th pixel down
         from the top I would ask for ImageArray[10,4]

         but if I want to translate the image 4 in the x direction and 10 in the y
         I would call im=im.translate((4,10))
    """

    # Proxy attributes for storing imported functions. Only do the import when needed
    _func_proxy = None

    # extra attributes for class beyond standard numpy ones

    # Default values for when we can't find the attribute already
    _defaults = {"debug": False, "_hardmask": False}

    fmts = ["png", "npy", "tiff", "tif"]

    # These will be overridden with instance attributes, but setting here allows ImageFile properties to be defined.
    debug = False
    filename = ""

    # now initialise class

    def __new__(cls, *args, **kargs):
        """Construct an ImageArray object.

        We're using __new__ rather than __init__ to imitate a numpy array as
        close as possible.
        """
        array_arg_keys = ["dtype", "copy", "order", "subok", "ndmin", "mask"]  # kwargs for array setup
        array_args = {k: kargs.pop(k) for k in array_arg_keys if k in kargs.keys()}
        user_metadata = kargs.get("metadata", {})

        # 0 args initialisation
        match args:
            case tuple() if len(args) == 0:
                ret = np.empty((0, 0), dtype=float).view(cls)
                # merge the results of __new__ from emtadataObject
            case (cls(),):
                ret = args[0]
            case (np.ndarray(),):
                # numpy array or ImageArray)
                if args[0].ndim < 2:
                    ret = np.atleast_2d(args[0]).view(ImageArray)
                else:
                    ret = args[0].view(ImageArray)
                kargs["metadata"] = getattr(args[0], "metadata", TypeHintedDict())
                kargs["metadata"].update(user_metadata)
            case (bool(),) if not args[0]:
                patterns = (("png", "*.png"), ("npy", "*.npy"))
                arg = get_filedialog(what="r", filetypes=patterns)
                if len(arg) == 0:
                    raise ValueError("No file given")
                # Filename- load datafile
                if not os.path.exists(arg):
                    raise ValueError(f"File path does not exist {arg}")
                ret = np.empty((0, 0), dtype=float).view(cls)
                ret = ret._load(arg, **array_args)  # pylint: disable=no-member
                kargs["metadata"] = getattr(ret, "metadata", TypeHintedDict())
                kargs["metadata"].update(user_metadata)
            case (str(),) | (Path(),):
                # Filename- load datafile
                if not os.path.exists(args[0]):
                    raise ValueError(f"File path does not exist {args[0]}")
                ret = np.empty((0, 0), dtype=float).view(cls)
                ret = ret._load(args[0], **array_args)  # pylint: disable=no-member
                kargs["metadata"] = getattr(ret, "metadata", TypeHintedDict())
                kargs["metadata"].update(user_metadata)
            case (ImageFile(),):
                # extract the image
                ret = args[0].image
                kargs["metadata"] = getattr(ret, "metadata", TypeHintedDict())
                kargs["metadata"].update(user_metadata)
            case (_,):
                try:  # try converting to a numpy array (eg a list type)
                    ret = np.asarray(args[0], **array_args).view(cls)
                    if ret.dtype == "O":  # object dtype - can't deal with this
                        raise ValueError
                except ValueError as err:  # ok couldn't load from iterable, we're done
                    raise ValueError(f"No constructor for {args[0]}") from err
            case _:
                raise ValueError(f"ImageArray expects 0 or 1 arguments, {len(args)} given")

        asfloat = kargs.pop("asfloat", False) or kargs.pop(
            "convert_float", False
        )  # convert_float for back compatibility
        if asfloat and ret.dtype.kind != "f":  # convert to float type in place
            ret = ret.convert(np.float64)

        ret.__dict__["kargs"] = kargs
        return ret

    def __array_finalize__(self, obj):
        """__array_finalize__ is a  necessary functions when subclassing numpy.ndarray to fix some behaviours.

        See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html for
        more info and examples
        Defaults below are only set when constructing an array using view
        eg np.arange(10).view(ImageArray). Otherwise filename and metadata
        attributes are just copied over (plus any other attributes set in
        _optinfo).
        """
        if not hasattr(self, "_optinfo"):
            setattr(self, "_optinfo", {"metadata": TypeHintedDict({}), "filename": ""})

        kargs = self.__dict__.pop("kargs", {})  # pylint: disable=access-member-before-definition
        tmp = metadataObject.__new__(metadataObject)
        tmp.__dict__.update(self.__dict__)  # pylint: disable=access-member-before-definition
        self.__dict__ = tmp.__dict__

        # Deal with kwargs
        user_metadata = kargs.pop("metadata", {})

        _debug = kargs.pop("debug", False)
        _title = kargs.pop("title", None)

        self.metadata.update(user_metadata)

        # all constructors call array_finalise so metadata is now initialised
        self.filename = self.metadata.setdefault("Loaded from", "")
        self.debug = _debug
        self._title = _title
        self._public_attrs = {"title": str, "filename": str}
        self._mask_color = "red"
        self._mask_alpha = 0.5

        # merge the results of __new__ from emtadataObject

        if getattr(self, "debug", False):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print(curframe, calframe)
        if obj is not None:
            self._optinfo.update(getattr(obj, "_optinfo", {}))

        super().__array_finalize__(obj=obj)

    def _load(self, filename, *args, **kargs):
        """Load an image from a file and return as a ImageArray."""
        cls = type(self)
        fmt = kargs.pop("fmt", os.path.splitext(filename)[1][1:])
        handlers = {"npy": cls._load_npy, "png": cls._load_png, "tiff": cls._load_tiff, "tif": cls._load_tiff}
        if fmt not in handlers:
            raise StonerLoadError(f"{fmt} is not a recognised format for loading.")
        ret = handlers[fmt](filename, **kargs)
        return ret

    @classmethod
    def _load_npy(cls, filename, **kargs):
        """Load image data from a numpy file."""
        image = np.load(filename)
        image = np.array(image, **kargs).view(cls)
        image.metadata["Loaded from"] = os.path.realpath(filename)  # pylint: disable=no-member
        image.filename = os.path.realpath(filename)
        return image

    @classmethod
    def _load_png(cls, filename, **kargs):  # pylint: disable=unused-argument
        """Create a new ImageArray from a png file."""
        with Image.open(filename, "r") as img:
            image = np.asarray(img).view(cls)
            # Since skimage.img_as_float() looks at the dtype of the array when mapping ranges, it's important to make
            # sure that we're not using too many bits to store the image in. This is a bit of a hack to reduce the
            # bit-depth...
            if np.issubdtype(image.dtype, np.integer):
                bits = np.ceil(np.log2(image.max()))
                if bits <= 8:
                    image = image.astype("uint8")
                elif bits <= 16:
                    image = image.astype("uint16")
                elif bits <= 32:
                    image = image.astype("uint32")
            for k in img.info:
                v = img.info[k]
                if v.startswith("b'"):
                    v = v.strip(" b'")
                    v = bytes(v)
                k = k.split("{")[0]
                image.metadata[k] = v
        image.metadata["Loaded from"] = os.path.realpath(filename)
        image.filename = os.path.realpath(filename)
        return image

    @classmethod
    def _load_tiff(cls, filename, **kargs):  # pylint: disable=unused-argument
        """Create a new ImageArray from a tiff file."""
        metadict = TypeHintedDict({})
        with Image.open(filename, "r") as img:
            image = np.asarray(img)
            if image.ndim == 3:
                image = io.imread(filename).view(np.uint32)[:, :, 0]  # Workaround for issues with more recent pillow
            tags = img.tag_v2
            if 270 in tags:
                from json import loads

                try:
                    userdata = loads(tags[270])
                    typ = userdata.get("type", cls.__name__)
                    mod = userdata.get("module", cls.__module__)

                    mod = import_module(mod)
                    typ = getattr(mod, typ)
                    if not issubclass(typ, ImageArray):
                        raise TypeError(f"Bad type in Tiff file {typ.__name__} is not a subclass of Stoner.ImageArray")
                    metadata = userdata.get("metadata", [])
                except (ValueError, TypeError, IOError):
                    metadata = []
            else:
                metadata = []
            metadict.import_all(metadata)

            # OK now try and sort out the datatype before loading
        dtype = metadict.get(
            "ImageArray.dtype", None
        )  # if tif was previously saved by Stoner then dtype should have been added to the metadata
        # If we convert to float, it's important to make
        # sure that we're not using too many bits to store the image in.
        # This is a bit of a hack to reduce the bit-depth...
        if dtype is None:
            if np.issubdtype(image.dtype, np.integer):
                bits = np.ceil(np.log2(image.max()))
                if bits <= 8:
                    dtype = "uint8"
                elif bits <= 16:
                    dtype = "uint16"
                elif bits <= 32:
                    dtype = "uint32"
            else:
                dtype = np.dtype(image.dtype).name  # retain the loaded datatype
        try:
            image = image.astype(dtype)
        except TypeError:  # Python 2.7 can throw up a bad type error here
            pass
        image = image.view(cls)
        image.update(metadict)
        image.metadata["Loaded from"] = os.path.realpath(filename)
        image.filename = os.path.realpath(filename)
        return image

    def _box(self, *args, **kargs):
        """Construct and indexing tuple for selecting areas for cropping and boxing.

        The box can be specified as:

            - (int): a fixed number of pxiels is removed from all sides
            - (float): the central region of the image is selected
            - None: the whole image is selected
            - False: The user can select a region of interest
            - (iterable of length 4) - assumed to give 4 integers to describe a specific box
        """
        if len(args) == 0 and "box" in kargs.keys():
            args = (kargs["box"],)  # back compatibility
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            args = tuple(RegionSelect()(self))
        match args:
            case (box,):
                match box:
                    case bool() if not box:
                        return slice(None, None, None), slice(None, None, None)
                    case Iterable() if len(box) == 4 and not isinstance(box, str):
                        box = [x for x in box]
                    case int() | np.int16() | np.int32() | np.int64():
                        box = [box, self.shape[1] - box, box, self.shape[0] - box]
                    case str():
                        box = self.metadata[box]
                        return self._box(*box)
                    case float() | np.float16() | np.float32() | np.float64():
                        box = [
                            round(self.shape[1] * box / 2),
                            round(self.shape[1] * (1 - box / 2)),
                            round(self.shape[1] * box / 2),
                            round(self.shape[1] * (1 - box / 2)),
                        ]
                        box = list([int(x) for x in box])
                    case _:
                        raise ValueError(f"crop accepts tuple of length 4, {len(args)} given.")
            case tuple() if len(args) in [2, 4]:
                box = list(args)
            case _:
                raise ValueError(f"crop accepts tuple of length 4, {len(args)} given.")

        for i, item in enumerate(box):  # replace None with max extent
            match item:
                case float() | np.float16() | np.float32() | np.float64() if 0 <= item <= 1:
                    if i < 2:
                        box[i] = int(round(self.shape[1] * item))
                    else:
                        box[i] = int(round(self.shape[0] * item))
                case float() | np.float16() | np.float32() | np.float64():
                    box[i] = int(round(item))
                case int() | np.int16() | np.int32() | np.int64():
                    pass
                case None:
                    box[i] = self.max_box[i]
                case _:
                    raise TypeError(f"Arguments for box should be floats, integers or None, not {type(item)}")
        return slice(box[2], box[3]), slice(box[0], box[1])

    #################################################################################################
    ################################################ Properties #####################################

    @property
    def aspect(self):
        """Return the aspect ratio (width/height) of the image."""
        return float(self.shape[1]) / self.shape[0]

    @property
    def centre(self):
        """Return the coordinates of the centre of the image."""
        return tuple(np.array(self.shape) / 2.0)

    @property
    def clone(self):
        """Duplicate the ImageFile and return the copy.

        Using .clone allows further methods to modify the clone, allowing the original immage to be unmodified.
        """
        ret = self.copy().view(type(self))
        self._optinfo["mask"] = self.mask  # Make sure we've updated our mask record
        self._optinfo["metadata"] = self.metadata  # Update metadata record
        for k, v in self._optinfo.items():
            try:
                setattr(ret, k, deepcopy(v))
            except (TypeError, ValueError, RecursionError):
                if isinstance(v, np.ndarray):
                    setattr(self, k, np.copy(v).view(v.__class__))
                else:
                    setattr(ret, k, copy(v))
        return ret

    @property
    def flat(self):
        """Return the numpy.ndarray.flat rather than a MaskedIterator."""
        return self.asarray().flat

    @property
    def max_box(self):
        """Return the maximum coordinate extent (xmin,xmax,ymin,ymax)."""
        box = (0, self.shape[1], 0, self.shape[0])
        return box

    @property
    def title(self):
        """Get a title for this image."""
        if self._title is None:
            return self.filename
        return self._title

    @title.setter
    def title(self, title):
        """Set the title of the current image."""
        if not isinstance(title, str):
            title = repr(title)
        self._title = title

    @property
    @clones
    def flip_h(self):
        """Clone the image and then mirror the image horizontally."""
        ret = self.clone[:, ::-1]
        return ret

    @property
    @clones
    def flip_v(self):
        """Clone the image and then mirror the image vertically."""
        ret = self.clone[::-1, :]
        return ret

    @property
    @clones
    def CW(self):
        """Clone the image and then rotate the imaage 90 degrees clockwise."""
        return self.clone.T[:, ::-1]

    @property
    @clones
    def CCW(self):
        """Clone the image and then rotate the imaage 90 degrees counter clockwise."""
        return self.clone.T[::-1, :]

    @property
    def draw(self):
        """Access the DrawProxy object for accessing the skimage draw sub module."""
        return DrawProxy(self, self)

    # ==============================================================================
    # OTHER SPECIAL METHODS
    # ==============================================================================

    def __getstate__(self):
        """Help with pickling ImageArrays."""
        ret = super().__getstate__()

        return {"numpy": ret, "ImageArray": {"metadata": self.metadata}}

    def __setstate__(self, state):
        """Help with pickling ImageArrays."""
        original = state.pop("numpy", tuple())
        local = state.pop("ImageArray", {})
        metadata = local.pop("metadata", {})
        super().__setstate__(original)
        self.metadata.update(metadata)

    def __delattr__(self, name):
        """Handle deleting attributes."""
        super().__delattr__(name)
        if name in self._optinfo:
            del self._optinfo[name]

    def __setattr__(self, name, value):
        """Set an attribute on the object."""
        super().__setattr__(name, value)
        # add attribute to those for copying in array_finalize. use value as
        # default.
        circ = ["_optinfo", "mask", "__dict__"]  # circular references
        proxy = ["_funcs"]  # can be reloaded for cloned arrays
        if name in circ + proxy:
            # Ignore these in clone
            pass
        else:
            self._optinfo.update({name: value})

    def __getitem__(self, index):
        """Patch indexing of strings to metadata."""
        if getattr(self, "debug", False):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print(curframe, calframe)
        if isinstance(index, ImageFile) and index.image.dtype == bool:
            index = index.image
        if isinstance(index, string_types):
            return self.metadata[index]
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        """Patch string index through to metadata."""
        if isinstance(index, ImageFile) and index.dtype == bool:
            index = index.image
        if isinstance(index, string_types):
            self.metadata[index] = value
        else:
            super().__setitem__(index, value)

    def __delitem__(self, index):
        """Patch indexing of strings to metadata."""
        if isinstance(index, string_types):
            del self.metadata[index]
        else:
            super().__delitem__(index)

    def save(self, filename=None, **kargs):
        """Stub method for a save function."""
        raise NotImplementedError(f"Save is not implemented in {self.__class__}")


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
)
@class_modifier(
    [ndi],
    transpose=True,
    adaptor=image_file_adaptor,
)
@class_modifier(imagefuncs, overload=True, adaptor=image_file_adaptor)
@class_wrapper(target=ImageArray, exclude_below=metadataObject)
class ImageFile(metadataObject):
    """An Image file type that is analogous to :py:class:`Stoner.Data`.

    This contains metadata and an image attribute which
    is an :py:class:`Stoner.Image.ImageArray` type which subclasses numpy ndarray and
    adds lots of extra image specific processing functions.

    Attributes:
        image (:py:class:`Stoner.Image.ImageArray`):
            A :py:class:`numpy.ndarray` subclass that stores the actual image data.
        metadata (:py:class:`Stoner.core.regexpDict`):
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

    The ImageFile owned attribute is image. All other calls including metadata
    are passed through to ImageArray (so no need to inherit from metadataObject).

    Almost all calls to ImageFile are passed through to the underlying ImageArray
    logic and ImageArray can be used as a standalone class.
    However because ImageArray subclasses an ndarray it is not possible to enter
    it in place. All attributes return an array instance which needs to be reassigned.
    ImageFile owns image and so can change in place.
    The penalty is that numpy ufuncs don't return ImageFile type

    so can do::

        imfile.asfloat() #imagefile.image is updated to float type however need to do:
        imfile.image = np.abs(imfile.image)

    whereas for imarray need to do::

            imarray = imagearray.asfloat()

    but::

            np.abs(imarray) #returns ImageArray type
    """

    # pylint: disable=no-member

    _protected_attrs = ["_fromstack"]  # these won't be passed through to self.image attrs
    _patterns = ["*.png", "*.tif", "*.jpeg", "*.jpg"]
    mime_type = ["image/png", "image/jpeg", "image/tiff", "application/octet-stream"]
    priority = 32

    def __init__(self, *args, **kargs):
        """Mostly a pass through to ImageArray constructor.

        Local attribute is image. All other attributes and calls are passed
        through to image attribute.

        There is one special case of creating an ImageFile from a :py:class:`Stoner.Core.DataFile`. In this case the
        the DataFile is assumed to contain (x,y,z) data that should be converted to a map of
        z on a regular grid of x,y. The columns for the x,y,z data can be taken from the DataFile's
        :py:attr:`Stoner.Core.DataFile.setas` attribute or overridden by providing xcol, ycol and zcol keyword
        arguments. A further *shape* keyword can spewcify the shape as a tuple or "unique" to use the unique values of
        x and y or if omitted asquare grid will be interpolated.

        """
        self._image = ImageArray()  # Ensire we have the image data in place
        super().__init__(*args, **kargs)
        args = list(args)
        if len(args) == 0:
            pass
        elif len(args) > 0 and isinstance(args[0], path_types):
            try:
                copy_into(self.__class__.load(args[0], **kargs), self)
                self._public_attrs = {"title": str, "filename": str}
                self._fromstack = kargs.pop("_fromstack", False)  # for use by ImageStack
                return
            except StonerLoadError:
                args[0] = ImageArray(*args, **kargs)
        if len(args) > 0 and isinstance(args[0], ImageFile):  # Fixing type
            self._image = args[0].image
            for k in args[0]._public_attrs:
                setattr(self, k, getattr(args[0], k, None))
        elif len(args) > 0 and isinstance(args[0], np.ndarray):  # Fixing type
            self._image = ImageArray(*args, **kargs)
            if isinstance(args[0], ImageArray):
                for k in args[0]._public_attrs:
                    setattr(self, k, getattr(args[0], k, None))
        elif len(args) > 0 and isinstance(
            args[0], DataFile
        ):  # Support initing from a DataFile that defines x,y,z coordinates
            self._init_from_datafile(*args, **kargs)
        self._public_attrs = {"title": str, "filename": str}
        self._fromstack = kargs.pop("_fromstack", False)  # for use by ImageStack

    ###################################################################################################################
    ############################# Properties #### #####################################################################

    @property
    def _repr_png_(self):
        return self._repr_png_private_

    @property
    def clone(self):
        """Make a copy of this ImageFile."""
        new = type(self)(self.image.clone)
        for attr in self.__dict__:
            if callable(getattr(self, attr)) or attr in ["image", "metadata"] or attr.startswith("_"):
                continue
            try:
                setattr(new, attr, deepcopy(getattr(self, attr)))
            except NotImplementedError:  # Deepcopying failed, so just copy a reference instead
                setattr(new, attr, getattr(self, attr))
        return new

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
        return DrawProxy(self.image, self)

    @property
    def image(self):
        """Access the image data."""
        return self._image

    @image.setter
    def image(self, v):
        """Ensure stored image is always an ImageArray."""
        filename = self._image.filename
        metadata = self._image.metadata
        # ensure setting image goes into the same memory block if from stack
        if (
            hasattr(self, "_fromstack")
            and self._fromstack
            and self._image.shape == v.shape
            and self._image.dtype == v.dtype
        ):
            self._image[:] = np.copy(v)
        elif isinstance(v, np.ndarray):
            self._image = np.copy(v).view(ImageArray)
        else:
            self._image = ImageArray(v)
        self.filename = filename
        self._image.metadata.update(metadata)
        self._image.metadata.update(getattr(v, "metadata", {}))

    @property
    def mask(self):
        """Get the mask of the underlying IamgeArray."""
        return MaskProxy(self)

    @mask.setter
    def mask(self, value):
        """Set the underlying ImageArray's mask."""
        if isinstance(value, ImageFile):
            value = value.image
        if isinstance(value, MaskProxy):
            value = value._mask
        self.image.mask = value

    ###################################################################################################################
    ############################# Special methods #####################################################################

    def __getitem__(self, n):
        """Pass through to ImageArray."""
        try:
            ret = self.image.__getitem__(n)
            if isinstance(ret, ImageArray) and ret.ndim == 2:
                retval = self.clone
                retval.image = ret
                return retval
            return ret
        except KeyError:
            if n not in self.metadata and n in self._image.metadata:
                self.metadata[n] = self._image.metadata[n]
            return self.metadata.__getitem__(n)

    def __setitem__(self, n, v):
        """Pass through to ImageArray."""
        if isinstance(n, string_types):
            self.metadata.__setitem__(n, v)
        else:
            self.image.__setitem__(n, v)

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

    def __delitem__(self, n):
        """Pass through to ImageArray."""
        try:
            self.image.__delitem__(n)
        except KeyError:
            self.metadata.__delitem__(n)

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

    def _init_from_datafile(self, *args, **kargs):
        """Initialise ImageFile from DataFile defining x,y,z coordinates.

        Args:
            args[0] (DataFile):
                A :py:class:`Stoner.Core.DataFile` instance that defines x,y,z coordinates or has columns specified
                in keywords.

        Keyword Args:
            xcol (column index):
                Column in the DataFile that has the x-co-ordinate
            ycol (column index):
                Column in the data file that defines the y-cordinate
            zcol (column index):
                Column in the datafile that defines the intensity
        """
        data = make_Data(args[0])
        shape = kargs.pop("shape", "unique")

        _ = data._col_args(**kargs)
        data.setas(x=_.xcol, y=_.ycol, z=_.zcol)  # pylint: disable=not-callable
        if isinstance(shape, string_types) and shape == "unique":
            shape = (len(np.unique(data.x)), len(np.unique(data.y)))
        elif isTuple(shape, int_types, int_types):
            pass
        else:
            shape = None
        X, Y, Z = data.griddata(_.xcol, _.ycol, _.zcol, shape=shape)
        self.image = Z.view(ImageArray)
        self.metadata = deepcopy(data.metadata)
        self["x_vector"] = np.unique(X)
        self["y_vector"] = np.unique(Y)

    def _load(self, filename, *args, **kargs):
        """Load an ImageFile by calling the ImageArray method instead."""
        self._image = ImageArray(filename, *args, **kargs)
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

        Note:
            The filename attribute of the current instance is updated by this method as well.
        """
        self.filename = file_dialog(mode, self.filename, type(self), ImageFile)
        return self.filename

    @classmethod
    def load(cls, *args, **kargs):
        """Create a :py:class:`ImageFile` from file abnd guessing a better subclass if necessary.

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

        Note:
            If *filetupe* is a string, then it is first tried as an exact match to a subclass name, otherwise it
            is used as a partial match and the first class in priority order is that matches is used.

            Some subclasses can be found in the :py:mod:`Stoner.formats` package.

            Each subclass is scanned in turn for a class attribute :py:attr:`Stoner.ImnageFile.priority` which governs
            the order in which they are tried. Subclasses which can make an early positive determination that a
            file has the correct format can have higher priority levels. Classes should return a suitable exception
            if they fail to load the file.

            If no class can load a file successfully then a RunttimeError exception is raised.
        """
        filename, args, kargs = get_filename(args, kargs)
        filetype = kargs.pop("filetype", None)
        debug = kargs.pop("debug", False)
        auto_load = kargs.pop("auto_load", filetype is None)
        if isinstance(filename, path_types) and urllib.parse.urlparse(str(filename)).scheme not in URL_SCHEMES:
            filename, filetype = get_file_name_type(filename, filetype, ImageFile)
        if filename is None or not filename:
            filename = file_dialog("r", filename, "ImageFile", ImageFile)
        elif not auto_load and not filetype:
            raise StonerLoadError("Cannot read data from non-path like filenames !")
        if auto_load:  # We're going to try every subclass we canA
            try:
                ret = auto_load_classes(filename, "Image", debug=debug, args=args, kargs=kargs)
            except StonerUnrecognisedFormat:
                ret = ImageFile()
                ret = ret._load(filename, *args, **kargs)
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
                ret = loader(make_Image(), filename, *args, **kargs)
                ret["Loaded as"] = filetype
            except StonerLoadError as err:
                raise ValueError(f"Unable to load {filename}") from err

        for k, i in kargs.items():
            if not callable(getattr(ret, k, lambda x: False)):
                setattr(ret, k, i)
        ret._kargs = kargs
        return ret

    def save(self, filename=None, **kargs):
        """Save the image into the file 'filename'.

        Args:
            filename (string, bool or None):
                Filename to save data as, if this is None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If filename is False then a file dialog is forced.

        Keyword Args:
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
        self.image.save(filename, **kargs)
