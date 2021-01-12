# -*- coding: utf-8 -*-
"""Implements core image handling classes for the :mod:`Stoner.Image` package."""
__all__ = ["ImageArray", "ImageFile"]
import os
from copy import copy, deepcopy
import inspect
from importlib import import_module
from functools import wraps
from PIL import Image
from io import BytesIO as StreamIO

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
    viewer,
)

from ..core.base import typeHintedDict, metadataObject
from ..Core import DataFile
from ..tools import isTuple, fix_signature, isLikeList, make_Data
from ..tools.decorators import class_modifier, image_file_adaptor, class_wrapper, clones
from ..compat import (
    string_types,
    get_filedialog,
    int_types,
    path_types,
)  # Some things to help with Python2 and Python3 compatibility
from .attrs import DrawProxy, MaskProxy
from .widgets import RegionSelect
from . import imagefuncs

IMAGE_FILES = [("Tiff File", "*.tif;*.tiff"), ("PNG files", "*.png", "Numpy Files", "*.npy")]

dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-(2 ** 63), 2 ** 63 - 1),
    np.uint64: (0, 2 ** 64 - 1),
    np.int32: (-(2 ** 31), 2 ** 31 - 1),
    np.uint32: (0, 2 ** 32 - 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}


def __add_core__(result, other):
    """Actually do result=result-other."""
    if isinstance(other, type(result)) and result.shape == other.shape:
        result.image += other.image
    elif isinstance(other, np.ndarray) and other.shape == result.shape:
        result.image += other
    elif isinstance(other, (int, float)):
        result.image += other
    else:
        return NotImplemented
    return result


def __floor_div_core__(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageArray

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image // other
    return result


def __div_core__(result, other):
    """Actually do result=result/other."""
    # Cheat and pass through to ImageArray

    if isinstance(other, ImageFile):
        other = other.image

    result.image = result.image / other
    return result


def __sub_core__(result, other):
    """Actually do result=result-other."""
    if isinstance(other, type(result)) and result.shape == other.shape:
        result.image -= other.image
    elif isinstance(other, np.ndarray) and other.shape == result.shape:
        result.image -= other
    elif isinstance(other, (int, float)):
        result.image -= other
    else:
        return NotImplemented
    return result


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
        viewer,
    ]
)
@class_modifier([ndi.interpolation, ndi.filters, ndi.measurements, ndi.morphology, ndi.fourier], transpose=True)
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
            The name of the file from which this image was laoded.
        title (str):
            The title of the image (defaults to the filename).
        mask (:py:class:`numpy.ndarray of bool`):
            The underlying mask data of the image. Masked elements (i.e. where mask=True) are ignored for many
            image operations. Indexing them will return the mask fill value (typically NaN, ot -1 or -MAXINT)
            draw (:py:class:`Stoner.Image.attrs.DrawProxy`):
            A sepcial object that allows the user to manipulate the image data by making use of
            :py:mod:`skimage.draw` functions as well as some additional drawing functions.
        clone (:py:class:`Stoner.ImageArry`):
            Return a duplicate copy of the current image - this allows subsequent methods to
            modify the cloned version rather than the original version.
        centre (tuple of (float,float)):
            The coordinates of the centre of the image.
        aspect (float):
            The aspect ratio (width/height) of the image.
        max_box (tuple (0,x-size,0-y-size)):
            The extent of the iamge size in a form suitable for use in defining a box.
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

    # Proxy attributess for storing imported functions. Only do the import when needed
    _func_proxy = None

    # extra attributes for class beyond standard numpy ones
    _extra_attributes_default = {"metadata": typeHintedDict({}), "filename": ""}

    # Default values for when we can't find the attribute already
    _defaults = {"debug": False, "_hardmask": False}

    fmts = ["png", "npy", "tiff", "tif"]

    # These will be overriden with isntance attributes, but setting here allows ImageFile properties to be defined.
    filename = None
    debug = False
    filename = ""

    # now initialise class

    def __new__(cls, *args, **kargs):
        """Construct an ImageArray object.

        We're using __new__ rather than __init__ to imitate a numpy array as
        close as possible.
        """
        if len(args) not in [0, 1]:
            raise ValueError(f"ImageArray expects 0 or 1 arguments, {len(args)} given")

        # Deal with kwargs
        array_arg_keys = ["dtype", "copy", "order", "subok", "ndmin", "mask"]  # kwargs for array setup
        array_args = {k: kargs.pop(k) for k in array_arg_keys if k in kargs.keys()}
        user_metadata = kargs.pop("metadata", {})
        asfloat = kargs.pop("asfloat", False) or kargs.pop(
            "convert_float", False
        )  # convert_float for back compatability
        _debug = kargs.pop("debug", False)
        _title = kargs.pop("title", None)

        # 0 args initialisation
        if len(args) == 0:
            ret = np.empty((0, 0), dtype=float).view(cls)
            # merge the results of __new__ from emtadataObject
            tmp = metadataObject.__new__(metadataObject, *args, **kargs)
            for k, v in tmp.__dict__.items():
                if k not in ret.__dict__:
                    ret.__dict__[k] = v

        else:

            # 1 args initialisation
            arg = args[0]
            loadfromfile = False
            if isinstance(arg, cls):
                ret = arg
            elif isinstance(arg, np.ndarray):
                # numpy array or ImageArray)
                if arg.ndim < 2:
                    ret = np.atleast_2d(arg).view(ImageArray)
                else:
                    ret = arg.view(ImageArray)
                ret.metadata = getattr(arg, "metadata", typeHintedDict())
            elif isinstance(arg, bool) and not arg:
                patterns = (("png", "*.png"), ("npy", "*.npy"))
                arg = get_filedialog(what="r", filetypes=patterns)
                if len(arg) == 0:
                    raise ValueError("No file given")
                loadfromfile = True
            elif isinstance(arg, path_types) or loadfromfile:
                # Filename- load datafile
                if not os.path.exists(arg):
                    raise ValueError(f"File path does not exist {arg}")
                ret = ret = np.empty((0, 0), dtype=float).view(cls)
                ret = ret._load(arg, **array_args)  # pylint: disable=no-member
            elif isinstance(arg, ImageFile):
                # extract the image
                ret = arg.image
            else:
                try:  # try converting to a numpy array (eg a list type)
                    ret = np.asarray(arg, **array_args).view(cls)
                    if ret.dtype == "O":  # object dtype - can't deal with this
                        raise ValueError
                except ValueError as err:  # ok couldn't load from iterable, we're done
                    raise ValueError(f"No constructor for {arg}") from err

            if asfloat and ret.dtype.kind != "f":  # convert to float type in place
                meta = ret.metadata  # preserve any metadata we may already have
                ret = ret.convert(np.float64)
                ret.metadata.update(meta)

            # merge the results of __new__ from emtadataObject
            tmp = metadataObject.__new__(metadataObject, *args, **kargs)
            for k, v in tmp.__dict__.items():
                if k not in ret.__dict__:
                    ret.__dict__[k] = v

        # all constructors call array_finalise so metadata is now initialised
        if "Loaded from" not in ret.metadata.keys():
            ret.metadata["Loaded from"] = ""
        ret.filename = ret.metadata["Loaded from"]
        ret.metadata.update(user_metadata)
        ret.debug = _debug
        ret._title = _title
        ret._public_attrs = {"title": str, "filename": str}
        ret._mask_color = "red"
        ret._mask_alpha = 0.5
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
        if getattr(self, "debug", False):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print(curframe, calframe)
        _extra_attributes = getattr(obj, "_optinfo", ImageArray._extra_attributes_default)
        setattr(self, "_optinfo", copy(_extra_attributes))
        for k, v in list(_extra_attributes.items()):
            try:
                setattr(self, k, getattr(obj, k, v))
            except AttributeError:  # Some versions of  python don't like this
                pass
        super().__array_finalize__(obj)

    def _load(self, filename, *args, **kargs):
        """Load an image from a file and return as a ImageArray."""
        cls = type(self)
        fmt = kargs.pop("fmt", os.path.splitext(filename)[1][1:])
        handlers = {"npy": cls._load_npy, "png": cls._load_png, "tiff": cls._load_tiff, "tif": cls._load_tiff}
        if fmt not in handlers:
            raise ValueError(f"{fmt} is not a recognised format for loading.")
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
                k = k.split("{")[0]
                image.metadata[k] = v
        image.metadata["Loaded from"] = os.path.realpath(filename)
        image.filename = os.path.realpath(filename)
        return image

    @classmethod
    def _load_tiff(cls, filename, **kargs):  # pylint: disable=unused-argument
        """Create a new ImageArray from a tiff file."""
        metadict = typeHintedDict({})
        with Image.open(filename, "r") as img:
            image = np.asarray(img)
            if image.ndim == 3:
                if image.shape[2] < 4:  # Need to add a dummy alpha channel
                    image = np.append(np.zeros_like(image[:, :, 0]), axis=2)
                image = image.view(dtype=np.uint32).reshape(image.shape[:-1])
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
            args = [kargs["box"]]  # back compatability
        elif len(args) not in (0, 1, 4):
            raise ValueError("box accepts 1 or 4 arguments, {len(args)} given.")
        if len(args) == 0 or (len(args) == 1 and args[0] is None):
            args = RegionSelect()(self)
        if len(args) == 1:
            box = args[0]
            if isinstance(box, bool) and not box:  # box=False is the same as all values
                return slice(None, None, None), slice(None, None, None)
            if isLikeList(box) and len(box) == 4:  # Full box as a list
                box = [x for x in box]
            elif isinstance(box, int):  # Take a border of n pixels out
                box = [box, self.shape[1] - box, box, self.shape[0] - box]
            elif isinstance(box, string_types):
                box = self.metadata[box]
                return self._box(*box)
            elif isinstance(box, float):  # Keep the central fraction of the image
                box = [
                    round(self.shape[1] * box / 2),
                    round(self.shape[1] * (1 - box / 2)),
                    round(self.shape[1] * box / 2),
                    round(self.shape[1] * (1 - box / 2)),
                ]
                box = list([int(x) for x in box])
            else:
                raise ValueError(f"crop accepts tuple of length 4, {len(box)} given.")
        else:
            box = list(args)
        for i, item in enumerate(box):  # replace None with max extent
            if isinstance(item, float) and 0 <= item <= 1:
                if i < 2:
                    box[i] = int(round(self.shape[1] * item))
                else:
                    box[i] = int(round(self.shape[0] * item))
            elif isinstance(item, float):
                box[i] = int(round(item))
            elif isinstance(item, int_types):
                pass
            elif item is None:
                box[i] = self.max_box[i]
            else:
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
        for k, v in self._optinfo.items():
            try:
                setattr(ret, k, deepcopy(v))
            except (TypeError, ValueError, RecursionError):
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
        """Access the DrawProxy opbject for accessing the skimage draw sub module."""
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
        # defualt.
        circ = ["_optinfo", "mask"]  # circular references
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
        viewer,
    ],
    adaptor=image_file_adaptor,
)
@class_modifier(
    [ndi.interpolation, ndi.filters, ndi.measurements, ndi.morphology, ndi.fourier],
    transpose=True,
    adaptor=image_file_adaptor,
)
@class_modifier(imagefuncs, overload=True, adaptor=image_file_adaptor)
@class_wrapper(target=ImageArray, exclude_below=metadataObject)
class ImageFile(metadataObject):

    """An Image file type that is analagous to :py:class:`Stoner.Data`.

    This contains metadata and an image attribute which
    is an :py:class:`Stoner.Image.ImageArray` type which subclasses numpy ndarray and
    adds lots of extra image specific processing functions.

    Attributes:
        image (:py:class:`Stoner.Image.ImageArray`):
            A :py:class:`numpy.ndarray` subclass that stores the actual image data.
        metadata (:py:class:`Stoner.core.regexpDict`):
            A dictionary of metadata items associated with this image.
        filename (str):
            The name of the file from which this image was laoded.
        title (str):
            The title of the image (defaults to the filename).
        mask (:py:class:`Stoner.Image.attrs.MaskProxy`):
            A special object that allows manipulation of the image's mask - thius allows the
            user to selectively disable regions of the image from rpocessing functions.
        draw (:py:class:`Stoner.Image.attrs.DrawProxy`):
            A sepcial object that allows the user to manipulate the image data by making use of
            :py:mod:`skimage.draw` functions as well as some additional drawing functions.
        clone (:py:class:`Stoner.ImageFile`):
            Return a duplicate copy of the current image - this allows subsequent methods to
            modify the cloned version rather than the original version.
        centre (tuple of (int,int)):
            The coordinates of the centre of the image.
        aspect (float):
            The aspect ratio (width/height) of the image.
        max_box (tuple (0,x-size,0-y-size)):
            The extent of the iamge size in a form suitable for use in defining a box.
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

    _image = None
    _protected_attrs = ["_fromstack"]  # these won't be passed through to self.image attrs
    _patterns = ["*.png", "*.tif", "*.jpeg", "*.jpg"]

    def __init__(self, *args, **kargs):
        """Mostly a pass through to ImageArray constructor.

        Local attribute is image. All other attributes and calls are passed
        through to image attribute.

        There is one special case of creating an ImageFile from a :py:class:`Stoner.Core.DataFile`. In this case the
        the DataFile is assummed to contain (x,y,z) data that should be converted to a map of
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
        """Access the DrawProxy opbject for accessing the skimage draw sub module."""
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
            self._image[:] = v
        else:
            self._image = ImageArray(v)
        self.filename = filename
        self.image.metadata.update(metadata)

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
            if isinstance(ret, ImageArray):
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
        obj, attr = self._where_attr(n)
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
        result = __add_core__(result, other)
        return result

    def __iadd__(self, other):
        """Implement the inplace subtract operator."""
        result = self
        result = __add_core__(result, other)
        return result

    def __truediv__(self, other):
        """Implement the divide operator."""
        result = self.clone
        result = __div_core__(result, other)
        return result

    def __itruediv__(self, other):
        """Implement the inplace divide operator."""
        result = self
        result = __div_core__(result, other)
        return result

    def __sub__(self, other):
        """Implement the subtract operator."""
        result = self.clone
        result = __sub_core__(result, other)
        return result

    def __isub__(self, other):
        """Implement the inplace subtract operator."""
        result = self
        result = __sub_core__(result, other)
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

    def __floordiv__(self, other):
        """Calculate and XMCD ratio on the images."""
        if not isinstance(other, type(self)):  # Only do XMCD type operations on ImageFiles of the same type
            result = self
            return __floor_div_core__(result, other)
        if self.image.dtype != other.image.dtype:
            raise ValueError(
                "Only ImageFiles with the same type of underlying image data can be used to calculate an XMCD ratio."
                + "Mimatch is {self.image.dtype} vs {other.image.dtype}"
            )
        if self.image.dtype.kind != "f":
            ret = self.clone.convert(float)
            other = other.clone.convert(float)
        else:
            ret = self.clone
        ret.image = (ret.image - other.image) / (ret.image + other.image)
        return ret

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
        """Initialise ImageFile from DataFile defining x,y,z co-ordinates.

        Args:
            args[0] (DataFile):
                A :py:class:`Stoner.Core.DataFile` instance that defines x,y,z co-ordinates or has columns specified
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
