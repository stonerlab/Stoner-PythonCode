# -*- coding: utf-8 -*-
"""Implements core image handling classes for the :mod:`Stoner.Image` package."""
__all__ = ["ImageArray", "ImageFile", "DrawProxy", "MaskProxy"]
import numpy as np
import os
import warnings
from copy import copy, deepcopy
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
    draw,
)
from PIL import Image
from PIL import PngImagePlugin  # for saving metadata
import matplotlib.pyplot as plt
from Stoner.Core import typeHintedDict, metadataObject, regexpDict, DataFile
from Stoner.Image.util import convert
from Stoner import Data
from Stoner.tools import istuple, fix_signature, islike_list, get_option
from Stoner.compat import (
    python_v3,
    string_types,
    get_filedialog,
    int_types,
)  # Some things to help with Python2 and Python3 compatibility
import inspect
from functools import wraps

if python_v3:
    from io import BytesIO as StreamIO
else:
    from cStringIO import StringIO as StreamIO


dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-2 ** 63, 2 ** 63 - 1),
    np.uint64: (0, 2 ** 64 - 1),
    np.int32: (-2 ** 31, 2 ** 31 - 1),
    np.uint32: (0, 2 ** 32 - 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}


def __add_core__(result, other):
    """Actually do result=result-other."""
    if isinstance(other, result.__class__) and result.shape == other.shape:
        result.image += other.image
    elif isinstance(other, np.ndarray) and other.shape == result.shape:
        result.image += other
    elif isinstance(other, (int, float)):
        result.image += other
    else:
        return NotImplemented
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
    if isinstance(other, result.__class__) and result.shape == other.shape:
        result.image -= other.image
    elif isinstance(other, np.ndarray) and other.shape == result.shape:
        result.image -= other
    elif isinstance(other, (int, float)):
        result.image -= other
    else:
        return NotImplemented
    return result


class ImageArray(np.ma.MaskedArray, metadataObject):

    """:py:class:`Stoner.Image.core.ImageArray` is a numpy array like class with a metadata parameter and pass through to skimage methods.

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
        metadata (dict):
            dictionary of metadata for the image
        clone (self):
            copy of self
        max_box (tuple):
            coordinate extent (xmin,xmax,ymin,ymax)


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

    # now initialise class

    if not python_v3:  # Ugh what a horrible hack!
        _mask = np.ma.MaskedArray([]).mask

    def __new__(cls, *args, **kargs):
        """Construct an ImageArray object.

        We're using __new__ rather than __init__ to imitate a numpy array as
        close as possible.
        """

        if len(args) not in [0, 1]:
            raise ValueError("ImageArray expects 0 or 1 arguments, {} given".format(len(args)))

        ### Deal with kwargs
        array_arg_keys = ["dtype", "copy", "order", "subok", "ndmin", "mask"]  # kwargs for array setup
        array_args = {k: kargs.pop(k) for k in array_arg_keys if k in kargs.keys()}
        user_metadata = kargs.pop("metadata", {})
        asfloat = kargs.pop("asfloat", False) or kargs.pop(
            "convert_float", False
        )  # convert_float for back compatability
        _debug = kargs.pop("debug", False)

        ### 0 args initialisation
        if len(args) == 0:
            ret = np.empty((0, 0), dtype=float).view(cls)
        else:

            ### 1 args initialisation
            arg = args[0]
            loadfromfile = False
            if isinstance(arg, cls):
                ret = arg
            elif isinstance(arg, np.ndarray):
                # numpy array or ImageArray)
                if len(arg.shape) == 2:
                    ret = arg.view(ImageArray)
                elif len(arg.shape) == 1:
                    arg = np.array([arg])  # force 2d array
                    ret = arg.view(ImageArray)
                else:
                    raise ValueError("Array dimension 0 must be at most 2. Got {}".format(len(arg.shape)))
                ret.metadata = getattr(arg, "metadata", typeHintedDict())
            elif isinstance(arg, bool) and not arg:
                patterns = (("png", "*.png"), ("npy", "*.npy"))
                arg = get_filedialog(what="r", filetypes=patterns)
                if len(arg) == 0:
                    raise ValueError("No file given")
                else:
                    loadfromfile = True
            elif isinstance(arg, string_types) or loadfromfile:
                # Filename- load datafile
                if not os.path.exists(arg):
                    raise ValueError("File path does not exist {}".format(arg))
                ret = ret = np.empty((0, 0), dtype=float).view(cls)
                ret = ret._load(arg, **array_args)
            elif isinstance(arg, ImageFile):
                # extract the image
                ret = arg.image
            else:
                try:  # try converting to a numpy array (eg a list type)
                    ret = np.asarray(arg, **array_args).view(cls)
                    if ret.dtype == "O":  # object dtype - can't deal with this
                        raise ValueError
                except ValueError:  # ok couldn't load from iterable, we're done
                    raise ValueError("No constructor for {}".format(type(arg)))
            if asfloat and ret.dtype.kind != "f":  # convert to float type in place
                meta = ret.metadata  # preserve any metadata we may already have
                ret = convert(ret, np.float64).view(ImageArray)
                ret.metadata.update(meta)

        # all constructors call array_finalise so metadata is now initialised
        if "Loaded from" not in ret.metadata.keys():
            ret.metadata["Loaded from"] = ""
        ret.filename = ret.metadata["Loaded from"]
        ret.metadata.update(user_metadata)
        ret.debug = _debug
        return ret

    def __array_finalize__(self, obj):
        """__array_finalize__ and __array_wrap__ are necessary functions when subclassing numpy.ndarray to fix some behaviours.

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
        _extra_attributes = getattr(obj, "_optinfo", deepcopy(ImageArray._extra_attributes_default))
        setattr(self, "_optinfo", copy(_extra_attributes))
        for k, v in _extra_attributes.items():
            try:
                setattr(self, k, getattr(obj, k, v))
            except AttributeError:  # Some versions of  python don't like this
                pass
        super(ImageArray, self).__array_finalize__(obj)

    def __array_prepare__(self, arr, context=None):
        """Support the numpy machinery for subclassing ndarray."""
        return super(ImageArray, self).__array_prepare__(arr, context)

    def __array_wrap__(self, out_arr, context=None):
        """Part of the numpy array machinery.

        see __array_finalize__ for info. This is for if ImageArray is called
        via ufuncs. array_finalize is called after.
        """
        ret = super(ImageArray, self).__array_wrap__(out_arr, context)
        return ret

    def __init__(self, *args, **kwargs):
        """Constructor method for :py:class:`ImageArray`.

        various forms are recognised

        .. py:function:: ImageArray('filename')
            :noindex:

            Creates the new ImageArray object and then loads data from the
            given *filename*.

        .. py:function:: ImageArray(array)
            :noindex:

            Creates a new ImageArray object and assigns the *array*.

        .. py:function:: ImageArray(ImageFile)
            :noindex:

            Creates a new ImageArray object and assigns the *array* using
            the ImageFile.image property.


        .. py:function:: ImageArray(ImageArray)
            :noindex:

            Creates the new ImageArray object and initialises all data from the
            existing :py:class:`ImageArray` instance. This on the face of it does the same as
            the assignment operator, but is more useful when one or other of the
            ImageArray objects is an instance of a sub - class of ImageArray

        .. py:function:: ImageArray(bool)
            :noindex:

           if arg[0] is bool and False then open a file dialog to locate the
           file to open.

        Args:
            arg (positional arguments): an argument that matches one of the
                definitions above
        Keyword Arguments: All keyword arguments that match public attributes are
                used to set those public attributes eg metadata.

                asfloat(bool):
                    if True  and loading the image from file, convert the image
                    to float values between 0 and 1 (necessary for some forms
                    of processing)
        """
        super(ImageArray, self).__init__(*args, **kwargs)

    def _load(self, filename, *args, **kargs):
        """Load an image from a file and return as a ImageArray."""
        cls = self.__class__
        fmt = kargs.pop("fmt", os.path.splitext(filename)[1][1:])
        handlers = {"npy": cls._load_npy, "png": cls._load_png, "tiff": cls._load_tiff, "tif": cls._load_tiff}
        if fmt not in handlers:
            raise ValueError("{} is not a recognised format for loading.".format(fmt))
        ret = handlers[fmt](filename, **kargs)
        return ret

    @classmethod
    def _load_npy(cls, filename, **kargs):
        """Load image data from a numpy file."""
        image = np.load(filename)
        image = np.array(image, **kargs).view(cls)
        image.metadata["Loaded from"] = os.path.realpath(filename)
        image.filename = os.path.realpath(filename)
        return image

    @classmethod
    def _load_png(cls, filename, **kargs):
        """Create a new ImageArray from a png file."""
        with Image.open(filename, "r") as img:
            image = np.asarray(img).view(cls)
            # Since skimage.img_as_float() looks at the dtype of the array when mapping ranges, it's important to make
            # sure that we're not using too many bits to store the image in. This is a bit of a hack to reduce the bit-depth...
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
                if "b'" in v:
                    v = v.strip(" b'")
                image.metadata[k] = v
        image.metadata["Loaded from"] = os.path.realpath(filename)
        image.filename = os.path.realpath(filename)
        return image

    @classmethod
    def _load_tiff(cls, filename, **kargs):
        """Create a new ImageArray from a tiff file."""
        metadict = typeHintedDict({})
        with Image.open(filename, "r") as img:
            image = np.asarray(img)
            tags = img.tag_v2
            if 270 in tags:
                from json import loads

                try:
                    metadata_string = tags[270]
                    metadata = loads(metadata_string)
                except Exception:
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
            args = kargs["box"]  # back compatability
        elif len(args) not in (1, 4):
            raise ValueError("box accepts 1 or 4 arguments, {} given.".format(len(args)))
        if len(args) == 1:
            box = args[0]
            if isinstance(box, bool) and not box:  # experimental
                print("Select crop area")
                box = self.draw_rectangle(box)
            elif islike_list(box) and len(box) == 4:  # Full box as a list
                box = [x for x in box]
            elif box is None:  # Whole image
                box = [0, self.shape[1], 0, self.shape[0]]
            elif isinstance(box, int):  # Take a border of n pixels out
                box = [box, self.shape[1] - box, box, self.shape[0] - box]
            elif isinstance(box, float):  # Keep the central fraction of the image
                box = [
                    round(self.shape[1] * box / 2),
                    round(self.shape[1] * (1 - box / 2)),
                    round(self.shape[1] * box / 2),
                    round(self.shape[1] * (1 - box / 2)),
                ]
                box = list([int(x) for x in box])
            else:
                raise ValueError("crop accepts tuple of length 4, {} given.".format(len(box)))
        else:
            box = list(args)
        for i, item in enumerate(box):  # replace None with max extent
            if isinstance(item, float):
                if i < 2:
                    box[i] = round(self.shape[1] * item)
                else:
                    box[i] = round(self.shape[0] * item)
            elif isinstance(item, int_types):
                pass
            elif item is None:
                box[i] = self.max_box[i]
            else:
                raise TypeError("Arguments for box should be floats, integers or None, not {}".format(type(item)))
        return slice(box[2], box[3]), slice(box[0], box[1])

    #################################################################################################
    ############ Properties #########################################################################

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
        """return a copy of the instance"""
        ret = ImageArray(np.copy(self))
        self._optinfo["mask"] = self.mask  # Make sure we've updated our mask record
        for k, v in self._optinfo.items():
            try:
                setattr(ret, k, deepcopy(v))
            except Exception:
                setattr(ret, k, copy(v))
        return ret

    @property
    def flat(self):
        """MaskedArray.flat doesn't work the same as array.flat."""
        return np.asarray(self).flat

    @property
    def max_box(self):
        """return the coordinate extent (xmin,xmax,ymin,ymax)"""
        box = (0, self.shape[1], 0, self.shape[0])
        return box

    #    @property
    #    def data(self):
    #        """alias for image[:]. Equivalence to Stoner.data behaviour"""
    #        return self[:]

    @property
    def CW(self):
        """Rotate clockwise by 90 deg."""
        return self.T[:, ::-1]

    @property
    def CCW(self):
        """Rotate counter-clockwise by 90 deg."""
        return self.T[::-1, :]

    @property
    def _funcs(self):
        """Return an index of possible callable functions in other modules, caching result if not alreadty there.

        Look in Stoner.Image.imagefuncs, scipy.ndimage.* an d scikit.* for functions. We assume that each function
        takes a first argument that is an ndarray of image data, so with __getattrr__ and _func_generator we
        can make a bound method through duck typing."""
        if self._func_proxy is None:  # Buyild the cache
            func_proxy = (
                regexpDict()
            )  # Cache is a regular expression dictionary - keys matched directly and then by regular expression

            # Get the Stoner.Image.imagefuncs mopdule first
            from Stoner.Image import imagefuncs

            for d in dir(imagefuncs):
                if not d.startswith("_"):
                    func = getattr(imagefuncs, d)
                    if callable(func) and func.__module__ == imagefuncs.__name__:
                        name = "{}__{}".format(func.__module__, d).replace(".", "__")
                        func_proxy[name] = func

            # Get the Stoner.Image.util mopdule next
            from Stoner.Image import util as SIutil

            for d in dir(SIutil):
                if not d.startswith("_"):
                    func = getattr(SIutil, d)
                    if callable(func) and func.__module__ == SIutil.__name__:
                        name = "{}__{}".format(func.__module__, d).replace(".", "__")
                        func_proxy[name] = func

            # Add scipy.ndimage functions
            import scipy.ndimage as ndi

            _sp_mods = [ndi.interpolation, ndi.filters, ndi.measurements, ndi.morphology, ndi.fourier]
            for mod in _sp_mods:
                for d in dir(mod):
                    if not d.startswith("_"):
                        func = getattr(mod, d)
                        if callable(func) and func.__module__ == mod.__name__:
                            func.transpose = True
                            name = "{}__{}".format(func.__module__, d).replace(".", "__")
                            func_proxy[name] = func
            # Add the scikit images modules
            _ski_modules = [
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
            for mod in _ski_modules:
                for d in dir(mod):
                    if not d.startswith("_"):
                        func = getattr(mod, d)
                        if callable(func):
                            name = "{}__{}".format(func.__module__, d).replace(".", "__")
                            func_proxy[name] = func
            self._func_proxy = func_proxy  # Store the cache
        return self._func_proxy

    # ==============================================================
    # function generator
    # ==============================================================
    def __dir__(self):
        """Implement code for dir()"""
        proxy = set(list(self._funcs.keys()))
        parent = set(dir(super(ImageArray, self)))
        mine = set(dir(ImageArray))
        return sorted(list(proxy | parent | mine))

    def __getattr__(self, name):
        """Magic attribute access method.

        Tries first to get the attribute via a superclass call, if this fails
        checks for some well known attribute names and supplies missing defaults.

        To handle magic calls into other modules, we have a regular expression dicitonary
        that stores and index of callables by full name where the . are changed to __
        If we can get a match to __<name> then we get that callable from our index.

        The callable is then passed to self._func_generator for wrapping into a
        'on the fly' method of this class.

        TODO:
            An alternative nested attribute system could be something like
            http://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
            might be cool sometime.
        """
        ret = None
        try:
            ret = getattr(super(ImageArray, self), name)
        except AttributeError:
            # first check kermit funcs
            if name.startswith("_") or name in ["debug"]:
                if name == "_hardmask":
                    ret = False
            elif ".*__{}$".format(name) in self._funcs:
                ret = self._funcs[".*__{}$".format(name)]
                ret = self._func_generator(ret)
            if ret is None:
                raise AttributeError("No attribute found of name {}".format(name))
        return ret

    def _func_generator(self, workingfunc):
        """Used by __getattr__ to wrap an arbitary callbable to make it a bound method of this class.

        Args:
            workingfunc (callable): The callable object to be wrapped.

        Returns:
            (function): A function with enclosure that holds additional information about this object.

        The function returned from here will call workingfunc with the first argument being a clone of this
        ImageArray. If the meothd returns an ndarray, it is wrapped back to our own class and the metadata dictionary
        is updated. If the function returns a :py:class:`Stoner.Data` object then this is also updated with our metadata.

        This method also updates the name and documentation strings for the wrapper to match the wrapped function -
        thus ensuring that Spyder's help window can generate useful information.

        """

        @wraps(workingfunc)
        def gen_func(*args, **kwargs):
            """Wrapped magic proxy function call."""
            transpose = getattr(workingfunc, "transpose", False)
            if transpose:
                change = self.clone.T
            else:
                change = self.clone
            r = workingfunc(change, *args, **kwargs)  # send copy of self as the first arg
            if isinstance(r, Data):
                pass  # Data return is ok
            elif isinstance(r, np.ndarray) and np.prod(r.shape) == np.max(r.shape):  # 1D Array
                r = Data(r)
                r.metadata = self.metadata.copy()
                r.column_headers[0] = workingfunc.__name__
            elif isinstance(r, np.ndarray):  # make sure we return a ImageArray
                if transpose:
                    r = r.view(type=self.__class__).T
                else:
                    r = r.view(type=self.__class__)
                sm = self.metadata.copy()  # Copy the currenty metadata
                sm.update(r.metadata)  # merge in any new metadata from the call
                r.metadata = sm  # and put the returned metadata as the merged data
            # NB we might not be returning an ndarray at all here !
            return r

        return fix_signature(gen_func, workingfunc)

    @property
    def draw(self):
        """DrawProxy is an opbject for accessing the skimage draw sub module."""
        return DrawProxy(self)

    # ==============================================================================
    # OTHER SPECIAL METHODS
    # ==============================================================================

    def __setattr__(self, name, value):
        """Set an attribute on the object."""
        super(ImageArray, self).__setattr__(name, value)
        # add attribute to those for copying in array_finalize. use value as
        # defualt.
        circ = ["_optinfo"]  # circular references
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
        else:
            return super(ImageArray, self).__getitem__(index)

    def __setitem__(self, index, value):
        """Patch string index through to metadata."""
        if isinstance(index, ImageFile) and index.dtype == bool:
            index = index.image
        if isinstance(index, string_types):
            self.metadata[index] = value
        else:
            super(ImageArray, self).__setitem__(index, value)

    def __delitem__(self, index):
        """Patch indexing of strings to metadata."""
        if isinstance(index, string_types):
            del self.metadata[index]
        else:
            super(ImageArray, self).__delitem__(index)

    ############################################################################################################
    ############### Custom Methods for ImageArray###############################################################

    def crop(self, *args, **kargs):
        """Crop the image.

        This is essentially like taking a view onto the array
        but uses image x,y coords (x,y --> col,row)
        Returns a view according to the coords given. If box is None it will
        allow the user to select a rectangle. If a tuple is given with None
        included then max extent is used for that coord (analagous to slice).
        If copy then return a copy of self with the cropped image.

        Args:
            box(tuple) or 4 separate args or None:
                (xmin,xmax,ymin,ymax)
                If None image will be shown and user will be asked to select
                a box (bit experimental)

        Keyword Arguments:
            copy(bool):
                If True return a copy of ImageFile with the cropped image
        Returns:
            (ImageArray):
                view or copy of array asked for

        Example:
            a=ImageFile(np.arange(12).reshape(3,4))

            a.crop(1,3,None,None)
        """
        box = self._box(*args, **kargs)
        ret = self[box]
        if "copy" in kargs.keys() and kargs["copy"]:
            ret = ret.clone
        return ret

    def dtype_limits(self, clip_negative=True):
        """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

        Args:
            image(ndarray):
                Input image.
            clip_negative(bool):
                If True, clip the negative range (i.e. return 0 for min intensity)
                even if the image dtype allows negative values.

        Returns:
            (imin, imax : tuple): Lower and upper intensity limits.
        """
        if clip_negative is None:
            clip_negative = True
        imin, imax = dtype_range[self.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax

    def asfloat(self, normalise=True, clip=False, clip_negative=False):
        """Return the image converted to floating point type.

        If currently an int type and normalise then floats will be normalised
        to the maximum allowed value of the int type.
        If currently a float type then no change occurs.
        If clip then clip values outside the range -1,1
        If clip_negative then further clip values to range 0,1

        Keyword Arguments:
            normalise(bool):
                normalise the image to the max value of current int type
            clip_negative(bool):
                clip negative intensity to 0
        """
        if self.dtype.kind == "f":
            ret = self
        else:
            ret = convert(self, dtype=np.float64, normalise=normalise)  # preserve metadata
            ret = ImageArray(ret)
            c = self.clone  # copy formatting and apply to new array
            for k, v in c._optinfo.items():
                setattr(ret, k, v)
        if clip or clip_negative:
            ret = ret.clip_intensity(clip_negative=clip_negative)
        return ret

    def clip_intensity(self, clip_negative=False, limits=None):
        """Clip intensity outside the range -1,1 or 0,1

        Keyword ArgumentsL
            clip_negative(bool):
                if True clip to range 0,1 else range -1,1
            limits (low,high): Clip the intensity between low and high rather than zero and 1.

        Ensure data range is -1 to 1 or 0 to 1 if clip_negative is True.

        """
        if limits is None:
            dl = self.dtype_limits(clip_negative=clip_negative)
        else:
            dl = list(limits)
        np.clip(self, dl[0], dl[1], out=self)

    def asint(self, dtype=np.uint16):
        """convert the image to unsigned integer format.

        May raise warnings about loss of precision.
        """
        if self.dtype.kind == "f" and (np.max(self) > 1 or np.min(self) < -1):
            self = self.normalise()
        ret = convert(self, dtype)
        return ret

    def save(self, filename=None, **kargs):
        """Saves the image into the file 'filename'.

        Metadata will be preserved in .png and .tif format.

        fmt can be 'png', 'npy', 'tif', 'tiff'  or a list of more than one of those.
        tif is recommended since metadata is lost in .npy format but data is
        converted to integer format for png so that definition cannot be
        saved.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If
                filename is False then a file dialog is forced.
        Keyword args:
            fmt (string or list): format to save data as. 'tif', 'png' or 'npy'
                or a list of them. If not included will guess from filename.

            forcetype (bool): integer data will be converted to np.float32 type
                for saving. if forcetype then preserve and save as int type (will
                                                                             be unsigned).

        Since Stoner.Image is meant to be a general 2d array often with negative
        and floating point data this poses a problem for saving images. Images
        are naturally saved as 8 or more bit unsigned integer values representing colour.
        The only obvious way to save an image and preserve negative data
        is to save as a float32 tif. This has the advantage over the npy
        data type which cannot be opened by external programs and will not
        save metadata.
        """
        # Standard filename block
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):
            # now go and ask for one
            filename = self.__file_dialog("w")

        def_fmt = os.path.splitext(filename)[1][1:]  # Get a default format from the filename
        if def_fmt not in self.fmts:  # Default to png if nothing else
            def_fmt = "png"
        fmt = kargs.pop("fmt", [def_fmt])

        if not isinstance(fmt, list):
            fmt = [fmt]
        if set(fmt) & set(self.fmts) == set([]):
            raise ValueError("fmt must be {}".format(",".join(self.fmts)))
        fmt = ["tiff" if f == "tif" else f for f in fmt]
        self.filename = filename
        for fm in fmt:
            saver = getattr(self, "save_{}".format(fm), "save_tif")
            if fm == "tiff":
                forcetype = kargs.pop("forcetype", False)
                saver(filename, forcetype)
            else:
                saver(filename)

    def save_png(self, filename):
        """Save the ImageArray with metadata in a png file.
        This can only save as 8bit unsigned integer so there is likely
        to be a loss of precision on floating point data"""
        pngname = os.path.splitext(filename)[0] + ".png"
        meta = PngImagePlugin.PngInfo()
        info = self.metadata.export_all()
        info = [(i.split("=")[0], "=".join(i.split("=")[1:])) for i in info]
        for k, v in info:
            meta.add_text(k, v)
        s = (self - self.min()) * 256 / (self.max() - self.min())
        im = Image.fromarray(s.astype("uint8"), mode="L")
        im.save(pngname, pnginfo=meta)

    def save_npy(self, filename):
        """Save the ImageArray as a numpy array."""
        npyname = os.path.splitext(filename)[0] + ".npy"
        np.save(npyname, np.array(self))

    def save_tiff(self, filename, forcetype=False):
        """Save the ImageArray as a tiff image with metadata
        PIL can save in modes "L" (8bit unsigned int), "I" (32bit signed int),
        or "F" (32bit signed float). In general max info is preserved for "F"
        type so if forcetype is not specified then this is the default. For
        boolean type data mode "L" will suffice and this is chosen in all cases.
        The type name is added as a string to the metadata before saving.

        Keyword Args:
            forcetype(bool): if forcetype then preserve data type as best as
                possible on save.
                Otherwise integer data will be converted to np.float32 type
                for saving. (bool will remain as int since there's no danger of
                loss of information)
        """
        from PIL.TiffImagePlugin import ImageFileDirectory_v2
        import json

        dtype = np.dtype(self.dtype).name  # string representation of dtype we can save
        self["ImageArray.dtype"] = dtype  # add the dtype to the metadata for saving.
        if forcetype:  # PIL supports uint8, int32 and float32, try to find the best match
            if self.dtype == np.uint8 or self.dtype.kind == "b":  # uint8 or boolean
                im = Image.fromarray(self, mode="L")
            elif self.dtype.kind in ["i", "u"]:
                im = Image.fromarray(self, mode="I")
            else:  # default to float32
                im = Image.fromarray(self.astype(np.float32), mode="F")
        else:
            if self.dtype.kind == "b":  # boolean we're not going to lose data by saving as unsigned int
                im = Image.fromarray(self, mode="L")
            else:  # try to convert everything else to float32 which can has maximum preservation of info
                im = Image.fromarray(self.astype(np.float32), mode="F")
        ifd = ImageFileDirectory_v2()
        ifd[270] = json.dumps(self.metadata.export_all())
        ext = os.path.splitext(filename)[1]
        if ext in [".tif", ".tiff"]:  # ensure extension is preserved in save
            pass
        else:  # default to tiff
            ext = ".tiff"
        tiffname = os.path.splitext(filename)[0] + ext
        im.save(tiffname, tiffinfo=ifd)

    def __file_dialog(self, mode):
        """Creates a file dialog box for loading or saving ~b ImageFile objects.

        Args:
            mode(string): The mode of the file operation  'r' or 'w'

        Returns:
            A filename to be used for the file operation.
        """
        patterns = ("png", "*.png")

        if self.filename is not None:
            filename = os.path.basename(self.filename)
            dirname = os.path.dirname(self.filename)
        else:
            filename = ""
            dirname = ""
        if "r" in mode:
            mode = "file"
        elif "w" in mode:
            mode = "save"
        else:
            mode = "directory"
        dlg = get_filedialog(what=mode, initialdir=dirname, initialfile=filename, filetypes=patterns)
        if len(dlg) != 0:
            self.filename = dlg
            return self.filename
        else:
            return None

    ############################################################################################################
    ############## Depricated Methods ##########################################################################

    def box(self, *args, **kargs):
        """Alias for :py:meth:`ImageArray.crop`"""
        warnings.warn("The box method was replaced by crop and will raise and error in future versions.")
        return self.crop(*args, **kargs)

    def crop_image(self, *args, **kargs):
        """Back compatability alias for :py:meth:`ImageArray.crop`"""
        warnings.warn("The crop_image method was replaced by crop and will raise and error in future versions.")
        return self.crop(*args, **kargs)

    def convert_float(self, clip_neg=True):
        """Deproicated compatability. :py:meth:`ImageArray.asfloat` preferred"""
        warnings.warn("The convert_float method was replaced by asfloat and will raise and error in future versions.")
        self.asfloat(normalise=False, clip_negative=clip_neg)

    def convert_int(self):
        """Depricated compatability meothd. :py:meth:`ImageArray.asint` preferred"""
        warnings.warn("The convert_int method was replaced by asint and will raise and error in future versions.")
        self.asint()


class ImageFile(metadataObject):

    """An Image file type that is analagous to DataFile.

    This contains metadata and an image attribute which
    is an ImageArray type which subclasses numpy ndarray and adds lots of extra
    image specific processing functions.

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

    def __init__(self, *args, **kargs):
        """Mostly a pass through to ImageArray constructor.

        Local attribute is image. All other attributes and calls are passed
        through to image attribute.

        There is one special case of creating an ImageFile from a :py:class:`Stoner.Core.DataFile`. In this case the
        the DataFile is assummed to contain (x,y,z) data that should be converted to a map of
        z on a regular grid of x,y. The columns for the x,y,z data can be taken from the DataFile's
        :py:attr:`Stoner.Core.DataFile.setas` attribute or overridden by providing xcol, ycol and zcol keyword arguments.
        A further *shape* keyword can spewcify the shape as a tuple or "unique" to use the unique values of x and y or if
        omitted asquare grid will be interpolated.

        """
        super(ImageFile, self).__init__(*args, **kargs)
        if len(args) == 0:
            self._image = ImageArray()
        elif len(args) > 0 and isinstance(args[0], string_types):
            self._image = ImageArray(*args, **kargs)
        elif len(args) > 0 and isinstance(args[0], ImageFile):  # Fixing type
            self._image = args[0].image
        elif len(args) > 0 and isinstance(args[0], np.ndarray):  # Fixing type
            self._image = ImageArray(*args, **kargs)
        elif len(args) > 0 and isinstance(
            args[0], DataFile
        ):  # Support initing from a DataFile that defines x,y,z coordinates
            self._init_from_datafile(*args, **kargs)
        self._fromstack = kargs.pop("_fromstack", False)  # for use by ImageStack

    #####################################################################################################################################
    ############################# Properties #### #######################################################################################

    @property
    def _repr_png_(self):
        if get_option("short_repr") or get_option("short_img_repr"):
            raise AttrbuteError("Suppressed graphical representation")
        else:
            return self._repr_png_private_

    @property
    def clone(self):
        """Make a copy of this ImageFile."""
        new = self.__class__(self.image.clone)
        for attr in self.__dict__:
            if callable(getattr(self, attr)) or attr in ["image", "metadata"]:
                continue
            try:
                setattr(new, attr, deepcopy(getattr(self, attr)))
            except NotImplementedError:  # Deepcopying failed, so just copy a reference instead
                setattr(new, attr, getattr(self, attr))
        return new

    @property
    def data(self):
        """alias for image[:]. Equivalence to Stoner.data behaviour"""
        return self.image

    @data.setter
    def data(self, value):
        """Simple minded pass through."""
        self.image = value

    @property
    def CW(self):
        """Rotate clockwise by 90 deg."""
        ret = self.clone
        ret.image = ret.image.CW
        return ret

    @property
    def CCW(self):
        """Rotate counter-clockwise by 90 deg."""
        ret = self.clone
        ret.image = ret.image.CCW
        return ret

    @property
    def image(self):
        """Access the image data."""
        return self._image

    @image.setter
    def image(self, v):
        """Ensure stored image is always an ImageArray."""
        filename = self.filename
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

    @property
    def filename(self):
        """Pass through to image attribute."""
        if self._image is not None:
            return self.image.filename
        else:
            return ""

    @filename.setter
    def filename(self, value):
        """Pass through to image attribute."""
        if self._image is None:
            self._image = ImageArray()
        self.image.filename = value

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

    @property
    def metadata(self):
        """Intercept metadata and redirect to image.metadata."""
        return self.image.metadata

    @metadata.setter
    def metadata(self, value):
        self.image.metadata = value

    #####################################################################################################################################
    ############################# Special methods #######################################################################################

    def __dir__(self):
        """Implement code for dir()"""
        proxy = set(dir(self.image))
        parent = set(dir(super(ImageFile, self)))
        mine = set(dir(ImageFile))
        return sorted(list(proxy | parent | mine))

    def __getitem__(self, n):
        """A pass through to ImageArray."""
        try:
            return self.image.__getitem__(n)
        except KeyError:
            if n not in self.metadata and n in self._image.metadata:
                self.metadata[n] = self._image.metadata[n]
            return self.metadata.__getitem__(n)

    def __setitem__(self, n, v):
        """A Pass through to ImageArray."""
        if isinstance(n, string_types):
            self.metadata.__setitem__(n, v)
        else:
            self.image.__setitem__(n, v)

    def __delitem__(self, n):
        """A Pass through to ImageArray."""
        try:
            self.image.__delitem__(n)
        except KeyError:
            self.metadata.__delitem__(n)

    def __getattr__(self, n):
        """"Handles attriobute access."""
        try:
            ret = super(ImageFile, self).__getattr__(n)
        except AttributeError:
            ret = getattr(self.image, n)
            if callable(ret):  # we have a method
                ret = self._func_generator(ret)  # modiy so that we can change image in place
        return ret

    def __setattr__(self, n, v):
        """Handles setting attributes."""
        if not hasattr(self, n) and n not in ImageFile._protected_attrs:
            setattr(self._image, n, v)
        else:
            super(ImageFile, self).__setattr__(n, v)

    def __add__(self, other):
        """Implement the subtract operator"""
        result = self.clone
        result = __add_core__(result, other)
        return result

    def __iadd__(self, other):
        """Implement the inplace subtract operator"""
        result = self
        result = __add_core__(result, other)
        return result

    if python_v3:

        def __truediv__(self, other):
            """Implement the divide operator"""
            result = self.clone
            result = __div_core__(result, other)
            return result

        def __itruediv__(self, other):
            """Implement the inplace divide operator"""
            result = self
            result = __div_core__(result, other)
            return result

    else:

        def __div__(self, other):
            """Implement the divide operator"""
            result = self.clone
            result = __div_core__(result, other)
            return result

        def __idiv__(self, other):
            """Implement the inplace divide operator"""
            result = self
            result = __div_core__(result, other)
            return result

    def __sub__(self, other):
        """Implement the subtract operator"""
        result = self.clone
        result = __sub_core__(result, other)
        return result

    def __isub__(self, other):
        """Implement the inplace subtract operator"""
        result = self
        result = __sub_core__(result, other)
        return result

    def __neg__(self):
        """Intelliegent negate function that handles unsigned integers."""
        ret = self.clone
        if self._image.dtype.kind == "u":
            h = dtype_range[self._image.dtype][1]
            ret.image = h - self.image
        else:
            ret.image = -self.image
        return ret

    def __invert__(self):
        """Equivalent to clockwise rotation"""
        return self.CW

    def __floordiv__(self, other):
        """Calculate and XMCD ratio on the images."""
        if not isinstance(other, ImageFile):
            return NotImplemented
        ret = self.clone
        ret.image = (self.image - other.image) / (self.image + other.image)
        return ret

    #####################################################################################################################################
    ############################# Private methods #######################################################################################

    def _init_from_datafile(self, *args, **kargs):
        """Initialise ImageFile from DataFile defining x,y,z co-ordinates.

        Args:
            args[0] (DataFile): A :py:class:`Stoner.Core.DataFile` instance that defines x,y,z co-ordinates or has columns specified in keywords.

        Keyword Args:
            xcol (column index): Column in the DataFile that has the x-co-ordinate
            ycol (column index): Column in the data file that defines the y-cordinate
            zcol (column index): Column in the datafile that defines the intensity
        """
        data = Data(args[0])
        shape = kargs.pop("shape", "unique")

        _ = data._col_args(**kargs)
        data.setas(x=_.xcol, y=_.ycol, z=_.zcol)  # pylint: disable=not-callable
        if isinstance(shape, string_types) and shape == "unique":
            shape = (len(np.unique(data.x)), len(np.unique(data.y)))
        elif istuple(shape, int_types, int_types):
            pass
        else:
            shape = None
        X, Y, Z = data.griddata(_.xcol, _.ycol, _.zcol, shape=shape)
        self.image = Z
        self.metadata = deepcopy(data.metadata)
        self["x_vector"] = np.unique(X)
        self["y_vector"] = np.unique(Y)

    def _func_generator(self, workingfunc):
        """ImageFile generator.

        Note:
            The wrapped functions take additional keyword arguments that are stripped off from the call.

            _box(:py:meth:`Stoner.ImageArray.crop` arguments): Crops the image first before calling the parent method.
            _ (bool, None): Controls whether a :py:class:`ImageArray` return will be substituted for the current :py:class:`ImageArray`.

            * True: - all ImageArray return types are substituted.
            * False (default) - Imagearray return types are substituted if they are the same size as the original
            * None - A copy of the current object is taken and the returned ImageArray provides the data.
        """

        @wraps(workingfunc)
        def gen_func(*args, **kargs):
            """This will wrap a called method."""

            box = kargs.pop("_box", None)
            if len(args) > 0:
                args = list(args)
                for ix, a in enumerate(args):
                    if isinstance(a, ImageFile):
                        args[ix] = a.image[self.image._box(box)]
            if (
                workingfunc.__name__ == "crop" and "_" not in kargs.keys()
            ):  # special case for common function crop which will change the array shape
                force = True
            else:
                force = kargs.pop("_", False)
            r = workingfunc(*args, **kargs)
            if (
                force is not None
                and isinstance(r, ImageArray)
                and (force or r.shape == self.image[self.image._box(box)].shape)
            ):
                # Enure that we've captured any metadata added inside the working function
                self.metadata.update(r.metadata)
                # Now swap the iamge in, but keep the metadata
                r.metadata = self.metadata
                filename = self.filename
                self.image = r
                self.filename = filename

                return self
            elif force is None:
                ret = self.clone
                ret.metadata.update(r.metadata)
                # Now swap the iamge in, but keep the metadata
                r.metadata = ret.metadata
                filename = ret.filename
                ret.image = r
                ret.filename = filename
                return ret
            else:
                return r

        return fix_signature(gen_func, workingfunc)

    def __repr__(self):
        return "{}({}) of shape {} ({}) and {} items of metadata".format(
            self.filename, type(self), self.shape, self.image.dtype, len(self.metadata)
        )

    def _repr_png_private_(self):
        """Provide a display function for iPython/Jupyter."""
        fig = self.image.imshow()
        plt.title(self.filename)
        data = StreamIO()
        fig.savefig(data, format="png")
        plt.close(fig)
        data.seek(0)
        ret = data.read()
        data.close()
        return ret

    def save(self, filename=None, **kargs):
        """Saves the image into the file 'filename'.

        Metadata will be preserved in .png and .tif format.

        fmt can be 'png', 'npy', 'tif', 'tiff'  or a list of more than one of those.
        tif is recommended since metadata is lost in .npy format but data is
        converted to integer format for png so that definition cannot be
        saved.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If
                filename is False then a file dialog is forced.
        Keyword args:
            fmt (string or list): format to save data as. 'tif', 'png' or 'npy'
            or a list of them. If not included will guess from filename.
        """
        # catch before metadataObject tries to take over.
        self.image.save(filename, **kargs)


class DrawProxy(object):
    """Provides a wrapper around scikit-image.draw to allow easy drawing of objects onto images."""

    def __init__(self, *args, **kargs):
        """Grab the parent image from the constructor."""
        self.img = args[0]

    def __getattr__(self, name):
        """Retiurn a callable function that will carry out the draw operation requested."""
        func = getattr(draw, name)

        @wraps(func)
        def _proxy(*args, **kargs):
            value = kargs.pop("value", np.ones(1, dtype=self.img.dtype)[0])
            coords = func(*args, **kargs)
            self.img[coords] = value
            return self.img

        return fix_signature(_proxy, func)

    def __dir__(self):
        """Pass through to the dir of skimage.draw."""
        return draw.__dir__()

    def annulus(self, r, c, radius1, radius2, shape=None, value=1.0):
        """Use a combination of two circles to draw and annulus.

        Args:
            r,c (float): Centre co-ordinates
            radius1,radius2 (float): Inner and outer radius.

        Keyword Arguments:
            shape (2-tuple, None): Confine the co-ordinates to staywith shape
            value (float): value to draw with
        Returns:
            A copy of the image with the annulus drawn on it.

        Notes:
            If radius2<radius1 then the sense of the whole shape is inverted
            so that the annulus is left clear and the filed is filled.
        """
        if shape is None:
            shape = self.img.shape
        invert = radius2 < radius1
        if invert:
            buf = np.ones(shape)
            fill = 0.0
            bg = 1.0
        else:
            buf = np.zeros(shape)
            fill = 1.0
            bg = 2.0
        radius1, radius2 = min(radius1, radius2), max(radius1, radius2)
        rr, cc = draw.circle(r, c, radius2, shape=shape)
        buf[rr, cc] = fill
        rr, cc = draw.circle(r, c, radius1, shape=shape)
        buf[rr, cc] = bg
        self.img = np.where(buf == 1, value, self.img)
        return self.img

    def rectangle(self, r, c, w, h, angle=0.0, shape=None, value=1.0):
        """Draw a rectangle on an image.

        Args:
            r,c (float): Centre co-ordinates
            w,h (float): Lengths of the two sides of the rectangle

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        if shape is None:
            shape = self.img.shape

        x1 = r - h / 2
        x2 = r + h / 2
        y1 = c - w / 2
        y2 = c + w / 2
        co_ords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        if angle != 0:
            centre = np.array([r, c])
            c, s, m = np.cos, np.sin, np.matmul
            r = np.array([[c(angle), -s(angle)], [s(angle), c(angle)]])
            co_ords = np.array([centre + m(r, xy - centre) for xy in co_ords])
        rr, cc = draw.polygon(co_ords[:, 0], co_ords[:, 1], shape=shape)
        self.img[rr, cc] = value
        return self.img

    def square(self, r, c, w, angle=0.0, shape=None, value=1.0):
        """Draw a square on an image.

        Args:
            r,c (float): Centre co-ordinates
            w (float): Length of the side of the square

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        return self.rectangle(r, c, w, w, angle=angle, shape=shape, value=value)


class MaskProxy(object):

    """Provides a wrapper to support manipulating the image mask easily."""

    @property
    def _IA(self):
        """Get the underliying image data."""
        return self._IF.image

    @property
    def _mask(self):
        """Get the mask for the underlying image."""
        self._IA.mask = np.ma.getmaskarray(self._IA)
        return self._IA.mask

    @property
    def data(self):
        """Get the underlying data as an array - compatibility accessor"""
        return self[:]

    @property
    def image(self):
        """Get the underlying data as an array - compatibility accessor"""
        return self[:]

    @property
    def draw(self):
        """Access the draw proxy opbject."""
        return DrawProxy(self._mask)

    def __init__(self, *args, **kargs):
        """Keep track of the underlying objects."""
        self._IF = args[0]

    def __getitem__(self, index):
        """Proxy through to mask index."""
        return self._mask.__getitem__(index)

    def __setitem__(self, index, value):
        """Proxy through to underlying mask."""
        self._IA.mask.__setitem__(index, value)

    def __delitem__(self, index):
        """Proxy through to underyling mask."""
        self._IA.mask.__delitem__(index)

    def __getattr__(self, name):
        """Checks name against self._IA._funcs and constructs a method to edit the mask as an image."""
        if hasattr(self._IA.mask, name):
            return getattr(self._IA.mask, name)
        if not ".*__{}$".format(name) in self._IA._funcs:
            raise AttributeError("{} not a callable mask method.".format(name))
        func = self._IA._funcs[".*__{}$".format(name)]

        @wraps(func)
        def _proxy_call(*args, **kargs):
            r = func(self._mask.astype(int), *args, **kargs)
            if isinstance(r, np.ndarray) and r.shape == self._IA.shape:
                self._IA.mask = r
            return r

        _proxy_call.__doc__ = func.__doc__
        _proxy_call.__name__ = func.__name__
        return fix_signature(_proxy_call, func)

    def __repr__(self):
        """Make a textual representation of the image."""
        return repr(self._mask)

    def __str__(self):
        """Make a textual representation of the image."""
        return repr(self._mask)

    def __neg__(self):
        """Invert the mask."""
        return -self._mask

    def _repr_png_(self):
        """Provide a display function for iPython/Jupyter."""
        fig = self._IA._funcs[".*imshow"](self._mask.astype(int))
        data = StreamIO()
        fig.savefig(data, format="png")
        plt.close(fig)
        data.seek(0)
        ret = data.read()
        data.close()
        return ret

    def clear(self):
        """Clear a mask."""
        self._IA.mask = np.zeros_like(self._IA)

    def invert(self):
        """Invert the mask."""
        self._IA.mask = ~self._IA.mask
