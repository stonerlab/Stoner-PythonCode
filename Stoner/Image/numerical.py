"""Prepare NumPy working images and dispatch ordinary numerical functions."""

from collections.abc import Iterable
from functools import cache
from pathlib import Path
from types import MethodType
import inspect
import json

import numpy as np
from PIL import Image

from ..core.base import TypeHintedDict
from ..core.storage import _copy_metadata
from .widgets import RegionSelect


@cache
def _functions():
    """Collect numerical operations once, without defining an array subclass."""
    from scipy import ndimage
    from skimage import color, exposure, feature, filters, measure, morphology, restoration, segmentation, transform, util
    from . import imagefuncs, kerrfuncs
    result = {}
    for module in (color, exposure, feature, filters, measure, morphology, restoration,
                   segmentation, transform, util, ndimage, imagefuncs, kerrfuncs):
        for name in dir(module):
            function = getattr(module, name)
            if (not name.startswith('_') and inspect.isfunction(function)
                    and function.__module__.startswith(module.__name__)
                    and not hasattr(np.ma.MaskedArray, name)):
                result[name] = function
    return result


def load_pixels(filename, **kwargs):
    """Read ordinary image pixels and stored metadata without a custom array class."""
    filename = str(filename)
    metadata = TypeHintedDict()
    if Path(filename).suffix.lower() == '.npy':
        pixels = np.load(filename, allow_pickle=False)
    else:
        with Image.open(filename) as source:
            pixels = np.array(source)
            is_tiff = hasattr(source, 'tag_v2')
            if not is_tiff:
                metadata.update(source.info)
            if is_tiff and pixels.ndim == 3:
                from skimage import io
                pixels = np.ascontiguousarray(io.imread(filename)).view(np.uint32)[:, :, 0]
            if is_tiff and 270 in source.tag_v2:
                try:
                    description = json.loads(source.tag_v2[270])
                    metadata.import_all(description.get('metadata', []))
                except (ValueError, TypeError):
                    pass
            if is_tiff and 'ImageArray.dtype' in metadata:
                pixels = pixels.astype(metadata['ImageArray.dtype'])
    metadata['Loaded from'] = str(Path(filename).resolve())
    result = numerical_image(pixels, metadata=metadata)
    result.filename = filename
    return result


def numerical_image(values=None, *, metadata=None, **kwargs):
    """Return a detached ordinary masked array for numerical image calculations.

    Metadata and bound numerical functions belong only to this working result.
    NumPy operations keep native array semantics and never own public image state.
    """
    if isinstance(values, (str, Path)):
        return load_pixels(values, **kwargs)
    if hasattr(values, 'export_storage'):
        metadata = values.metadata.copy() if metadata is None else metadata
        values = values.to_numpy()
    original = values
    options = {key: kwargs[key] for key in ('dtype', 'mask', 'order') if key in kwargs}
    result = np.ma.array(np.empty((0, 0)) if values is None else values, copy=True, **options)
    result.mask = np.ma.getmaskarray(result)
    source_metadata = metadata if metadata is not None else getattr(original, 'metadata', {})
    result.metadata = (_copy_metadata(source_metadata) if isinstance(source_metadata, TypeHintedDict)
                       else TypeHintedDict(source_metadata))
    result.filename = getattr(original, 'filename', '')
    result.debug = False
    result._title = getattr(original, '_title', None)
    result._mask_color = getattr(original, '_mask_color', 'red')
    result._mask_alpha = getattr(original, '_mask_alpha', 0.5)
    result._public_attrs = {'filename': str}
    result.fmts = ['png', 'npy', 'tiff', 'tif']
    if hasattr(original, 'tesseractable'):
        result.tesseractable = original.tesseractable
    if result.ndim == 2:
        result.max_box = (0, result.shape[1], 0, result.shape[0])
        result.centre = (result.shape[0] / 2, result.shape[1] / 2)
        result.aspect = result.shape[1] / result.shape[0] if result.shape[0] else 0
    from .attrs import DrawProxy
    result.draw = DrawProxy(result, result)
    result._box = MethodType(box, result)
    for name, function in _functions().items():
        def invoke(*args, _function=function, **options):
            output = _function(result, *args, **options)
            if isinstance(output, np.ndarray) and output.ndim == 2:
                output = numerical_image(output, metadata=getattr(output, 'metadata', result.metadata))
                if hasattr(result, 'tesseractable'):
                    output.tesseractable = result.tesseractable
                return output
            return output
        setattr(result, name, invoke)
    return result


def box(self, *args, **kwargs):
    """Construct an indexing tuple for selecting an image rectangle.

    The box can be specified as:

        - (int): a fixed number of pixels is removed from all sides
        - (float): the central region of the image is selected
        - None: the user selects a region of interest
        - False: the whole image is selected
        - (iterable of length 4) - assumed to give 4 integers to describe a specific box
    """
    if len(args) == 0 and "box" in kwargs:
        args = (kwargs["box"],)  # back compatibility
    if len(args) == 0 or (len(args) == 1 and args[0] is None):
        args = tuple(RegionSelect()(self))
    match args:
        case (box,):
            match box:
                case bool() if not box:
                    return slice(None, None, None), slice(None, None, None)
                case Iterable() if len(box) == 4 and not isinstance(box, str):
                    box = list(box)
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
                    box = list((int(x) for x in box))
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
