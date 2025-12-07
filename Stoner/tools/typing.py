# -*- coding: utf-8 -*-
"""Typing definitions."""
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple, Union

from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.data import Data
    from ..core.setas import setas
    from ..image.core import ImageArray, ImageFile
else:
    setas = type("setas", (), {})
    Data = type("Data", (), {})
    ImageArray = type("ImageArray", (), {})
    ImageFile = type("ImageFile", (), {})

Args = Tuple[Any]
Kwargs = Mapping[str, Any]
Single_Index = Union[str, int, Pattern]
Index = Union[Single_Index, Sequence[Union[int, str, Pattern]]]
Filename = Union[str, Path, bool]

RegExp = Pattern
Path_Types = Filename


# Setas
Setas_Base = Sequence[str]
Setas_Dict = Mapping[str, Index]
Setas = Union[Setas_Base, Setas_Dict, str, setas]

# Other useful types
Numeric = Union[float, int, complex]
NumericArray = Union[Sequence[Numeric], NDArray]
