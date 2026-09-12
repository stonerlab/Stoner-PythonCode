# -*- coding: utf-8 -*-
"""Typing definitions."""

from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Any, ForwardRef, Mapping, Sequence, Tuple, Union

from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.data import Data
    from ..core.setas import Setas
    from ..Image.core import ImageArray, ImageFile
else:
    # Resolve after package import, without importing partially initialised classes.
    Setas = ForwardRef("core.setas.Setas", module="Stoner")
    Data = ForwardRef("core.data.Data", module="Stoner")
    ImageArray = ForwardRef("Image.core.ImageArray", module="Stoner")
    ImageFile = ForwardRef("Image.core.ImageFile", module="Stoner")

Args = Tuple[Any]
Kwargs = Mapping[str, Any]
SingleIndex = Union[str, int, Pattern]
Index = Union[SingleIndex, Sequence[Union[int, str, Pattern]]]
Filename = Union[str, Path, bool]

RegExp = Pattern
PathTypes = Filename


# Setas
Setas_Base = Sequence[str]
Setas_Dict = Mapping[str, Index]
Setas = Union[Setas_Base, Setas_Dict, str, Setas]

# Other useful types
Numeric = Union[float, int, complex]
NumericArray = Union[Sequence[Numeric], NDArray]
