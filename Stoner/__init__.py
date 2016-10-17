"""Stoner Python Package: Utility classes for simple data analysis scripts.

See http://github.com/~gb119/Stoner-PythonCode for more details."""

__all__=['Core', 'Analysis', 'Plot', 'PlotFormats','Folders','DataFile','Data','DataFolder']

# These fake the old namespace if you do an import Stoner

from .Core import DataFile
from .Util import Data
from Stoner.Folders import DataFolder

__version_info__ = ('0', '6', '4')
__version__ = '.'.join(__version_info__)

