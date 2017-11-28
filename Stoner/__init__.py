"""Stoner Python Package: Utility classes for simple data analysis scripts.

See http://github.com/~gb119/Stoner-PythonCode for more details.

"""

__all__=['Core', 'Analysis', 'plot', 'tools','FileFormats','Folders','DataFile','Data','DataFolder']

# These fake the old namespace if you do an import Stoner

from .Core import DataFile,Data
from .Folders import DataFolder

from os import path as _path_
__version_info__ = ('0', '7', '2')
__version__ = '.'.join(__version_info__)

__home__=_path_.realpath(_path_.dirname(__file__))
