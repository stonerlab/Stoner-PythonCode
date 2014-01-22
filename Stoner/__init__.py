"""Stoner Python Package: Utility classes for simple data analysis scripts.

See http://github.com/~gb119/Stoner-PythonCode for more details."""

__all__=['Core', 'Analysis', 'Plot', 'FileFormats', 'Folders']

# These fake the old namespace if you do an import Stoner

# Support both Python 2.7 and 3
from Stoner.compat import *


from .Core import *
from .Analysis import *
from .Plot import *
from .FileFormats import *
from .Folders import *

__version_info__ = ('0', '1', '8')
__version__ = '.'.join(__version_info__)

