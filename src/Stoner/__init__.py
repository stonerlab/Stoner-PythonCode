#----------------------------------------------------------------------------- 
#   $Id: __init__.py,v 1.5 2011/01/08 20:30:02 cvs Exp $
#   AUTHOR:     MATTHEW NEWMAN, CHRIS ALLEN, GAVIN BURNELL
#   DATE:       24/11/2010
#-----------------------------------------------------------------------------
#

__all__=['Core', 'Analysis', 'Plot']

# These fake the old namespace if you do an import Stoner
from .Core import *
from .Analysis import *
from Plot import *


