"""Stoner.plot sub-package - contains classes and functions for visuallising data."""
from .core import PlotMixin,hsl2rgb
from .core import PlotFile as _PF_
__all__ = ["PlotMixin","hsl2rgb","formats","utils"]
def PlotFile(*args,**kargs):
    """Wrapper to raise DepricationWarning."""
    from warnings import warn
    warn("PlotFile to be withdrawn in 0.8, either use Stoner.Data or Stoner.plot.PlotMixin.",DeprecationWarning)
    return _PF_(*args,**kargs)