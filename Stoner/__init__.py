"""The Stoner Python package provides utility classes for writing simple data analysis scripts more easily.  It has been developed by members
of the `Condensed Matter Group<http://www.stoner.leeds.ac.uk/>` at the `University of Leeds<http://www.leeds.ac.uk>`.
"""
# pylint: disable=import-error
__all__=['Core', 'Analysis', 'plot', 'Image','tools','FileFormats','Folders','Data','DataFolder','set_option','get_option']

# These fake the old namespace if you do an import Stoner
from sys import float_info
import inspect as _inspect_


import Stoner.Core as Core
import Stoner.FileFormats as FileFormats
import Stoner.plot.core as plot
import Stoner.Analysis as Analysis
import Stoner.tools as tools
import Stoner.Folders as Folders
from .Folders import DataFolder

from .compat import _lmfit,Model
from .tools import format_error,set_option,get_option

from os import path as _path_
__version_info__ = ('0', '8', '0')
__version__ = '.'.join(__version_info__)

__home__=_path_.realpath(_path_.dirname(__file__))

class Data(Analysis.AnalysisMixin,plot.PlotMixin,Core.DataFile):

    """A merged class of :py:class:`Stoner.Core.DataFile`, :py:class:`Stoner.Analysis.AnalysisMixin` and :py:class:`Stoner.plot.PlotMixin`

    Also has the :py:mod:`Stoner.FielFormats` loaded redy for use.
    This 'kitchen-sink' class is intended as a convenience for writing scripts that carry out both plotting and
    analysis on data files.
    """

    def format(self,key,**kargs):
        r"""Return the contents of key pretty formatted using :py:func:`format_error`.

        Args:
            fmt (str): Specify the output format, opyions are:

                *  "text" - plain text output
                * "latex" - latex output
                * "html" - html entities

            escape (bool): Specifies whether to escape the prefix and units for unprintable characters in non text formats )default False)
            mode (string): If "float" (default) the number is formatted as is, if "eng" the value and error is converted
                to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a scientifc,
                i.e. mantissa and exponent format is used.
            units (string): A suffix providing the units of the value. If si mode is used, then appropriate si prefixes are
                prepended to the units string. In LaTeX mode, the units string is embedded in \mathrm
            prefix (string): A prefix string that should be included before the value and error string. in LaTeX mode this is
                inside the math-mode markers, but not embedded in \mathrm.

        Returns:
            A pretty string representation.

        The if key="key", then the value is self["key"], the error is self["key err"], the default prefix is self["key label"]+"=" or "key=",
        the units are self["key units"] or "".

        """
        mode=kargs.pop("mode","float")
        units=kargs.pop("units",self.get(key+" units","")	)
        prefix=kargs.pop("prefix","{} = ".format(self.get(key+" label","{}".format(key))))
        latex=kargs.pop("latex",False)
        fmt=kargs.pop("fmt","latex" if latex else "text")
        escape=kargs.pop("escape",False)

        try:
            value=float(self[key])
        except (ValueError, TypeError):
            raise KeyError("{} should be a floating point value of the metadata not a {}.".format(key,type(self[key])))
        try:
            error=float(self[key+" err"])
        except KeyError:
            error=float_info.epsilon
        return format_error(value,error,fmt=fmt,mode=mode,units=units,prefix=prefix,scape=escape)

    def annotate_fit(self,model,x=None,y=None,z=None,text_only=False,**kargs):
        """Annotate a plot with some information about a fit.

        Args:
            mode (callable or lmfit.Model): The function/model used to describe the fit to be annotated.

        Keyword Parameters:
            x (float): x co-ordinate of the label
            y (float): y co-ordinate of the label
            z (float): z co-ordinbate of the label if the current axes are 3D
            prefix (str): The prefix placed ahead of the model parameters in the metadata.
            text_only (bool): If False (default), add the text to the plot and return the current object, otherwise,
                return just the text and don't add to a plot.
            prefix(str): If given  overridges the prefix from the model to determine a prefix to the parameter names in the metadata

        Returns:
            (Datam, str): A copy of the current Data instance if text_only is False, otherwise returns the text.

        If *prefix* is not given, then the first prefix in the metadata lmfit.prefix is used if present,
        otherwise a prefix is generated from the model.prefix attribute. If *x* and *y* are not specified then they
        are set to be 0.75 * maximum x and y limit of the plot.
        """
        mode=kargs.pop("mode","float")
        if _lmfit and _inspect_.isclass(model) and issubclass(model,Model):
            prefix=kargs.pop("prefix",self.get("lmfit.prefix",model.__name__))
            model=model()
        elif _lmfit and isinstance(model,Model):
            prefix=kargs.pop("prefix",self.get("lmfit.prefix",model.__class__.__name__))
        elif callable(model):
            prefix=kargs.pop("prefix",model.__name__)
            model=Model(model)
        else:
            raise RuntimeError("model should be either an lmfit.Model or a callable function, not a {}".format(type(model)))

        if prefix is not None:

            if isinstance(prefix,(list,tuple)):
                prefix=prefix[0]

            prefix=prefix.strip(" :")
            prefix="" if prefix == "" else prefix+":"

        else:
            if isinstance(prefix,(list,tuple)):
                prefix=prefix[0]

            if model.prefix=="":
                prefix=""
            else:
                prefix=model.prefix+":"

        if x is None:
            xl,xr=self.xlim() # pylint: disable=not-callable
            x=(xr-xl)*0.75+xl
        if y is None:
            yb,yt=self.ylim() # pylint: disable=not-callable
            y=0.5*(yt-yb)+yb

        try: # if the model has an attribute display params then use these as the parameter anmes
            for k,display_name in zip(model.param_names,model.display_names):
                if prefix:
                    self["{}{} label".format(prefix,k)]=display_name
                else:
                    self[k+" label"]=display_name
        except (AttributeError,KeyError):
            pass

        text= "\n".join([self.format("{}{}".format(prefix,k),fmt="latex",mode=mode) for k in model.param_names])
        try:
            self["{}chi^2 label".format(prefix)]=r"\chi^2"
            text+="\n"+self.format("{}chi^2".format(prefix),fmt="latex",mode=mode)
        except KeyError:
            pass
            
        if not text_only:
            ax=self.fig.gca()
            if "zlim" in ax.properties():
                #3D plot then
                if z is None:
                    zb,zt=ax.properties()["zlim"]
                    z=0.5*(zt-zb)+zb
                ax.text3D(x,y,z,text)
            elif "arrowprops" in kargs:
                ax.annotate(text, xy=(x,y), **kargs)
            else:
                ax.text(x,y,text,  **kargs)
            ret=self
        else:
            ret=text
        return ret

