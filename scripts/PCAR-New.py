"""Python  script for Analysing PCAR data using Stoner classes and lmfit.

Gavin Burnell g.burnell@leeds.ac.uk
"""

# pylint: disable=invalid-name
import configparser as ConfigParser
import pathlib
import inspect
from importlib import import_module

import numpy as np
from Stoner import Data
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini


class working(Data):
    """Utility class to manipulate data and plot it."""

    def __init__(self, *args, **kwargs):
        """Define local attributes."""
        super().__init__(*args, **kwargs)
        self.vcol = None
        self.gcol = None
        self.config = None
        self.show_plot = True
        self.save_fit = True
        self.report = True
        self.fancyresults = True
        self.method = "lmfit"
        self.model = (
            "Stoner.analysis.fitting.models.superconductivity.Strijkers"
        )
        self.p0 = None

    def load_config(self):
        """Load the config file to set up the fitting."""
        my_file = inspect.getfile(self.__class__)
        inifile = my_file.replace(".py", ".ini")
        if not pathlib.Path(inifile).exists():
            raise RuntimeError(
                f"Could not find the fitting ini file {inifile}!"
            )

        tmp = cfg_data_from_ini(inifile)
        self.setas = tmp.setas.clone
        self.column_headers = tmp.column_headers
        self.metadata = tmp.metadata
        self.data = tmp.data
        self.vcol = self.find_col(self.setas["x"])
        self.gcol = self.find_col(self.setas["y"])
        self.filename = tmp.filename

        model, p0 = cfg_model_from_ini(inifile, data=self)

        for k in model.param_hints:
            setattr(self, k, model.param_hints[k])

        config = ConfigParser.ConfigParser()
        config.read(inifile)
        self.config = config

        # Some config variables we'll need later
        self.show_plot = config.getboolean("Options", "show_plot")
        self.save_fit = config.getboolean("Options", "save_fit")
        self.report = config.getboolean("Options", "print_report")
        self.fancyresults = config.has_option(
            "Options", "fancy_result"
        ) and config.getboolean("Options", "fancy_result")
        self.method = config.get("Options", "method")
        self.model = model
        self.p0 = p0

    def Discard(self):
        """Optionally throw out some high bias data."""
        discard = self.config.has_option(
            "Data", "discard"
        ) and self.config.getboolean("Data", "discard")
        if discard:
            v_limit = self.config.get("Data", "v_limit")
            print(f"Discarding data beyond v_limit={v_limit}")
            self.del_rows(self.vcol, lambda x, y: abs(x) > v_limit)
        return self

    def RescaleV(self):
        """Rescale the voltage data if the options are set."""
        if self.config.has_option(
            "Options", "rescale_v"
        ) and self.config.getboolean("Options", "rescale_v"):
            vscale = self.config.getfloat("Data", "v_scale")
            self.data[:, self.vcol] *= vscale
            print(f"Rescaled voltage data by {vscale}")
        return self

    def Preprocess(self):
        """Run an arbitart function over the data if the option is specified."""
        preprocessor_name = self.config.get(
            "Options", "preprocessor", fallback="_pass"
        )
        if "." in preprocessor_name:
            module = ".".join(preprocessor_name.split(".")[:-1])
            preprocessor = preprocessor_name.split(".")[-1]
            module = import_module(module)
            preprocessor = getattr(module, preprocessor)
        else:
            preprocessor = globals()[preprocessor_name]

        return preprocessor(self)

    def Normalise(self):
        """Normalise the data if the relevant options are turned on in the config file.

        Use either a simple normalisation constant or go fancy and try to use a background function.
        """
        if self.config.has_option(
            "Options", "normalise"
        ) and self.config.getboolean("Options", "normalise"):
            print("Normalising Data")
            Gn = self.config.getfloat("Data", "Normal_conductance")
            if self.config.has_option(
                "Options", "fancy_normaliser"
            ) and self.config.getboolean("Options", "fancy_normaliser"):
                fraction = 1 - self.config.getfloat(
                    "Data", "background_fraction", fallback=0.1
                )
                normaliser_name = self.config.get(
                    "Options", "normaliser_function", fallback="quadratic"
                )
                if "." in normaliser_name:
                    module = ".".join(normaliser_name.split(".")[:-1])
                    normaliser = normaliser_name.split(".")[-1]
                    module = import_module(module)
                    normaliser = getattr(module, normaliser)
                else:
                    normaliser = globals()[normaliser_name]
                vmax, _ = self.max(self.vcol)
                vmin, _ = self.min(self.vcol)
                p, pv = self.curve_fit(
                    normaliser,
                    bounds=lambda x, y: (x > fraction * vmax)
                    or (x < fraction * vmin),
                )
                normaliser_repr = getattr(
                    normaliser,
                    "representation",
                    f"{normaliser_name}(V,{','.join(p)})",
                ).format(p)
                print(
                    f"Fitted normal conductance background of G={normaliser_repr}"
                )
                self["normalise.coeffs"] = p
                self["normalise.coeffs_err"] = np.sqrt(np.diag(pv))
                self["normaliser_name"] = normaliser_name
                self.apply(
                    lambda x: x[self.gcol] / normaliser(x[self.vcol], *p),
                    self.gcol,
                    header=self.column_headers[self.gcol],
                )
            else:
                self.apply(
                    lambda x: x[self.gcol] / Gn,
                    self.gcol,
                    header=self.column_headers[self.gcol],
                )
        return self

    def offset_correct(self):
        """Centre the data.

        - look for peaks and troughs within 5 of the initial delta value
        take the average of these and then subtract it.
        """
        if self.config.has_option(
            "Options", "remove_offset"
        ) and self.config.getboolean("Options", "remove_offset"):
            print("Doing offset correction")
            peaks = self.peaks(
                ycol=self.gcol,
                width=len(self) / 20,
                xcol=self.vcol,
                poly=4,
                peaks=True,
                troughs=True,
                full_data=False,
            )
            peaks = list(
                filter(lambda x: abs(x) < 4 * self.delta["value"], peaks)
            )
            if peaks:
                offset = np.mean(np.array(peaks))
                print(
                    f"Removing offset by peaks method - Mean offset = {offset}"
                )
            else:
                v_data = self // self.vcol
                offset = (v_data.min() + v_data.max()) / 2
                print(
                    f"Removing offset by v-range method - Mean offset = {offset}"
                )
            self[:, self.vcol] -= offset
        return self

    def Decompose(self):
        """Optionally decompose the signal intro symmetric and antisymmetric parts."""
        if self.config.has_option(
            "Options", "decompose"
        ) and self.config.getboolean("Options", "decompose"):
            print("Doing decomposition to symmetrize data")
            self.decompose(xcol=self.vcol, ycol=self.gcol, sym=self.gcol)
            self.setas(x=self.vcol, y=self.gcol)
            self.data = self.data[
                ~np.any(np.isnan([self.data[:, self.gcol]]), axis=1)
            ]
        return self

    def plot_results(self):
        """Do the plotting of the data and the results."""
        self.figure()  # Make a new figure and show the results
        self.plot_xy(
            self.vcol,
            [self.gcol, "Fit"],
            fmt=["ro", "b-"],
            label=["Data", "Fit"],
        )
        bbox_props = {
            "boxstyle": "square,pad=0.3",
            "fc": "white",
            "ec": "b",
            "lw": 2,
        }
        if self.fancyresults:
            self.annotate_fit(
                self.model,
                x=0.05,
                y=0.65,
                xycoords="axes fraction",
                bbox=bbox_props,
                fontsize=11,
            )
        return self

    def Fit(self):
        """Run the fitting code."""
        # Run a pre-fitting data munge chain
        self.Preprocess().RescaleV().Discard().offset_correct().Decompose().Normalise()
        chi2 = self.p0.shape[0] > 1

        method = getattr(self, self.method)

        if not chi2:  # Single fit mode, consider whether to plot and save etc
            fit = method(
                self.model,
                p0=self.p0,
                result=True,
                header="Fit",
                output="report",
            )

            if self.show_plot:
                self.plot_results()
            if self.save_fit:
                self.save(False)
            if self.report:
                print(fit.fit_report())
            return fit
        d = Data(self)
        fit = d.lmfit(
            self.model, p0=self.p0, result=True, header="Fit", output="data"
        )

        if self.show_plot:
            fit.plot(multiple="panels", capsize=3)
            fit.yscale = "log"  # Adjust y scale for chi^2
            fit.tight_layout()
        if self.save_fit:
            fit.filename = None
            fit.save(False)
        return None


def quadratic_abs(x, a, b, c):
    """A function that returns a quadratic expression, but in terms of abs(x)."""
    x = np.abs(x)
    return c + x * (b + a * x)


def _pass(x):
    """A do nothing preprocess function."""
    return x


def down_only(data):
    """Selects only the down sweep data - this is a hack that works becuase Set V is not noisy"""
    ix = np.where(np.diff(np.sign(np.diff(data["Set V"]))) != 0)[:2][0]
    dix = np.diff(ix) > 10
    dix = np.append(dix, True)
    ix = ix[dix]
    ix = ix[:2]

    data.data = data.data[slice(*ix)]
    return data


if __name__ == "__main__":
    d = working()
    d.load_config()
    fit = d.Fit()
