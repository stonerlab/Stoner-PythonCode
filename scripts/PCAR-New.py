"""Python  script for Analysing PCAR data using Stoner classes and lmfit.

Gavin Burnell g.burnell@leeds.ac.uk
"""

# pylint: disable=invalid-name
import configparser as ConfigParser
import pathlib
import inspect

import numpy as np
from Stoner import Data
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini
from Stoner.analysis.fitting.models.generic import quadratic


class working(Data):
    """Utility class to manipulate data and plot it."""

    def __init__(self, *args, **kargs):
        """Initialise the fitting code."""
        super().__init__(*args, **kargs)

    def load_config(self):
        my_file = inspect.getfile(self.__class__)
        inifile = my_file.replace(".py", ".ini")
        if not pathlib.Path(inifile).exists():
            raise RuntimeError(
                f"Could not find the fitting ini file {inifile}!"
            )

        tmp = cfg_data_from_ini(inifile, filename=False)
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
            "Data", "dicard"
        ) and self.config.getboolean("Data", "discard")
        if discard:
            v_limit = self.config.get("Data", "v_limit")
            print("Discarding data beyond v_limit={}".format(v_limit))
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
                vmax, _ = self.max(self.vcol)
                vmin, _ = self.min(self.vcol)
                p, pv = self.curve_fit(
                    quadratic,
                    bounds=lambda x, y: (x > 0.9 * vmax) or (x < 0.9 * vmin),
                )
                print(
                    "Fitted normal conductance background of G="
                    + str(p[0])
                    + "V^2 +"
                    + str(p[1])
                    + "V+"
                    + str(p[2])
                )
                self["normalise.coeffs"] = p
                self["normalise.coeffs_err"] = np.sqrt(np.diag(pv))
                self.apply(
                    lambda x: x[self.gcol] / quadratic(x[self.vcol], *p),
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
            self.data = self.data[~np.any(np.isnan(self.data), axis=1)]
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
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=2)
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
        self.RescaleV().Discard().offset_correct().Decompose().Normalise()
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


if __name__ == "__main__":
    d = working()
    d.load_config()
    fit = d.Fit()
