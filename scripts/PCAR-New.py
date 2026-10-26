"""Python  script for Analysing PCAR data using Stoner classes and lmfit.

Gavin Burnell g.burnell@leeds.ac.uk
"""

# pylint: disable=invalid-name
import configparser as ConfigParser
import pathlib
from importlib import import_module

import numpy as np
from Stoner import Data
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini
from Stoner.analysis.fitting.models.generic import quadratic
from Stoner.tools import copy_into


class working(Data):
    """Utility class to manipulate data and plot it."""

    def load_config(self, inifile=None, filename=None):
        """Load data and fitting options from the accompanying INI file."""
        inifile = str(inifile or pathlib.Path(__file__).with_suffix(".ini"))
        if not pathlib.Path(inifile).exists():
            raise RuntimeError(
                f"Could not find the fitting ini file {inifile}!"
            )

        tmp = cfg_data_from_ini(inifile, filename=filename)
        copy_into(tmp, self)
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
        self.diagnostics = config.has_option(
            "Options", "diagnostics"
        ) and config.getboolean("Options", "diagnostics")
        self.method = config.get("Options", "method")
        self.model = model
        self.p0 = p0
        return self

    def Discard(self):
        """Optionally throw out some high bias data."""
        discard = self.config.has_option(
            "Data", "discard"
        ) and self.config.getboolean("Data", "discard")
        if discard:
            v_limit = self.config.getfloat("Data", "v_limit")
            print(f"Discarding data beyond v_limit={v_limit}")
            self.del_rows(self.vcol, lambda x, y: abs(x) > v_limit)
        return self

    def RescaleV(self):
        """Rescale the voltage data if the options are set."""
        if self.config.has_option(
            "Options", "rescale_v"
        ) and self.config.getboolean("Options", "rescale_v"):
            vscale = self.config.getfloat("Data", "v_scale")
            self[:, self.vcol] = self.column(self.vcol) * vscale
            print(f"Rescaled voltage data by {vscale}")
        return self

    def Preprocess(self):
        """Run a configured preprocessing function over the data."""
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
                    normaliser = (
                        quadratic
                        if normaliser_name == "quadratic"
                        else globals()[normaliser_name]
                    )
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
                    f"{normaliser_name}(V,{','.join(map(str, p))})",
                ).format(*p)
                print(
                    f"Fitted normal conductance background of G={normaliser_repr}"
                )
                self["normalise.coeffs"] = p
                self["normalise.coeffs_err"] = np.sqrt(np.diag(pv))
                self["normaliser_name"] = normaliser_name
                self[:, self.gcol] = self.column(self.gcol) / normaliser(
                    self.column(self.vcol), *p
                )
            else:
                if not np.isfinite(Gn) or Gn == 0:
                    raise ValueError(
                        "Normal_conductance must be finite and non-zero"
                    )
                self[:, self.gcol] = self.column(self.gcol) / Gn
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
                width=max(5, len(self) // 20),
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
            self[:, self.vcol] = self.column(self.vcol) - offset
        return self

    def Decompose(self):
        """Optionally decompose the signal intro symmetric and antisymmetric parts."""
        if self.config.has_option(
            "Options", "decompose"
        ) and self.config.getboolean("Options", "decompose"):
            print("Doing decomposition to symmetrize data")
            self.decompose(xcol=self.vcol, ycol=self.gcol, sym=self.gcol)
            self.setas(x=self.vcol, y=self.gcol)
            self.del_rows(
                ~np.any(np.isnan(self.to_numpy(masked=False)), axis=1)
            )
        return self

    def plot_results(self):
        """Do the plotting of the data and the results."""
        self.plot.new_figure()
        ax = self.plot.line(
            self.vcol,
            [self.gcol, "Fit"],
            fmt=["ro", "b-"],
            label=["Data", "Fit"],
        )
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=2)
        if self.fancyresults:
            self.annotate_fit(
                self.model,
                ax=ax,
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
        if self.diagnostics:
            self.plot(label="Raw")
        for step in [
            "Preprocess",
            "RescaleV",
            "Discard",
            "offset_correct",
            "Decompose",
            "Normalise",
        ]:
            step_method = getattr(self, step)
            step_method()
            if self.diagnostics:
                self.plot(label=step)

        chi2 = self.p0.shape[0] > 1

        method = getattr(self, self.method.replace("-", "_"))

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
            figure = fit.plot.new_figure()
            columns = fit.setas.cols
            with fit.plot.StackVertical(
                len(columns.ycol), figure=figure
            ) as panels:
                for i, ycol in enumerate(columns.ycol):
                    yerr = columns.yerr[i] if i < len(columns.yerr) else False
                    options = {"capsize": 3} if yerr is not False else {}
                    fit.plot(y=ycol, yerr=yerr, ax=panels[i], **options)
                fit.plot.yscale = "log"
        if self.save_fit:
            fit.filename = None
            fit.save(False)
        return fit


def quadratic_abs(x, a, b, c):
    """A function that returns a quadratic expression, but in terms of abs(x)."""
    x = np.abs(x)
    return c + x * (b + a * x)


def _pass(x):
    """A do nothing preprocess function."""
    return x


def down_only(data):
    """Keep the first complete falling sweep between well-separated Set V turning points."""
    voltage = data.column("Set V")
    ix = np.flatnonzero(np.diff(np.sign(np.diff(voltage))) != 0) + 1
    dix = np.diff(ix) > 10
    dix = np.append(dix, True)
    ix = ix[dix]
    for start, stop in zip(ix[:-1], ix[1:]):
        if voltage[start] > voltage[stop]:
            keep = np.zeros(len(data), dtype=bool)
            keep[start : stop + 1] = True
            data.del_rows(keep)
            return data
    raise ValueError(
        "The down-sweep preprocessor requires a complete falling sweep between two turning points"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filename",
        nargs="?",
        type=pathlib.Path,
        help="Data file; omitted opens a file dialog",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_suffix(".ini"),
    )
    args = parser.parse_args()
    d = working()
    d.load_config(args.config, args.filename)
    fit = d.Fit()
    if d.show_plot or d.diagnostics:
        d.plot.show()
