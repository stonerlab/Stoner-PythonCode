"""Fit critical temperatures in resistance-versus-temperature ramps."""

from pathlib import Path

import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.generic import linear

DEFAULT_FILE = (
    Path(__file__).resolve().parents[1]
    / "sample-data"
    / ("6221-Lockin-DAQ Temperature Control -30.0Deg 0.004T.txt")
)


def fit(
    filename=DEFAULT_FILE,
    t_col=": T2",
    r_cols=("Nb::R", "YIG-Nb-Co::R"),
    iterator="iterator",
    threshold=0.85,
):
    """Fit each ramp independently and return the critical temperatures and uncertainties.

    Notes:
        Each resistance column is normalised independently. Fits use only the
        central transition, with resistance as X and temperature as Y. The
        input file or Data object is copied; figures use explicit panel axes.
    """
    source = Data(filename)
    result = Data()
    for ramp in source.split(iterator):
        if not len(ramp):
            continue
        row = [ramp.mean(iterator)]
        figure = ramp.plot.new_figure(figsize=(8, 4))
        with ramp.plot.MultiPanel(
            (1, len(r_cols)), figure=figure, adjust_figsize=False
        ) as panels:
            for i, r_col in enumerate(r_cols):
                data = ramp.clone
                data.setas(x=t_col, y=r_col)
                data.del_rows(np.isfinite(data.x) & np.isfinite(data.y))
                data.normalise(base=(-1.0, 1.0), replace=True)
                data = ~data
                data.curve_fit(
                    linear,
                    bounds=lambda x, r: -threshold < x < threshold,
                    result=True,
                    p0=[7.0, 0.0],
                )
                data.setas[-1] = "y"
                ax = data.plot(
                    ax=panels[i],
                    fmt=["k.", "r-"],
                    title=f"Ramp {row[0]}: {r_col}",
                )
                data.annotate_fit(linear, ax=ax, x=0.1, y=0.1, fontsize="small")
                row.extend(
                    [data["linear:intercept"], data["linear:intercept err"]]
                )
        result += np.array(row)
    result.column_headers = [
        "Ramp",
        *[label for col in r_cols for label in (f"{col} Tc", f"{col} dTc")],
    ]
    result.setas = "x" + "ye" * len(r_cols)
    result.plot.new_figure()
    result.plot(fmt=["."] * len(r_cols))
    return result


if __name__ == "__main__":
    fit().plot.show()
