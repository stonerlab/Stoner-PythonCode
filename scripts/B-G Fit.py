"""Fit a Bloch-Gruneisen resistance model to temperature-dependent data."""

import re

import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import BlochGrueneisen


def select_col(data, message):
    """Prompt for a valid column index when the header cannot be identified."""
    for i, col in enumerate(data.column_headers):
        print(f"{i} : {col}")
    while True:
        try:
            column = int(input(message))
            if 0 <= column < data.shape[1]:
                return column
        except ValueError:
            pass
        print("Please select a column number from the list")


def guess_col(data, patterns, message):
    """Find one matching column, otherwise ask the user to select it."""
    for pattern in patterns:
        try:
            columns = data.find_col(re.compile(pattern), force_list=True)
        except KeyError:
            continue
        if len(columns) == 1:
            return columns[0]
    return select_col(data, message)


def fit(filename=False):
    """Load or copy data, fit resistance versus temperature and return the plotted Data."""
    data = Data(filename)
    temperature = guess_col(
        data, [r"^[Tt][\s\(]", r"[Tt]emp"], "Temperature column: "
    )
    resistance = guess_col(data, [r"[Rr]ho", r"[Rr]es"], "Resistance column: ")
    data.mask |= np.isnan(data.to_numpy(masked=False))
    data.del_rows()
    data.setas(x=temperature, y=resistance)
    model = BlochGrueneisen()
    model.set_param_hint("n", vary=False, value=5)
    print(f"Initial guesses: {model.guess(data.y, x=data.x)}")
    data.fit.lmfit(model, absolute_sigma=False, result=True, header="Bloch")
    data.setas(Bloch="y", reset=False)
    ax = data.plot(
        fmt=["r.", "b-"], label=["Data", "Bloch-Grüneisen fit"], markersize=1
    )
    data.plot.xlabel = "Temperature (K)"
    data.plot.ylabel = r"Resistance ($\Omega$)"
    data.plot.annotate_fit(model, ax=ax, x=0.05, y=0.35, mode="eng")
    return data


if __name__ == "__main__":
    fit().plot.show()
