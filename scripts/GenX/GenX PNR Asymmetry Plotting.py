"""Plot polarised neutron reflectivity, spin asymmetry and SLD from GenX exports."""

import numpy as np

from Stoner import DataFolder


def plot_exports(folder=False):
    """Plot three GenX datasets ordered by their dataset metadata and return the combined data."""
    files = (
        folder.clone
        if isinstance(folder, DataFolder)
        else DataFolder(folder, pattern="*.dat")
    )
    files.sort("dataset")
    up = files[2].clone
    up &= files[1]
    sld = files[0].clone
    up.column_headers = [
        "q_up",
        "Isim_up",
        "I_up",
        "dI_up",
        "q_down",
        "Isim_down",
        "I_down",
        "dI_down",
    ]
    up.plot.new_figure()
    ax = up.plot.line("q_up", "Isim_up", fmt="r-", label=None)
    up.plot.line("q_down", "Isim_down", ax=ax, fmt="b-", label=None)
    up.plot.errorbar(
        "q_up", "I_up", ax=ax, fmt="ro", yerr="dI_up", label="Up", ms=2
    )
    up.plot.errorbar(
        "q_down", "I_down", ax=ax, fmt="bx", yerr="dI_down", label="Down", ms=2
    )
    up.plot.xlabel = r"Momentum transfer $Q$ ($\AA^{-1}$)"
    up.plot.ylabel = "Normalised intensity"
    up.plot.title = "Spin up and down signals"
    up.plot.yscale = "log"

    up.diffsum("I_up", "I_down", header="SpinAsym_data")
    up.diffsum("Isim_up", "Isim_down", header="SpinAsym_sim")
    intensity_up, intensity_down = up.column("I_up"), up.column("I_down")
    total = intensity_up + intensity_down
    error = (
        2
        * np.sqrt(
            (intensity_down * up.column("dI_up")) ** 2
            + (intensity_up * up.column("dI_down")) ** 2
        )
        / total**2
    )
    up.add_column(error, "dSpinAsym")
    up.plot.new_figure()
    ax = up.plot.errorbar(
        "q_up",
        "SpinAsym_data",
        fmt="ro",
        yerr="dSpinAsym",
        label="Spin asymmetry data",
        ms=2,
    )
    up.plot.line(
        "q_up",
        "SpinAsym_sim",
        ax=ax,
        fmt="r-",
        label="Calculated spin asymmetry",
    )
    up.plot.xlabel = r"Momentum transfer $Q$ ($\AA^{-1}$)"
    up.plot.ylabel = "Spin asymmetry"
    up.plot.title = "Spin asymmetry data and calculation"

    sld.add("real sld +", "real sld -", header="Structure")
    sld.subtract("real sld +", "real sld -", header="Magnetic")
    sld.plot.new_figure()
    sld.plot.line(0, ["Structure", "Magnetic"])
    sld.plot.xlabel = "Distance z (nm)"
    sld.plot.ylabel = "Scattering potential"
    sld.plot.title = "Magnetic and structural scattering potentials"
    return up, sld


if __name__ == "__main__":
    plot_exports()[0].plot.show()
