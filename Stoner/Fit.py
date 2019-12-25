"""Stoner.Fit: Functions and lmfit.Models for fitting data.

Functions should accept an array of x values and a number of parmeters,
they should then return an array of y values the same size as the x array.

Models are subclasses of lmfit.Model that represent the corresponding function

Please do keep documentation up to date, see other functions for documentation examples.

All the functions here defined for scipy.optimize.curve\_fit to call themm
i.e. the parameters are expanded to separate arguements.
"""
__all__ = [
    "Arrhenius",
    "BDR",
    "BlochGrueneisen",
    "FMR_Power",
    "FluchsSondheimer",
    "FowlerNordheim",
    "Ic_B_Airy",
    "Inverse_Kittel",
    "KittelEquation",
    "Langevin",
    "Linear",
    "Lorentzian_diff",
    "ModArrhenius",
    "NDimArrhenius",
    "PowerLaw",
    "Quadratic",
    "RSJ_Noiseless",
    "RSJ_Simple",
    "Simmons",
    "StretchedExp",
    "Strijkers",
    "TersoffHammann",
    "VFTEquation",
    "WLfit",
    "arrhenius",
    "bdr",
    "blochGrueneisen",
    "cfg_data_from_ini",
    "cfg_model_from_ini",
    "fluchsSondheimer",
    "fmr_power",
    "fowlerNordheim",
    "ic_B_airy",
    "inverse_kittel",
    "kittelEquation",
    "langevin",
    "linear",
    "lorentzian_diff",
    "make_model",
    "modArrhenius",
    "nDimArrhenius",
    "powerLaw",
    "quadratic",
    "rsj_noiseless",
    "rsj_simple",
    "simmons",
    "stretchedExp",
    "strijkers",
    "tersoffHammann",
    "vftEquation",
    "wlfit",
]

from warnings import warn

from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini, make_model, _get_model_

from Stoner.analysis.fitting.models.generic import (
    Linear,
    Lorentzian_diff,
    PowerLaw,
    Quadratic,
    StretchedExp,
    linear,
    lorentzian_diff,
    powerLaw,
    quadratic,
    stretchedExp,
)


from Stoner.analysis.fitting.models.thermal import (
    Arrhenius,
    ModArrhenius,
    NDimArrhenius,
    VFTEquation,
    arrhenius,
    modArrhenius,
    nDimArrhenius,
    vftEquation,
)

from Stoner.analysis.fitting.models.magnetism import (
    FMR_Power,
    Inverse_Kittel,
    KittelEquation,
    Langevin,
    fmr_power,
    inverse_kittel,
    kittelEquation,
    langevin,
)

from Stoner.analysis.fitting.models.tunnelling import (
    BDR,
    FowlerNordheim,
    Simmons,
    TersoffHammann,
    bdr,
    fowlerNordheim,
    simmons,
    tersoffHammann,
)

from Stoner.analysis.fitting.models.e_transport import (
    BlochGrueneisen,
    FluchsSondheimer,
    WLfit,
    blochGrueneisen,
    fluchsSondheimer,
    wlfit,
)

from Stoner.analysis.fitting.models.superconductivity import (
    RSJ_Noiseless,
    RSJ_Simple,
    Strijkers,
    Ic_B_Airy,
    rsj_noiseless,
    rsj_simple,
    strijkers,
    ic_B_airy,
)

warn("*" * 80 + "\nStoner.Fit is a depricated module - Stoner.analysis.fitting now!\n" + "*" * 80)
