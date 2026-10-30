#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit Ic(B) to Airy function."""

# pylint: disable=invalid-name
import os

from Stoner import Data, __home__
from Stoner.analysis.fitting.models.superconductivity import Ic_B_Airy

os.chdir(os.path.join(__home__, "..", "doc", "samples", "Fitting"))

data = Data("data/Ic_B.txt", setas={"x": "Magnet Output", "y": "Ic"})

data.fit.lmfit(Ic_B_Airy, result=True, header="Fit")

data.setas = {"x": "Magnet Output", "y": ["Ic", "Fit"]}
data.plot(fmt=["r+", "b-"])

data.plot.annotate_fit(Ic_B_Airy, mode="eng", x=0.6, y=0.5, fontsize="small")

data.plot.title = r"Critical current vs Field for $4\mu m$ junction"
data.plot.xlabel = r"Magnetic Field $\mu_0H (\mathrm{T})$"
data.plot.ylabel = r"Critical Current $I_c (A)$"
