#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Ic(B) to Airy function.
"""

from Stoner import Data, __home__
from Stoner.Fit import Ic_B_Airy

import os

os.chdir(os.path.join(__home__, "..", "doc", "samples", "Fitting"))

data = Data("data/Ic_B.txt", setas={"x": "Magnet Output", "y": "Ic"})

data.lmfit(Ic_B_Airy, result=True, header="Fit")

data.setas = {"x": "Magnet Output", "y": ["Ic", "Fit"]}
data.plot(fmt=["r+", "b-"])

data.annotate_fit(Ic_B_Airy, mode="eng", x=0.6, y=0.5, fontsize="small")

data.title = "Critical current vs Field for $4\mu m$ junction"
data.xlabel = r"Magnetic Field $\mu_0H (\mathrm{T})$"
data.ylabel = r"Critical Current $I_c (A)$"
