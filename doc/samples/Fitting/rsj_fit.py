#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit IV data to various RSJ models."""
# pylint: disable=invalid-name
import os

from Stoner import Data, __home__
from Stoner.analysis.fitting.models.superconductivity import (
    RSJ_Noiseless,
    RSJ_Simple,
)

os.chdir(os.path.join(__home__, "..", "doc", "samples", "Fitting"))

data = Data("data/IV.txt", setas={"x": "Current", "y": "Voltage"})

# Fit data with both versions of the RSJ model
data.lmfit(RSJ_Simple, result=True, header="Simple", prefix="simple")
data.lmfit(RSJ_Noiseless, result=True, header="Noiseless", prefix="noiseless")

# Set column assignments and plot the data and fits
data.setas = {"x": "Current", "y": ["Voltage", "Simple", "Noiseless"]}
data.plot(fmt=["r+", "b-", "g-"])

# Annotate fits
data.annotate_fit(
    RSJ_Simple, prefix="simple", mode="eng", x=0.15, y=0.1, fontsize="small"
)
data.annotate_fit(
    RSJ_Noiseless,
    prefix="noiseless",
    mode="eng",
    x=0.55,
    y=0.1,
    fontsize="small",
)
