#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo of the make_model decorator."""
# pylint: disable=invalid-name, redefined-outer-name
from numpy import linspace
from numpy.random import normal, seed

from Stoner import Data
from Stoner.analysis.fitting.models import make_model

seed(12345)  # Ensure consistent random numbers


# Make our model
@make_model
def simple_model(x, m, c):
    """A straight line."""
    return x * m + c


# Add a function to guess parameters (optional, but)
@simple_model.guesser
def guess_vals(y, x=None):
    """Should guess parameter values really!"""
    m = (y.max() - y.min()) / (x[y.argmax()] - x[y.argmin()])
    c = x.mean() * m - y.mean()  # return one value per parameter
    return [m, c]


# Add a function to sry vonstraints on parameters (optional)
@simple_model.hinter
def hint_parameters():
    """Five some hints about the parameter."""
    return {"m": {"max": 10.0, "min": 0.0}, "c": {"max": 5.0, "min": -5.0}}


# Create some x,y data
x = linspace(0, 10, 101)
y = 4.5 * x - 2.3 + normal(scale=0.4, size=len(x))

# Make The Data object
d = Data(x, y, setas="xy", column_headers=["X", "Y"])

# Do the fit
d.lmfit(simple_model, result=True)

# Plot the result
d.setas = "xyy"
d.plot(fmt=["r+", "b-"])
d.title = "Simple Model Fit"
d.annotate_fit(simple_model, x=0.05, y=0.5)
