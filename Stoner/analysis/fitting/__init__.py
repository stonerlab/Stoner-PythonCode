"""Provides additional functionality for doing curve fitting to data."""

__all__ = ["DataFitter", "ODR_Model", "models", "functions"]
from . import functions, models
from .accessor import DataFitter
from .classes import ODR_Model
