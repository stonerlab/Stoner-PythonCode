"""Provides additional functionality for doing curve fitting to data."""

__all__ = ["ODR_Model", "models", "functions"]
from . import functions, models
from .classes import ODR_Model
