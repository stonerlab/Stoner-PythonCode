"""Provides additional functionality for doing curve fitting to data."""
__all__ = ["odr_Model", "FittingMixin", "models"]
from . import models
from .mixins import FittingMixin, odr_Model
