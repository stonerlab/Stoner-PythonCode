"""Provides additional functionality for doing curve fitting to data."""

__all__ = ["odr_Model", "FittingMixin", "models"]
from .mixins import odr_Model, FittingMixin
from . import models
