"""Shim — canonical implementation lives in tabular.supervised.models."""
from .supervised.models import create_estimator

__all__ = ["create_estimator"]
