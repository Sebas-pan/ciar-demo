"""Shim — canonical implementation lives in tabular.supervised.pipeline."""
import warnings
warnings.warn(
    "unified_pipeline.tabular.pipeline is deprecated, use tabular.supervised.pipeline",
    DeprecationWarning,
    stacklevel=2,
)
from .supervised.pipeline import run_tabular

__all__ = ["run_tabular"]
