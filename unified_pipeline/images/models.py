"""Shim — use images.detection.models or images.classification.models directly."""
import warnings
warnings.warn("unified_pipeline.images.models is deprecated, use images.detection.models or images.classification.models", DeprecationWarning, stacklevel=2)
from .detection.models import create_estimator, identify_model_type

__all__ = ["create_estimator", "identify_model_type"]
