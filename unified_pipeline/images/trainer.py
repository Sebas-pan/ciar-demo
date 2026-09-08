"""Shim — use images.detection.trainer or images.common.device."""
import warnings
warnings.warn("unified_pipeline.images.trainer is deprecated, use images.detection.trainer or images.common.device", DeprecationWarning, stacklevel=2)
from .detection.trainer import resolve_device, run_training
from .common.device import resolve_device as resolve_device_common

__all__ = ["resolve_device", "run_training"]
