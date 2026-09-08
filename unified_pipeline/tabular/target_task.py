"""Shim — canonical implementation lives in tabular.supervised.target_task."""
from .supervised.target_task import resolve_target, resolve_task

__all__ = ["resolve_target", "resolve_task"]
