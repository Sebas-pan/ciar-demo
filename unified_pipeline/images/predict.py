"""Shim — use images.detection.predict"""
from .detection.predict import predict_image, run_predictions, summarize_result, save_prediction
__all__ = ["predict_image", "run_predictions", "summarize_result", "save_prediction"]
