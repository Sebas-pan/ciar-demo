"""Dispatcher — routes to detection (YOLO) or classification based on model.type."""
import warnings

_CLASSIFICATION_TYPES = {
    "vgg16", "vgg16_bn", "vgg16-bn", "vgg_bn", "vgg",
    "resnet50", "resnet", "resnet-50", "resnet_50",
    "resnet18", "resnet_18", "resnet101", "resnet_101",
    "efficientnet_b0", "efficientnet", "efficientnetb0",
    "mobilenet_v3_large", "mobilenet_v3", "mobilenet", "mobilenetv3",
    "densenet121", "densenet", "densenet_121",
    "vit_b_16", "vit", "vitb16", "vit_b16",
    "swin_t", "swin", "swint", "swin_tiny",
    "convnext_tiny", "convnext", "convnexttiny",
}

_DETECTION_TYPES = {
    "yolo11", "yolo11_detection", "yolov8", "yolov8_detection",
    "yolo26", "yolo26_detection", "yolo", "detection",
}


def run_image(cfg):
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "yolo11")).lower()
    variant = str(model_cfg.get("variant", "")).lower()
    key = model_type.replace("-", "_")
    vkey = variant.replace("-", "_")
    if key in _CLASSIFICATION_TYPES or vkey in _CLASSIFICATION_TYPES:
        from .classification.pipeline import run_classification
        return run_classification(cfg)
    if key in _DETECTION_TYPES or vkey.startswith("yolov") or vkey.startswith("yolo"):
        from .detection.pipeline import run_image as _run_detection
        return _run_detection(cfg)
    # Fallback: variant names decide, else default to detection (legacy).
    if key.startswith("yolo") or vkey.startswith("yolo"):
        from .detection.pipeline import run_image as _run_detection
        return _run_detection(cfg)
    from .classification.pipeline import run_classification
    return run_classification(cfg)
