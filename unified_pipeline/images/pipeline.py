"""Dispatcher — routes to detection (YOLO) or classification (VGG16/ResNet50) based on model.type."""
import warnings

def run_image(cfg):
    model_type = str(cfg.get("model", {}).get("type", "yolo11")).lower()
    if model_type in ("vgg16", "vgg16_bn", "vgg16-bn", "vgg_bn", "vgg",
                      "resnet50", "resnet", "resnet-50", "resnet_50"):
        from .classification.pipeline import run_classification
        return run_classification(cfg)
    # default: YOLO detection
    from .detection.pipeline import run_image as _run_detection
    # also support legacy name run_image inside detection
    # detection pipeline exposes run_image; we call it
    return _run_detection(cfg)
