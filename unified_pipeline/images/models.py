import os

YOLO_VARIANTS = {
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "yolov11l": "yolo11l.pt",
    "yolov11x": "yolo11x.pt",
}

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def create_estimator(cfg):
    """Crea un modelo YOLO11 de ultralytics a partir de la config.

    model.type: 'yolo11' (o 'yolo11_detection').
    model.variant: 'yolov11n|s|m|l|x' (default 'yolov11n').
    Resuelve el checkpoint (descarga a unified_pipeline/weights/ si falta).
    """
    from ultralytics import YOLO
    from ultralytics.utils.downloads import safe_download

    model_cfg = cfg["model"]
    weights = model_cfg.get("weights")
    variant = model_cfg.get("variant", "yolov11n")

    if weights:
        if not os.path.exists(weights):
            raise FileNotFoundError("YOLO weights not found: {}".format(weights))
        return YOLO(weights)

    if variant not in YOLO_VARIANTS:
        raise ValueError(
            "Unsupported YOLO variant '{}'. Use one of {}.".format(
                variant, ", ".join(sorted(YOLO_VARIANTS))
            )
        )

    local = os.path.join(_WEIGHTS_DIR, YOLO_VARIANTS[variant])
    if not os.path.exists(local):
        os.makedirs(_WEIGHTS_DIR, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}".format(
            YOLO_VARIANTS[variant])
        if not safe_download(url=url, file=local, unzip=False, delete=False):
            raise RuntimeError("Could not download {}".format(url))

    # cargamos con ruta absoluta para evitar depender del weights_dir global
    model = YOLO(local)
    model._path = model.ckpt_path = local
    return model


def identify_model_type():
    return "image_yolo_detection"