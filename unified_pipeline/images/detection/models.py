import os
import shutil

YOLO_VARIANTS = {
    "yolov11n": "yolo11n.pt",
    "yolo11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolo11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "yolo11m": "yolo11m.pt",
    "yolov11l": "yolo11l.pt",
    "yolo11l": "yolo11l.pt",
    "yolov11x": "yolo11x.pt",
    "yolo11x": "yolo11x.pt",
    "yolov8n": "yolov8n.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26m": "yolo26m.pt",
}

# Per-variant download base (v8.4.0 tag predates YOLO26).
_VARIANT_URL_BASE = {
    "yolo26n": "https://github.com/ultralytics/assets/releases/download/v8.4.164/",
}
_DEFAULT_URL_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# unified_pipeline/weights/ (three levels up from this file), as documented.
_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "weights")


def create_estimator(cfg):
    """Crea un modelo YOLO de ultralytics a partir de la config.

    model.type: 'yolo11' | 'yolov8' | 'yolo26' (o '*_detection').
    model.variant: 'yolov11n|s|m|l|x' | 'yolov8n' | 'yolo26n' (default 'yolov11n').
    Resuelve el checkpoint (reutiliza yolo26n.pt de la raiz si existe,
    descarga a unified_pipeline/weights/ si falta, o deja que ultralytics
    resuelva el cache como ultimo recurso).
    """
    from ultralytics import YOLO
    from ultralytics.utils.downloads import safe_download

    model_cfg = cfg["model"]
    weights = model_cfg.get("weights")
    variant = str(model_cfg.get("variant", "yolov11n")).strip().lower()

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

    filename = YOLO_VARIANTS[variant]
    local = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(local):
        # Reuse repo-root copy (e.g. yolo26n.pt already downloaded).
        root_copy = os.path.join(_REPO_ROOT, filename)
        if os.path.exists(root_copy):
            os.makedirs(_WEIGHTS_DIR, exist_ok=True)
            shutil.copy(root_copy, local)
    if not os.path.exists(local):
        os.makedirs(_WEIGHTS_DIR, exist_ok=True)
        base = _VARIANT_URL_BASE.get(variant, _DEFAULT_URL_BASE)
        url = "{}{}".format(base, filename)
        try:
            ok = safe_download(url=url, file=local, unzip=False, delete=False)
        except Exception as exc:
            ok = False
            dl_exc = exc
        else:
            dl_exc = None
        if not ok:
            # Remove partial download so the next run retries cleanly.
            for p in (local, local + ".part"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            # Last resort: let ultralytics resolve its own cache/download.
            try:
                return YOLO(filename)
            except Exception:
                pass
            raise RuntimeError("Could not download {}".format(url)) from dl_exc

    # cargamos con ruta absoluta para evitar depender del weights_dir global
    model = YOLO(local)
    model._path = model.ckpt_path = local
    return model


def identify_model_type():
    return "image_yolo_detection"