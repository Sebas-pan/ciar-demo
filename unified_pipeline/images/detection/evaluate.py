import json
import os

import yaml

from .trainer import resolve_device


def _read_names(data):
    if isinstance(data, dict):
        names = data["names"]
    else:
        with open(data, "r", encoding="utf-8") as fh:
            names = yaml.safe_load(fh)["names"]
    if isinstance(names, dict):
        try:
            keys = sorted(names.keys(), key=lambda k: int(k))
        except (ValueError, TypeError):
            keys = sorted(names.keys(), key=str)
        return [str(names[k]) for k in keys]
    return [str(n) for n in names]


def run_evaluation(cfg, model, data, out_dir):
    """Valida el modelo sobre el split de test y guarda métricas + gráficas.

    Retorna dict con métricas mAP50/mAP50-95 por clase y globales.
    """
    import warnings
    os.makedirs(out_dir, exist_ok=True)
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 640))
    device = resolve_device(cfg)
    conf = train_cfg.get("conf", 0.25)
    iou = train_cfg.get("iou", 0.5)
    split = train_cfg.get("test_split", "test")
    # Fall back to val when the generated data.yaml has no test split.
    try:
        with open(data, "r", encoding="utf-8") as fh:
            _spec = yaml.safe_load(fh) or {}
        if split not in _spec:
            warnings.warn(
                "split '{}' not in data.yaml; falling back to 'val'".format(split))
            split = "val" if "val" in _spec else "test"
    except (OSError, ValueError):
        pass
    # Workers must come from config: ultralytics defaults to 8, which crashes
    # on Windows the same way as training (see trainer.workers). Default 0.
    workers = int(train_cfg.get("workers", 0))

    results = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        split=split,
        workers=workers,
        project=os.path.dirname(out_dir),
        name=os.path.basename(out_dir),
        exist_ok=True,
        plots=True,
        save_json=True,
    )

    names = _read_names(data)
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": {},
    }
    if results.box.ap is not None:
        ap = results.box.ap  # (nc, 10) or (10,) when nc==1
        import numpy as np
        ap = np.asarray(ap, dtype=float)
        if ap.ndim == 1:
            ap = ap.reshape(1, -1)
        for i, name in enumerate(names):
            if i >= ap.shape[0]:
                break
            metrics["per_class"][name] = {
                "AP50": float(results.box.ap50[i]) if results.box.ap50 is not None and i < len(results.box.ap50) else None,
                "AP50-95": float(ap[i].mean()),
            }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return metrics, metrics_path