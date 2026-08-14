import json
import os

import yaml

from .trainer import resolve_device


def _read_names(data):
    if isinstance(data, dict):
        return data["names"]
    with open(data, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["names"]


def run_evaluation(cfg, model, data, out_dir):
    """Valida el modelo sobre el split de test y guarda métricas + gráficas.

    Retorna dict con métricas mAP50/mAP50-95 por clase y globales.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 640))
    device = resolve_device(cfg)
    conf = train_cfg.get("conf", 0.25)
    iou = train_cfg.get("iou", 0.5)
    split = train_cfg.get("test_split", "test")

    results = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        split=split,
        project=out_dir,
        name="val",
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
        ap = results.box.ap  # (nc, 10) => [Iou@0.5..0.95]...
        for i, name in enumerate(names):
            metrics["per_class"][name] = {
                "AP50": float(results.box.ap50[i]) if results.box.ap50 is not None else None,
                "AP50-95": float(ap[i].mean()) if ap.ndim >= 1 else None,
            }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return metrics, metrics_path