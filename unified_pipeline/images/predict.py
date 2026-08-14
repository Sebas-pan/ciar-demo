import json
import os

from .trainer import resolve_device


def predict_image(cfg, model, image_path, conf=0.25, iou=0.5):
    """Predicción YOLO sobre una imagen. Retorna el objeto Results; boxes en:
    results.boxes.xyxy (Nx4), .conf (N), .cls (N), names = results.names.
    """
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 640))
    device = resolve_device(cfg)
    return model.predict(
        source=image_path,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
        stream=False,
    )[0]


def save_prediction(result, out_path):
    """Guarda la imagen anotada (imagen + cajas) en out_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    result.save(out_path)
    return out_path


def summarize_result(result):
    """Convierte un objeto Results a dict JSON-serializable."""
    boxes = result.boxes
    dets = []
    cls_names = getattr(result, "names", {})
    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        conf = boxes.conf.cpu().numpy().tolist()
        cls = boxes.cls.cpu().numpy().astype(int).tolist()
        for box, c, lbl in zip(xyxy, conf, cls):
            dets.append({
                "class": int(lbl),
                "class_name": cls_names.get(int(lbl), str(lbl)),
                "confidence": round(float(c), 4),
                "bbox_xyxy": [round(float(v), 1) for v in box],
            })
    return {"image": getattr(result, "path", None), "detections": dets}


def run_predictions(cfg, model, images, out_dir, conf=0.25, iou=0.5):
    """Ejecuta predicciones sobre una lista de rutas de imagen y las guarda.

    Retorna lista de dicts resumidos.
    """
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for img_path in images:
        result = predict_image(cfg, model, img_path, conf=conf, iou=iou)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(str(img_path)))[0] + "_pred.jpg")
        save_prediction(result, out_path)
        summary = summarize_result(result)
        summary["annotated"] = out_path
        summaries.append(summary)
    json_path = os.path.join(out_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    return summaries, json_path