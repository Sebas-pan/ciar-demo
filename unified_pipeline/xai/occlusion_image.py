import os

import cv2
import numpy as np

from ..images.predict import predict_image, summarize_result
from ..io import write_json
from .overlay import compose_overlay


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cell_overlap_weight(box, cell_rect):
    """Fraction of the box area covered by the cell rect (0..1)."""
    x1 = max(box[0], cell_rect[0]); y1 = max(box[1], cell_rect[1])
    x2 = min(box[2], cell_rect[2]); y2 = min(box[3], cell_rect[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / area if area > 0 else 0.0


def _predict_array(cfg, model, img_bgr, out_dir, conf=0.01, iou=0.5):
    """Predict on an in-memory BGR array (no JPEG round-trip).

    Falls back to a unique temp file if the backend rejects ndarrays.
    """
    import tempfile
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 640))
    from ..images.detection.trainer import resolve_device
    device = resolve_device(cfg)
    try:
        res = model.predict(
            source=img_bgr, imgsz=imgsz, device=device,
            conf=conf, iou=iou, verbose=False, stream=False)[0]
        return summarize_result(res)["detections"]
    except Exception:
        fd, tmp = tempfile.mkstemp(suffix=".jpg", dir=out_dir)
        os.close(fd)
        try:
            cv2.imwrite(tmp, img_bgr)
            return summarize_result(
                predict_image(cfg, model, tmp, conf=conf, iou=iou))["detections"]
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def run_occlusion(cfg, model, image_paths, out_dir,
                  grid=(8, 8), fill="gray", min_conf=0.25):
    """Occlusion Sensitivity model-agnostic.

    Paraliza cada celda de una grilla sobre la imagen, re-predice y mide la
    caída de confianza de las detecciones de referencia cuya caja cae en esa
    celda. Devuelve mapa de atribución y overlay.
    """
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(grid, int):
        grid = (grid, grid)
    cols, rows = grid

    records = []
    artifacts = []
    names = model.names

    for img_path in image_paths:
        result = predict_image(cfg, model, str(img_path), conf=min_conf, iou=0.5)
        dets = summarize_result(result)["detections"]
        if not dets:
            continue

        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        H, W = orig.shape[:2]
        refs = [{
            "box": det["bbox_xyxy"],
            "cls": det["class"],
            "conf": det["confidence"],
            "cx": (det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2,
            "cy": (det["bbox_xyxy"][1] + det["bbox_xyxy"][3]) / 2,
        } for det in dets]

        cell_w = W / cols
        cell_h = H / rows
        attribution = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                occluded = orig.copy()
                x1, y1 = int(c * cell_w), int(r * cell_h)
                x2, y2 = min(int((c + 1) * cell_w), W), min(int((r + 1) * cell_h), H)
                if fill == "blur":
                    patch = cv2.GaussianBlur(occluded[y1:y2, x1:x2], (31, 31), 0)
                    occluded[y1:y2, x1:x2] = patch
                else:
                    occluded[y1:y2, x1:x2] = (128, 128, 128)

                tmp = os.path.join(out_dir, "_occluded_tmp.jpg")
                new_dets = _predict_array(cfg, model, occluded, out_dir,
                                          conf=0.01, iou=0.5)

                cell_rect = (c * cell_w, r * cell_h,
                             (c + 1) * cell_w, (r + 1) * cell_h)
                for ref in refs:
                    # Credit every cell overlapping the box, weighted by
                    # overlap fraction (not just the center cell).
                    w = _cell_overlap_weight(ref["box"], cell_rect)
                    if w <= 0:
                        continue
                    best = 0.0
                    for nd in new_dets:
                        if nd["class"] != ref["cls"]:
                            continue
                        if _iou(ref["box"], nd["bbox_xyxy"]) > 0.3:
                            best = max(best, nd["confidence"])
                    drop = max(0.0, ref["conf"] - best)
                    attribution[r, c] += w * drop

        if attribution.max() > 0:
            attribution = attribution / attribution.max()

        # upscale al tamaño de la imagen (clip: CUBIC overshoots [0,1])
        heat = cv2.resize(attribution, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0, 1)
        heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

        boxes = np.array([[d["bbox_xyxy"][0], d["bbox_xyxy"][1],
                           d["bbox_xyxy"][2], d["bbox_xyxy"][3]] for d in dets], dtype=np.float32)
        labels = [d["class"] for d in dets]
        confs = [d["confidence"] for d in dets]

        overlay = compose_overlay(orig, heat_bgr, boxes, labels, confs, names,
                                  alpha=0.55, out_size=(W, H))

        stem = os.path.splitext(os.path.basename(str(img_path)))[0]
        png_path = os.path.join(out_dir, f"occlusion_{stem}.png")
        npy_path = os.path.join(out_dir, f"occlusion_{stem}.npy")
        cv2.imwrite(png_path, overlay)
        np.save(npy_path, attribution)

        artifacts.append(png_path)
        records.append({
            "image": str(img_path),
            "grid": list(grid),
            "png": os.path.basename(png_path),
            "npy": os.path.basename(npy_path),
            "class_name": names[dets[0]["class"]] if dets and dets[0]["class"] < len(names) else None,
            "ref_classes": sorted({d["class"] for d in dets}),
        })

    if not records:
        raise ValueError("Occlusion no encontró detecciones en las imágenes dadas.")

    write_json(os.path.join(out_dir, "occlusion.json"),
               {"grid": list(grid), "fill": fill, "instances": records})
    return {"saved": True, "artifacts": artifacts, "instances": records}