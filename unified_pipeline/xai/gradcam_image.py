import os

import cv2
import numpy as np
import torch

from ..images.predict import predict_image, summarize_result
from ..images.trainer import resolve_device
from ..io import write_json
from .overlay import letterbox, compose_overlay, rescale_boxes


def run_gradcam(cfg, model, image_paths, out_dir,
                mode="max_detection", target_class=None,
                min_conf=0.25, imgsz=None):
    """Grad-CAM adaptado a detección YOLO11.

    Captura las activaciones de entrada de la rama de clases (head.cv3) con un
    forward hook bajo inference_mode y construye el mapa ponderando cada canal
    por la confianza de la clase objetivo (sigmoid de sus logits), en lugar de
    retropropagar una pérdida. Ver _forward_gradcam.

    mode:
      - 'max_detection': usa la clase del bbox de mayor confianza (si hay).
      - 'class': usa target_class explícito.
      - sin detecciones: la clase se infiere de los logits globales.
    Retorna dict resumen.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_cfg = cfg["model"].get("train", {})
    raw_module = model.model
    device = resolve_device(cfg)
    raw_module.to(device)
    if imgsz is None:
        imgsz = int(train_cfg.get("imgsz", 640))
    names = model.names

    det = _det_heads(raw_module)
    raw_module.eval()

    artifacts = []
    cam_records = []

    for img_path in image_paths:
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue

        dets = []
        try:
            result = predict_image(cfg, model, str(img_path), conf=min_conf, iou=0.5)
            dets = summarize_result(result)["detections"]
        except Exception:
            pass

        tensor, canvas, ratio, dw, dh = letterbox(orig, imgsz, stride=32)
        H, W = canvas.shape[:2]

        # clase objetivo
        if mode == "class" and target_class is not None:
            target_cls = int(target_class)
        elif dets:
            target_cls = max(dets, key=lambda d: d["confidence"])["class"]
        else:
            target_cls = None  # inferir de los logits

        out = _forward_gradcam(raw_module, det, tensor, H, W, device, target_cls)
        if out is None:
            continue
        full_cam, chosen_cls = out
        target_cls = chosen_cls

        boxes = np.array([[d["bbox_xyxy"][0], d["bbox_xyxy"][1],
                           d["bbox_xyxy"][2], d["bbox_xyxy"][3]] for d in dets],
                         dtype=np.float32) if dets else np.zeros((0, 4))
        labels = [d["class"] for d in dets]
        confs = [d["confidence"] for d in dets]
        boxes_scaled = rescale_boxes(boxes, ratio, dw, dh)

        heat_bgr = _heat_to_bgr(full_cam, (H, W))
        overlay_bgr = compose_overlay(
            canvas, heat_bgr, boxes_scaled, labels, confs, names,
            alpha=0.55, out_size=(W, H))

        stem = os.path.splitext(os.path.basename(str(img_path)))[0]
        png_path = os.path.join(out_dir, f"gradcam_{stem}.png")
        npy_path = os.path.join(out_dir, f"gradcam_{stem}.npy")
        cv2.imwrite(png_path, overlay_bgr)
        np.save(npy_path, full_cam)

        artifacts.append(png_path)
        cam_records.append({
            "image": str(img_path),
            "target_class": int(target_cls),
            "class_name": names[int(target_cls)],
            "detections": len(dets),
            "png": os.path.basename(png_path),
            "npy": os.path.basename(npy_path),
        })

    if not cam_records:
        raise ValueError("Grad-CAM no generó ningún mapa (revisa las imágenes).")

    write_json(os.path.join(out_dir, "gradcam.json"),
               {"mode": mode, "instances": cam_records})
    return {"saved": True, "artifacts": artifacts, "instances": cam_records}


def _det_heads(raw_module):
    det = raw_module.model[-1]
    if hasattr(det, "cv3"):
        return det
    raise ValueError("No Detect head (cv3) found. Ensure model is a YOLO detection model.")


def _forward_gradcam(raw_module, det, tensor, H, W, device, target_cls=None):
    """Mapa de calor por activación ponderada con la confianza de la clase objetivo.

    En lugar de retropropagar una BCE por píxel (cuyo gradiente es inverso a la
    confianza de la clase — el fondo queda con mayor magnitud — y con abs() la
    atribución se invierte), cada canal se pondera por cuánto contribuye a la
    presencia de la clase objetivo y el mapa se activa con ReLU y se enmascara
    con la confianza:

        score    = sigmoid(logits_clase)                    # (1,1,h,w)
        weights  = mean_spatial(score * activaciones)       # (1,C,1,1)
        cam      = ReLU(sum_canales(weights * activaciones)) * score

    El fondo (score≈0) queda en azul y el objeto (score≈1) en rojo.
    Retorna (cam normalizada, clase elegida) o None.
    """
    captured = {}

    def hook(module, args, out):
        captured["features"] = list(args[0])

    handle = det.register_forward_hook(hook)
    with torch.inference_mode():
        raw_module(tensor.to(device))
    handle.remove()

    features = captured.get("features")
    if not features:
        return None

    with torch.inference_mode():
        feats_norm = []
        for feat in features:
            f = torch.empty_like(feat, dtype=feat.dtype, device=device,
                                 memory_format=torch.contiguous_format)
            f.copy_(feat.to(device))
            feats_norm.append(f)

        act_list = [det.cv3[i](feat) for i, feat in enumerate(feats_norm)]

        if target_cls is None:
            nc = act_list[0].shape[1]
            scores = torch.zeros((nc,), device=device)
            for act in act_list:
                scores = scores + torch.sigmoid(act).sum(dim=(0, 2, 3))
            target_cls = int(torch.argmax(scores).item())

        full_cam = None
        for act in act_list:
            cls_logits = act[:, target_cls:target_cls + 1, :, :]
            score = torch.sigmoid(cls_logits)
            weights = (score * act).mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True).relu() * score
            cam = torch.nn.functional.interpolate(
                cam, size=(H, W), mode="bilinear")[0, 0]
            cam = cam.detach().cpu().numpy()
            full_cam = cam if full_cam is None else full_cam + cam

    if full_cam is None or full_cam.max() <= 0:
        return None
    return full_cam / full_cam.max(), target_cls


def _heat_to_bgr(cam1d, shape):
    heat = (np.clip(cam1d, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)