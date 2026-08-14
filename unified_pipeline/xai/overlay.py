import cv2
import numpy as np


def letterbox(img_bgr, new_size, stride=32):
    """Resize con padding (letterbox) como hace ultralytics.

    Retorna: (tensor CHW RGB normalizado [0,1], canvas RGB uint8, ratio, pad_w, pad_h).
    """
    import torch

    h, w = img_bgr.shape[:2]
    r = min(new_size / h, new_size / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    # múltiplo de stride por compatibilidad
    new_w = max(int(round(new_w / stride)) * stride, stride)
    new_h = max(int(round(new_h / stride)) * stride, stride)
    dw = (new_size - new_w) // 2
    dh = (new_size - new_h) // 2

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    tensor = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0)
    return tensor, canvas, r, dw, dh


def rescale_boxes(boxes_xyxy, ratio, dw, dh):
    """Conversión de cajas en coordenadas originales a coordenadas del canvas letterbox."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy * 0
    return boxes_xyxy * ratio + [dw, dh, dw, dh]


def heatmap_to_bgr(heatmap, canvas_size, alpha=0.65, colormap=cv2.COLORMAP_JET):
    """heatmap: array (H, W) en [0,1]. Devuelve imagen BGR del tamaño canvas_size."""
    heat = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, colormap)
    canvas = np.full((canvas_size, canvas_size, 3), 114, dtype=np.uint8)
    canvas = colored
    return canvas, colored


def compose_overlay(base_bgr, heat_bgr, boxes_xyxy, labels, confs, names,
                    alpha=0.55, out_size=None):
    """Superpone heatmap (cámara de calor) + cajas + etiquetas sobre la base."""
    heat_f = heat_bgr.astype(np.float32)
    base_f = base_bgr.astype(np.float32)
    overlay = cv2.addWeighted(heat_f, alpha, base_f, 1 - alpha, 0).astype(np.uint8)

    for box, label, conf in zip(boxes_xyxy, labels, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0) if names[label] == "Cat" else (255, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = "{} {:.2f}".format(names[label], float(conf))
        cv2.putText(overlay, text, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if out_size is not None:
        overlay = cv2.resize(overlay, out_size, interpolation=cv2.INTER_LINEAR)
    return overlay


def optical_attention(canvas_rgb):
    """Placeholder para window re-export; canvas_rgb es BGR en realidad."""
    return canvas_rgb