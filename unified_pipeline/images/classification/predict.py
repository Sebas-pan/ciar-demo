"""Predict for VGG16 classification."""
import os
import json
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..common.transforms import get_transforms
from ..common.device import resolve_device


def predict_image_classification(cfg, model, image_path):
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 224))
    device = resolve_device(cfg)
    if isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device if isinstance(device, str) else "cpu")
    transform = get_transforms(imgsz, augment=False, is_train=False)
    img = Image.open(str(image_path)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    return prob


def run_predictions_classification(cfg, model, images, out_dir, names):
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for img_path in images:
        prob = predict_image_classification(cfg, model, img_path)
        pred_idx = int(prob.argmax())
        summaries.append({
            "image": str(img_path),
            "pred_class": pred_idx,
            "pred_name": names[pred_idx] if pred_idx < len(names) else str(pred_idx),
            "confidence": float(prob[pred_idx]),
            "probs": {names[i]: float(prob[i]) for i in range(len(names))},
        })
    json_path = os.path.join(out_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    return summaries, json_path
