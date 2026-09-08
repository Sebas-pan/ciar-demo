"""Simple occlusion sensitivity for classification (reuses classification predict)."""
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms


def run_occlusion_classification(cfg, model, image_paths, out_dir, names=None, grid=(8,8)):
    os.makedirs(out_dir, exist_ok=True)
    device_str = cfg["model"].get("device", "cpu")
    if isinstance(device_str, int):
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    if isinstance(device_str, str) and device_str.isdigit():
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str if isinstance(device_str, str) else "cpu")
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    instances=[]
    for img_path in image_paths:
        try:
            pil = Image.open(str(img_path)).convert("RGB")
            orig = cv2.imread(str(img_path))
            if orig is None:
                orig = np.array(pil)[:,:,::-1]
            tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                base_out = torch.softmax(model(tensor), dim=1)[0]
                base_pred = int(base_out.argmax().item())
                base_conf = float(base_out[base_pred].item())
            h, w = orig.shape[:2]
            gh, gw = grid
            heat = np.zeros((gh, gw), dtype=float)
            for i in range(gh):
                for j in range(gw):
                    occluded = orig.copy()
                    y1, y2 = int(i*h/gh), int((i+1)*h/gh)
                    x1, x2 = int(j*w/gw), int((j+1)*w/gw)
                    cv2.rectangle(occluded, (x1,y1), (x2,y2), (128,128,128), -1)
                    pil_occ = Image.fromarray(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB))
                    t = transform(pil_occ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = torch.softmax(model(t), dim=1)[0]
                        conf = float(out[base_pred].item())
                    heat[i,j] = base_conf - conf
            # normalize and overlay
            heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            heat_up = cv2.resize(heat_norm, (w,h))
            heat_color = cv2.applyColorMap(np.uint8(255*heat_up), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.6, heat_color, 0.4, 0)
            out_path = os.path.join(out_dir, f"occlusion_{os.path.basename(str(img_path))}")
            cv2.imwrite(out_path, overlay)
            instances.append({"image": str(img_path), "pred": base_pred, "base_conf": base_conf, "out": out_path})
        except Exception as exc:
            instances.append({"image": str(img_path), "error": str(exc)})
    return {"instances": instances}
