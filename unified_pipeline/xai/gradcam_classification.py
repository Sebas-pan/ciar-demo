"""Grad-CAM for image classification (VGG16 / ResNet50)."""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

def run_gradcam_classification(cfg, model, image_paths, out_dir, names=None):
    os.makedirs(out_dir, exist_ok=True)
    device_str = cfg["model"].get("device", "cpu")
    if isinstance(device_str, int):
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    if isinstance(device_str, str) and device_str.isdigit():
        device_str = f"cuda:{device_str}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str if isinstance(device_str, str) else "cpu")
    # Resolve target layer (default depends on architecture)
    is_resnet = hasattr(model, "layer4")
    default_layer = "layer4" if is_resnet else "features.28"
    layer_name = (cfg.get("xai",{}).get("gradcam",{}).get("layer") or default_layer)
    # Hook
    features = {}
    gradients = {}
    def forward_hook(m, inp, out):
        features["value"] = out
    def backward_hook(m, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Navigate to layer: VGG `features.28`, ResNet `layer4[-1]`
    try:
        if hasattr(model, "layer4") and ("layer4" in str(layer_name)):
            # last bottleneck block of layer4 (Sequential)
            target_layer = model.layer4[-1]
        elif hasattr(model, "features"):
            try:
                idx = int(str(layer_name).split(".")[-1])
            except ValueError:
                idx = 28
            target_layer = model.features[idx]
        else:
            target_layer = list(model.children())[-2]
    except Exception:
        target_layer = list(model.modules())[-2]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook) if hasattr(target_layer, "register_full_backward_hook") else target_layer.register_backward_hook(backward_hook)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    instances = []
    for img_path in image_paths:
        try:
            pil = Image.open(str(img_path)).convert("RGB")
            orig = cv2.imread(str(img_path))
            if orig is None:
                orig = np.array(pil)[:,:,::-1]
            tensor = transform(pil).unsqueeze(0).to(device)
            model = model.to(device)
            model.eval()
            model.zero_grad()
            out = model(tensor)
            pred_idx = int(out.argmax(dim=1).item())
            score = out[0, pred_idx]
            score.backward()

            grad = gradients["value"].detach()  # (1, C, H, W)
            fmap = features["value"].detach()
            weights = grad.mean(dim=(2,3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * fmap).sum(dim=1).squeeze(0).cpu().numpy()
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
            out_path = os.path.join(out_dir, f"gradcam_{os.path.basename(str(img_path))}")
            cv2.imwrite(out_path, overlay)
            instances.append({"image": str(img_path), "pred": pred_idx, "pred_name": names[pred_idx] if names and pred_idx < len(names) else str(pred_idx), "out": out_path})
        except Exception as exc:
            instances.append({"image": str(img_path), "error": str(exc)})

    h1.remove()
    h2.remove()
    return {"instances": instances, "layer": layer_name}
