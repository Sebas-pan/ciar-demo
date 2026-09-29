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
    default_layer = _default_gradcam_layer(model)
    layer_name = (cfg.get("xai",{}).get("gradcam",{}).get("layer") or default_layer)
    # Hook
    features = {}
    gradients = {}
    def forward_hook(m, inp, out):
        features["value"] = out
    def backward_hook(m, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Navigate to layer across architectures (VGG/ResNet/EfficientNet/
    # MobileNet/DenseNet/ViT/Swin/ConvNeXt + explicit dotted-path override).
    try:
        target_layer = _resolve_gradcam_layer(model, layer_name)
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

            grad = gradients["value"].detach()
            fmap = features["value"].detach()
            cam = _build_cam(grad, fmap)
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


def _build_cam(grad, fmap):
    """Build 2D CAM from CNN (B,C,H,W) or transformer (B,T,C) tensors."""
    import math
    if grad.ndim == 4 and fmap.ndim == 4:
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        return (weights * fmap).sum(dim=1).squeeze(0).cpu().numpy()
    if grad.ndim == 3 and fmap.ndim == 3:
        # ViT/Swin tokens: drop CLS token, reshape patch grid to 2D.
        weights = grad.mean(dim=1, keepdim=True)  # (1, 1, C)
        tok = (weights * fmap).sum(dim=-1).squeeze(0)  # (T,)
        toks = tok[1:] if tok.numel() > 1 else tok
        side = int(math.sqrt(float(toks.numel())))
        if side * side != toks.numel():
            side = int(math.ceil(math.sqrt(float(toks.numel()))))
            pad = side * side - toks.numel()
            if pad > 0:
                import torch as _t
                toks = _t.cat([toks, toks[-1:].repeat(pad)])
        return toks.reshape(side, side).cpu().numpy()
    # Fallback: squeeze whatever we got to 2D
    arr = fmap.squeeze(0).cpu().numpy()
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    return arr


def _default_gradcam_layer(model):
    # ResNet family
    if hasattr(model, "layer4"):
        return "layer4"
    cls = type(model).__name__.lower()
    # DenseNet: last dense block
    if hasattr(model, "features") and hasattr(getattr(model, "features", None), "denseblock4"):
        return "features.denseblock4"
    # Vision Transformer: last encoder MLP (diffuse — occlusion is primary)
    if hasattr(model, "encoder"):
        return "encoder.layers.encoder_layer_11"
    # EfficientNet / MobileNet / ConvNeXt / Swin: last feature block
    if hasattr(model, "features"):
        try:
            return "features.-1"
        except Exception:
            pass
    if "vgg" in cls:
        return "features.28"
    return "features.-1"


def _resolve_gradcam_layer(model, layer_name):
    name = str(layer_name or "")
    # 1. Explicit dotted path, e.g. "features.8", "features.denseblock4",
    #    "encoder.layers.encoder_layer_11", "layer4".
    if "." in name or name in ("features", "layer4", "encoder"):
        try:
            return _walk_attr(model, name)
        except Exception:
            pass
    # 2. Architecture fallbacks
    if hasattr(model, "layer4") and "layer4" in name:
        try:
            return model.layer4[-1]
        except Exception:
            return model.layer4
    if hasattr(model, "features"):
        feats = model.features
        # DenseNet denseblock
        if hasattr(feats, "denseblock4") and "dense" in name:
            block = feats.denseblock4
            try:
                return block[-1]
            except Exception:
                return block
        # Numeric index: "features.28" / "features.-1" / "28"
        tail = name.split(".")[-1]
        try:
            idx = int(tail)
            return feats[idx]
        except (ValueError, IndexError, TypeError):
            pass
        # Last block fallback for EfficientNet/MobileNet/Swin/ConvNeXt
        try:
            return feats[-1]
        except Exception:
            pass
    if hasattr(model, "encoder"):
        try:
            layers = model.encoder.layers
            last = layers[-1]
            # Prefer the MLP sub-block output for CAM
            if hasattr(last, "mlp"):
                return last.mlp
            return last
        except Exception:
            pass
    return list(model.children())[-2]


def _walk_attr(root, dotted):
    cur = root
    for part in str(dotted).split("."):
        if part == "-1" or part == "[-1]":
            cur = cur[-1]
            continue
        if part.endswith("]") and "[" in part:
            attr, idx_txt = part[:-1].split("[", 1)
            if attr:
                cur = getattr(cur, attr)
            cur = cur[int(idx_txt)]
            continue
        try:
            cur = getattr(cur, part)
            continue
        except AttributeError:
            pass
        # numeric index into Sequential/ModuleList
        cur = cur[int(part)]
    # Unwrap trailing Sequential single block (e.g. ResNet layer4[-1])
    return cur
