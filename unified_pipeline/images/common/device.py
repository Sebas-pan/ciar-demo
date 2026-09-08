"""Shared device resolution for images (detection + classification)."""

def resolve_device(cfg):
    import torch
    device = cfg["model"].get("device", "auto")
    if isinstance(device, str):
        device = device.strip().lower()
    if device in ("auto", "gpu", "best"):
        return 0 if torch.cuda.is_available() else "cpu"
    if isinstance(device, str) and device.isdigit():
        return int(device) if torch.cuda.is_available() else "cpu"
    return device
