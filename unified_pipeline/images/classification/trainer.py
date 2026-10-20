"""Training loop for VGG16 classification (no augmentation per spec)."""
import os

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from ...data.image_classification_loader import load_classification_dataset
from ..common.transforms import get_transforms
from ..common.device import resolve_device


class _ImageDataset(Dataset):
    def __init__(self, bundle, transform):
        self.images = bundle.images
        self.labels = bundle.labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = str(self.images[idx])
        label = int(self.labels[idx])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # return black image on error
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


def run_training_classification(cfg, model, bundles, run_dir):
    """Train VGG16/ResNet50 classifier and save best checkpoint to run_dir/train/weights/best.pt"""
    model_tag = str(cfg.get("model", {}).get("type", "classifier")).upper()
    train_cfg = cfg["model"].get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    batch = int(train_cfg.get("batch", 32))
    workers = int(train_cfg.get("workers", 4))
    lr = float(train_cfg.get("lr", 0.0001))
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    patience = int(train_cfg.get("patience", 5))
    seed = int(train_cfg.get("seed", 42))
    imgsz = int(train_cfg.get("imgsz", 224))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    # No augmentation per user spec
    augment = bool(train_cfg.get("augment", False))

    raw_device = resolve_device(cfg)
    device = raw_device
    # resolve_device may return 0 for gpu; convert to torch.device
    if isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device if isinstance(device, str) else "cpu")

    # Banner: show if CUDA or CPU
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(device)
            mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"[{model_tag}] Using device: {device} ({dev_name}, {mem_gb:.1f}GB) | CUDA available: True")
        else:
            print(f"[{model_tag}] Using device: {device} (CPU) | CUDA available: {torch.cuda.is_available()} | cfg device: {raw_device}")
    except Exception:
        print(f"[{model_tag}] Using device: {device} | CUDA available: {torch.cuda.is_available()}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Bundles
    train_bundle = next((b for b in bundles if b.split == "train"), bundles[0])
    val_bundle = next((b for b in bundles if b.split in ("valid","val")), None)
    if val_bundle is None and len(bundles) > 1:
        val_bundle = bundles[1]

    train_ds = _ImageDataset(train_bundle, get_transforms(imgsz, augment=augment, is_train=True))
    val_ds = _ImageDataset(val_bundle, get_transforms(imgsz, augment=False, is_train=False)) if val_bundle and len(val_bundle.images)>0 else None

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True) if val_ds else None

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_state = None
    no_improve = 0
    epoch_losses = []

    # tqdm: outer epoch bar, disable if not a tty
    use_tqdm = sys.stdout.isatty()
    epoch_pbar = tqdm(range(epochs), desc=f"{model_tag} epochs", unit="epoch", ncols=100, disable=not use_tqdm)

    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        batches = 0
        # inner batch tqdm (leave=False, only when tty)
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False, ncols=100, disable=not use_tqdm)
        for imgs, labels in batch_iter:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            if use_tqdm:
                batch_iter.set_postfix(loss=f"{running_loss/batches:.4f}")

        epoch_loss = running_loss / max(batches, 1)
        epoch_losses.append(epoch_loss)

        # validation
        acc = None
        if val_loader is not None:
            acc = _evaluate_accuracy(model, val_loader, device, use_tqdm=use_tqdm)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if use_tqdm:
                epoch_pbar.set_postfix(best_acc=f"{best_acc:.4f}", val_acc=f"{acc:.4f}", no_improve=no_improve, loss=f"{running_loss/max(batches,1):.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss {running_loss/max(batches,1):.4f} - val_acc {acc:.4f} - best {best_acc:.4f}")
            if patience and no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience {patience})")
                break
        else:
            if use_tqdm:
                epoch_pbar.set_postfix(loss=f"{running_loss/max(batches,1):.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss {running_loss/max(batches,1):.4f}")

    # If no val, best is last
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        train_dir = os.path.join(run_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="tab:red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(train_dir, "loss_curve.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Save
    out_dir = os.path.join(run_dir, "train", "weights")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    torch.save({"state_dict": best_state, "cfg": cfg["model"]}, best_path)
    # also load best into model for immediate evaluation
    model.load_state_dict(best_state)
    return best_path


def _evaluate_accuracy(model, loader, device, use_tqdm=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc="Val", leave=False, ncols=80, disable=not use_tqdm) if use_tqdm else loader
        for imgs, labels in iterator:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total else 0.0
