"""Evaluation for VGG16 classification."""
import json
import os

import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from ..common.device import resolve_device
from ..common.transforms import get_transforms
from .trainer import _ImageDataset


def run_evaluation_classification(cfg, model, bundles, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_cfg = cfg["model"].get("train", {})
    imgsz = int(train_cfg.get("imgsz", 224))
    batch = int(train_cfg.get("batch", 32))
    workers = int(train_cfg.get("workers", 4))
    device = resolve_device(cfg)
    if isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device if isinstance(device, str) else "cpu")

    test_bundle = next((b for b in bundles if b.split == "test"), bundles[-1])
    test_ds = _ImageDataset(test_bundle, get_transforms(imgsz, augment=False, is_train=False))
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers)

    model_tag = str(cfg.get("model", {}).get("type", "classifier")).upper()
    model = model.to(device)
    # Device banner for eval
    try:
        dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else "CPU"
        print(f"[{model_tag}][Eval] Using device: {device} ({dev_name})")
    except Exception:
        print(f"[{model_tag}][Eval] Using device: {device}")
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []
    use_tqdm = sys.stdout.isatty()
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing", unit="batch", ncols=80, disable=not use_tqdm):
            imgs = imgs.to(device)
            out = model(imgs)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(labels.numpy().tolist())
            y_proba.extend(prob.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=test_bundle.names, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class": {},
        "n_test": int(len(y_true)),
    }
    # per-class derived from report
    for name in test_bundle.names:
        if name in report:
            metrics["per_class"][name] = {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # confusion matrix plot (scales with class count so 45+ classes stay legible)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        nc = len(test_bundle.names)
        figsize = (min(20, max(8, int(0.45 * nc))), min(16, max(6, int(0.45 * nc))))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(test_bundle.names, rotation=90, fontsize=6 if nc > 10 else 8)
        ax.set_yticklabels(test_bundle.names, fontsize=6 if nc > 10 else 8)
        if nc <= 10:
            vmax = max(max(row) for row in cm)
            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                            color="white" if cm[i][j] > vmax / 2 else "black", fontsize=7)
        ax.set_xlabel("Pred", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title("Confusion Matrix", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    return metrics, os.path.join(out_dir, "metrics.json")
