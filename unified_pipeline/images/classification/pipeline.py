"""Classification pipeline for VGG16 / ResNet50 on PetImages."""
import datetime
import os
import shutil

from ...data.image_classification_loader import build_classification_spec, load_classification_dataset
from ...io import write_json
from ..common.device import resolve_device
from .models import create_vgg16, create_resnet50, create_classifier, identify_model_type
from .trainer import run_training_classification
from .evaluate import run_evaluation_classification
from .persist import build_classification_metadata
from .eda import run_eda_classification
from .predict import run_predictions_classification


def run_classification(cfg):
    dataset_dir = cfg["data"]["source"]
    model_cfg = cfg["model"]
    variant = model_cfg.get("variant", "vgg16")
    model_type = model_cfg.get("type", "vgg16")

    # split config for flat datasets
    split_cfg = model_cfg.get("train", {}).get("split", [0.7,0.15,0.15])
    data_spec = build_classification_spec(dataset_dir, split_ratios=tuple(split_cfg))
    bundles = load_classification_dataset(dataset_dir, split_ratios=tuple(split_cfg), seed=int(model_cfg.get("train",{}).get("seed",42)))

    # names/nc from bundles
    names = bundles[0].names
    nc = bundles[0].nc
    data_spec["names"] = names
    data_spec["nc"] = nc

    outputs = cfg["output"]
    run_dir = os.path.join(outputs["dir"], "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # EDA
    eda_dir = os.path.join(run_dir, "eda_image")
    eda_summary = run_eda_classification(bundles, eda_dir)

    # Model
    model = create_classifier(cfg, num_classes=nc)
    tag = str(model_type or variant or "classifier").upper()
    raw_device = model_cfg.get("device", "auto")
    device = resolve_device(cfg)
    # Banner: show if CUDA or CPU before training
    try:
        import torch
        if isinstance(device, int):
            _dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            _dev_obj = torch.device(_dev)
        else:
            _dev_obj = torch.device(str(device))
        # Normalize device for summary (cuda:0 vs 0)
        device = _dev_obj
        if _dev_obj.type == "cuda" and torch.cuda.is_available():
            print(f"[{tag}] Using device: {_dev_obj} ({torch.cuda.get_device_name(_dev_obj)}) | cfg device: {raw_device}")
        else:
            print(f"[{tag}] Using device: {_dev_obj} (CPU) | cfg device: {raw_device} | CUDA available: {torch.cuda.is_available()}")
    except Exception as _e:
        print(f"[{tag}] Using device: {device} | cfg device: {raw_device} (banner error: {_e})")
    train_cfg = model_cfg.get("train", {})

    t0 = datetime.datetime.now()
    best_src = run_training_classification(cfg, model, bundles, run_dir)
    train_minutes = round((datetime.datetime.now()-t0).total_seconds()/60,2)

    # Evaluate on test
    metrics, _ = run_evaluation_classification(cfg, model, bundles, os.path.join(run_dir, "val"))

    # Persist
    weights_out = os.path.join(run_dir, "model", "best.pt")
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
    if os.path.exists(best_src):
        shutil.copy(best_src, weights_out)
    else:
        # fallback save state_dict directly
        import torch
        torch.save({"state_dict": model.state_dict()}, weights_out)

    meta = build_classification_metadata(
        dataset_dir=dataset_dir, data_spec=data_spec, variant=variant, weights="best.pt",
        metrics=metrics, train_cfg=train_cfg, device=device, source=dataset_dir, model_type=model_type
    )
    meta["model_path"] = "best.pt"
    meta["identify"] = identify_model_type(model_type)
    meta["run_seconds_train"] = round(train_minutes*60,1)
    write_json(os.path.join(run_dir, "model", "metadata.json"), meta)

    # Predictions demo on test
    test_bundle = next((b for b in bundles if b.split=="test"), bundles[-1])
    demo_count = int(train_cfg.get("predict_demo", 4))
    demo_images = [str(p) for p in test_bundle.images[:demo_count]]
    pred_dir = os.path.join(run_dir, "predictions")
    predictions, _ = run_predictions_classification(cfg, model, demo_images, pred_dir, names=names)

    # XAI — classification Grad-CAM + occlusion
    xai_dir = os.path.join(run_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    xai_cfg = cfg.get("xai", {})
    xai_methods = xai_cfg.get("methods", ["gradcam"])
    xai = {"gradcam": None, "occlusion": None}

    if "gradcam" in xai_methods:
        try:
            from ...xai.gradcam_classification import run_gradcam_classification
            out = run_gradcam_classification(cfg, model, demo_images, xai_dir, names=names)
            xai["gradcam"] = out
        except Exception as exc:
            xai["gradcam"] = {"error": str(exc)}

    if "occlusion" in xai_methods:
        try:
            # reuse occlusion_image but with classification predict wrapper
            from ...xai.occlusion_classification import run_occlusion_classification
            out = run_occlusion_classification(cfg, model, demo_images, xai_dir, names=names)
            xai["occlusion"] = out
        except Exception as exc:
            # fallback: mark skipped if not implemented
            xai["occlusion"] = {"error": str(exc)}

    # Report
    if outputs.get("save_report", True):
        _build_classification_report(run_dir, metrics, names, variant, device, train_minutes, predictions, xai, xai_dir, eda_summary, meta, narrative=bool(xai_cfg.get("narrative", True)), model_label=f"{model_type} ({variant})")

    # Normalize split_shapes to train/val/test (valid -> val)
    raw_splits = {b.split: len(b.images) for b in bundles}
    split_shapes = {
        "train": raw_splits.get("train", 0),
        "val": raw_splits.get("valid", raw_splits.get("val", 0)),
        "test": raw_splits.get("test", 0),
    }
    return {
        "data_type": "image",
        "task": "classification",
        "model": f"{model_type} ({variant})",
        "target": "/".join(names),
        "source": dataset_dir,
        "split_shapes": split_shapes,
        "metrics": metrics,
        "xai": xai,
        "run_dir": run_dir,
        "validation": None,
        "warnings": [],
        "device": str(device),
    }


def _build_classification_report(run_dir, metrics, names, variant, device, train_minutes, predictions, xai, xai_dir, eda, model_meta, narrative=True, model_label=None):
    from ...xai.narrator import narrate_image as det_narrate_image
    # For classification, craft simple markdown
    try:
        import torch
        if isinstance(device, int):
            dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                dev_str += f" ({torch.cuda.get_device_name(torch.device(dev_str))})"
        else:
            dev_str = str(device)
            if str(device) != "cpu" and torch.cuda.is_available() and "cuda" in str(device):
                try:
                    dev_str += f" ({torch.cuda.get_device_name(torch.device(str(device)))})"
                except Exception:
                    pass
    except Exception:
        dev_str = str(device)
    lines = [f"# Image Classification Report — {model_label or f'VGG16 ({variant})'}", f"*Device: {dev_str}*", ""]
    if narrative:
        lines.append("## What the model did")
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        if acc is not None:
            lines.append(f"Accuracy **{acc:.3f}** ({acc*100:.1f}%), F1 **{f1:.3f}** on {metrics.get('n_test', '?')} test images ({', '.join(names)}).")
        lines.append(f"Trained {train_minutes} min on device **{dev_str}**.")
        lines.append("")
    lines.append("## Metrics")
    for k in ["accuracy","precision","recall","f1"]:
        if k in metrics:
            lines.append(f"- **{k}**: {metrics[k]:.4f}")
    lines.append("")
    if "per_class" in metrics:
        lines.append("### Per-class")
        for cls, vals in metrics["per_class"].items():
            lines.append(f"- **{cls}**: P {vals['precision']:.3f} R {vals['recall']:.3f} F1 {vals['f1']:.3f} (n={vals['support']})")
        lines.append("")
    lines.append("## Predictions (demo)")
    for p in predictions:
        lines.append(f"- `{os.path.basename(p['image'])}` → **{p['pred_name']}** ({p['confidence']:.2%})")
    lines.append("")
    lines.append("## XAI")
    for k, v in xai.items():
        if v and "error" in v:
            lines.append(f"- **{k}**: ERROR {v['error']}")
        elif v:
            lines.append(f"- **{k}**: OK")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    write_json(os.path.join(xai_dir, "xai_summary.json"), xai)
