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
from ...xai.llm_narrator import narrate_image_classification as llm_narrate_image
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
    use_llm = bool(xai_cfg.get("use_llm", False))
    if use_llm:
        _llm_narrate_image_fn = lambda m, n, p, x: llm_narrate_image(m, n, p, x)
    else:
        _llm_narrate_image_fn = None
    if outputs.get("save_report", True):
        _build_classification_report(run_dir, metrics, names, variant, device, train_minutes, predictions, xai, xai_dir, eda_summary, meta, narrative=bool(xai_cfg.get("narrative", True)), model_label=f"{model_type} ({variant})", narrate_image_func=_llm_narrate_image_fn)

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


def _build_classification_report(run_dir, metrics, names, variant, device, train_minutes, predictions, xai, xai_dir, eda, model_meta, narrative=True, model_label=None, narrate_image_func=None):
    from ...xai.narrator import narrate_classification
    # Para clasificación, markdown en español a la par del reporte de detección
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
    label = model_label or "Clasificador ({})".format(variant)
    lines = [f"# Reporte — Clasificación de imágenes {label}", ""]
    lines.append("## Entrenamiento")
    lines.append("- Modelo: `{}` (variante `{}`)".format(
        (model_meta or {}).get("model_type", variant), variant))
    lines.append("- Dispositivo: `{}` · tiempo: {} min".format(dev_str, train_minutes))
    lines.append("- Clases: {}".format(", ".join(names)))
    lines.append("- EDA: `class_distribution.png`, `sample_grid.jpg` y "
                 "`eda_classification.json` (distribución por split, stats de tamaño)")
    if isinstance(eda, dict) and eda.get("splits"):
        splits_txt = ", ".join(
            "{}: {}".format(k, v) for k, v in eda["splits"].items())
        lines.append("- Splits (imágenes): {}".format(splits_txt))
    lines.append("- Curva de pérdida: `train/loss_curve.png`")
    lines.append("")

    lines.append("## Métricas (test)")
    acc = metrics.get("accuracy")
    if acc is not None:
        lines.append("- Exactitud (accuracy): **{:.4f}**".format(float(acc)))
    for k in ["precision", "recall", "f1"]:
        if k in metrics:
            lines.append("- {}: {:.4f}".format(
                {"precision": "Precisión (media)", "recall": "Recall (media)", "f1": "F1 (media)"}[k],
                float(metrics[k])))
    if metrics.get("n_test") is not None:
        lines.append("- Imágenes de test: {}".format(metrics["n_test"]))
    for cls, vals in (metrics.get("per_class") or {}).items():
        lines.append("    - {} · P={:.4f} · R={:.4f} · F1={:.4f} (n={})".format(
            cls,
            float(vals.get("precision", 0.0)),
            float(vals.get("recall", 0.0)),
            float(vals.get("f1", 0.0)),
            vals.get("support", "?"),
        ))
    cm = metrics.get("confusion_matrix")
    lines.append("")
    lines.append("## Matriz de confusión (test)")
    if isinstance(cm, list) and cm and all(isinstance(r, list) for r in cm):
        n_cm = len(cm)
        lines.append("![Matriz de confusión](val/confusion_matrix.png)")
        lines.append("")
        # top confused pairs (off-diagonal max)
        try:
            import numpy as _np
            cm_arr = _np.array(cm, dtype=float)
            _np.fill_diagonal(cm_arr, -1.0)
            flat = cm_arr.reshape(-1)
            top_idx = _np.argsort(flat)[-5:][::-1]
            names_list = list(names)
            pairs = []
            for idx in top_idx:
                i = int(idx // n_cm)
                j = int(idx % n_cm)
                v = float(cm_arr[i, j])
                if v < 0:
                    continue
                pairs.append("{}→{}: {}".format(
                    names_list[i] if i < len(names_list) else i,
                    names_list[j] if j < len(names_list) else j,
                    int(v)))
            if pairs:
                lines.append("Pares más confundidos: " + "; ".join(pairs))
        except Exception:
            pass
    else:
        lines.append("No disponible.")
    lines.append("")

    if narrative:
        narrate = narrate_image_func or narrate_classification
        text = narrate(metrics, names, predictions, xai)
        if text:
            lines.append(text)
            lines.append("")

    lines.append("## Predicciones de ejemplo (`predictions/`)")
    for p in predictions or []:
        conf = p.get("confidence")
        conf_txt = "{:.2%}".format(float(conf)) if conf is not None else "n/d"
        lines.append("- `{}` → **{}** ({})".format(
            os.path.basename(p.get("image") or ""), p.get("pred_name"), conf_txt))
    lines.append("")

    lines.append("## Explicabilidad (`xai/`)")
    lines.append("")
    for method in ["gradcam", "occlusion"]:
        entry = (xai or {}).get(method)
        lines.append("### {}".format(method.upper()))
        if entry is None:
            lines.append("No ejecutado.")
        elif isinstance(entry, dict) and "error" in entry:
            lines.append("**ERROR**: {}".format(entry["error"]))
        else:
            lines.append("Artefactos:")
            for inst in (entry or {}).get("instances", []):
                png = os.path.basename(inst.get("out") or "")
                cls_name = inst.get("pred_name") or inst.get("class_name")
                if cls_name is None and isinstance(inst.get("pred"), int):
                    idx = inst["pred"]
                    if 0 <= idx < len(names):
                        cls_name = names[idx]
                if cls_name:
                    lines.append("- `{}` (clase predicha: {})".format(png, cls_name))
                else:
                    lines.append("- `{}`".format(png))
        lines.append("")

    lines.append("## Cómo interpretar los mapas")
    lines.append("""- **Grad-CAM**: mapa de calor sobre la última capa convolucional.
    Las zonas rojas muestran qué regiones **respaldan** la clase predicha.
- **Occlusion Sensitivity**: se ocluye la imagen por celdas y se mide la caída
    de confianza (softmax) de la clase predicha. Celdas rojas = la confianza
    **depende** de esa zona.
""")
    lines.append("Artículo de referencia: métricas en `val/metrics.json`, "
                 "predicciones en `predictions/predictions.json`, "
                 "detalle XAI en `xai/xai_summary.json` y mapas `.jpg`.")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    write_json(os.path.join(xai_dir, "xai_summary.json"), xai)
