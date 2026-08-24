import datetime
import os
import shutil

from ..data.image_loader import build_yolo_data_spec, load_yolo_dataset, write_yolo_data_yaml
from ..io import write_json
from ..xai.gradcam_image import run_gradcam
from ..xai.lime_image import run_lime_image
from ..xai.occlusion_image import run_occlusion
from ..xai.report_image import build_image_report
from .eda import run_eda
from .evaluate import run_evaluation
from .models import create_estimator, identify_model_type as _identify_yolo
from .persist import build_image_metadata
from .predict import run_predictions
from .trainer import resolve_device, run_training

SPLITS = ("train", "valid", "test")


def run_image(cfg):
    dataset_dir = cfg["data"]["source"]
    data_spec = build_yolo_data_spec(dataset_dir)
    bundles = load_yolo_dataset(dataset_dir, splits=SPLITS)

    names = data_spec["names"]
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "yolo11")
    variant = model_cfg.get("variant", "yolov11n")

    outputs = cfg["output"]
    run_dir = os.path.join(
        outputs["dir"], "runs",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    # data.yaml absoluto (ultralytics requiere str, no dict, en esta versión)
    data_yaml = write_yolo_data_yaml(data_spec, os.path.join(run_dir, "data.yaml"))

    # --- EDA imagen ---
    eda_dir = os.path.join(run_dir, "eda_image")
    eda_summary = run_eda(bundles, eda_dir)

    # --- Modelo ---
    model = create_estimator(cfg)
    device = resolve_device(cfg)
    train_cfg = model_cfg.get("train", {})

    t0 = datetime.datetime.now()
    run_training(cfg, model, data_yaml, run_dir)
    train_minutes = round((datetime.datetime.now() - t0).total_seconds() / 60, 2)

    # --- Evaluación en test ---
    metrics, metrics_path = run_evaluation(cfg, model, data_yaml,
                                           os.path.join(run_dir, "val"))

    # --- Persistencia del artefacto ---
    weights_out = os.path.join(run_dir, "model", "best.pt")
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
    src = os.path.join(run_dir, "train", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, weights_out)

    meta = build_image_metadata(
        dataset_dir=dataset_dir,
        data_spec=data_spec,
        variant=variant,
        weights="best.pt",
        metrics=metrics,
        train_cfg=train_cfg,
        device=device,
        source=dataset_dir,
    )
    meta["model_path"] = "best.pt"
    meta["identify"] = _identify_yolo()
    meta["run_seconds_train"] = round(train_minutes * 60, 1)
    write_json(os.path.join(run_dir, "model", "metadata.json"), meta)

    # --- Predicciones demo sobre test ---
    test_bundle = bundles[2] if len(bundles) > 2 else bundles[-1]
    demo_count = int(train_cfg.get("predict_demo", 4))
    demo_images = [str(p) for p in test_bundle.images[:demo_count]]
    pred_dir = os.path.join(run_dir, "predictions")
    predictions, pred_json = run_predictions(
        cfg, model, demo_images, pred_dir,
        conf=float(train_cfg.get("conf", 0.25)),
        iou=float(train_cfg.get("iou", 0.5)),
    )

    # --- XAI ---
    xai_dir = os.path.join(run_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    xai_cfg = cfg.get("xai", {})
    xai_methods = xai_cfg.get("methods", ["gradcam", "occlusion", "lime_image"])
    xai = {"gradcam": None, "occlusion": None, "lime_image": None}

    # imágenes para XAI: primeras con detección por método
    xai_x = demo_images

    gradcam_cfg = xai_cfg.get("gradcam") or {}
    if "gradcam" in xai_methods:
        try:
            out = run_gradcam(
                cfg, model, xai_x, xai_dir,
                mode=gradcam_cfg.get("mode", "max_detection"),
                target_class=gradcam_cfg.get("target_class"),
                min_conf=float(gradcam_cfg.get("min_conf", 0.25)),
            )
            xai["gradcam"] = out
        except Exception as exc:
            xai["gradcam"] = {"error": str(exc)}

    occlusion_cfg = xai_cfg.get("occlusion") or {}
    if "occlusion" in xai_methods:
        try:
            out = run_occlusion(
                cfg, model, xai_x, xai_dir,
                grid=occlusion_cfg.get("grid", [8, 8]),
                fill=occlusion_cfg.get("fill", "gray"),
                min_conf=float(occlusion_cfg.get("min_conf", 0.25)),
            )
            xai["occlusion"] = out
        except Exception as exc:
            xai["occlusion"] = {"error": str(exc)}

    lime_cfg = xai_cfg.get("lime_image") or {}
    if "lime_image" in xai_methods:
        try:
            out = run_lime_image(
                cfg, model, xai_x, xai_dir,
                num_samples=int(lime_cfg.get("num_samples", 1000)),
                num_features=int(lime_cfg.get("num_features", 10)),
                min_conf=float(lime_cfg.get("min_conf", 0.25)),
            )
            xai["lime_image"] = out
        except Exception as exc:
            xai["lime_image"] = {"error": str(exc)}

    if outputs.get("save_report", True):
        build_image_report(
            run_dir=run_dir, metrics=metrics, names=names,
            variant=variant, device=device, train_minutes=train_minutes,
            predictions=predictions, xai=xai, xai_dir=xai_dir,
            eda=eda_summary, model_meta=meta,
            narrative=bool(xai_cfg.get("narrative", True)),
        )

    # limpiar temporales de XAI
    for tmp in ("_lime_samp.jpg", "_occluded_tmp.jpg"):
        p = os.path.join(xai_dir, tmp)
        if os.path.exists(p):
            os.remove(p)

    return {
        "data_type": "image",
        "task": "detection",
        "model": "{} ({})".format(model_type, variant),
        "target": "/".join(names),
        "source": dataset_dir,
        "split_shapes": {
            "train": len(bundles[0].images),
            "val": len(bundles[1].images),
            "test": len(bundles[2].images),
        },
        "metrics": metrics,
        "xai": xai,
        "run_dir": run_dir,
        "validation": None,
        "warnings": [],
    }