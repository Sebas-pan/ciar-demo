import datetime
import json
import os


def build_image_metadata(
    dataset_dir, data_spec, variant, weights, metrics,
    train_cfg, device, source,
):
    return {
        "version": "1.0",
        "data_type": "image",
        "framework": "torch",
        "task": "detection",
        "model_type": "yolo11",
        "variant": variant,
        "weights": weights,
        "nc": data_spec["nc"],
        "names": data_spec["names"],
        "metrics": metrics,
        "train": {k: v for k, v in train_cfg.items() if k != "augment"} if train_cfg else {},
        "device": str(device),
        "dataset_dir": dataset_dir,
        "source": source,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)