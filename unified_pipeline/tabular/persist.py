import datetime
import json
import os


def build_tabular_metadata(
    bundle, task, model_type, hyperparameters,
    feature_names, classes, metrics, split_cfg,
):
    return {
        "version": "1.0",
        "data_type": "tabular",
        "framework": "sklearn",
        "task": task,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "feature_names": feature_names,
        "classes": classes,
        "metrics": metrics,
        "split": split_cfg,
        "source": bundle.source,
        "description": bundle.description,
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)