import os

import yaml

DEFAULTS = {
    "data_type": "auto",
    "data": {"source": None, "target": "auto", "drop_columns": []},
    "supervised": {"task": "auto"},
    "split": {
        "train_val_test": [0.7, 0.15, 0.15],
        "stratify": True,
        "random_state": 42,
    },
    "model": {"type": "random_forest", "params": {}, "search": None},
    "xai": {
        "feature_importance": True,
        "shap": True,
        "lime": True,
        "num_features": 10,
        "num_instances": 3,
        "narrative": True,
        "top_features": 15,
    },
    "output": {"dir": "outputs", "save_model": True, "save_report": True},
}


def _merge(base, override):
    from copy import deepcopy

    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    if not cfg["data"].get("source"):
        raise ValueError("data.source is required in the config file")
    cfg["output"]["dir"] = os.path.abspath(cfg["output"]["dir"])
    return cfg