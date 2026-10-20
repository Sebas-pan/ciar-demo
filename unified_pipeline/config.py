import os

import yaml

DEFAULTS = {
    "data_type": "auto",
    "data": {"source": None, "target": "auto", "drop_columns": []},
    "supervised": {"task": "auto"},
    "unsupervised": {"task": None},
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
        "methods": None,
        "permutation_repeats": 5,
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
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            user = yaml.safe_load(fh) or {}
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-16") as fh:
            user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError("Config file must contain a top-level mapping: {}".format(path))
    cfg = _merge(DEFAULTS, user)
    if not cfg["data"].get("source"):
        raise ValueError("data.source is required in the config file")
    _validate_config(cfg)
    # output.dir stays CWD-relative (repo root): all shipped configs assume
    # running from the repo root (e.g. "unified_pipeline/outputs").
    cfg["output"]["dir"] = os.path.abspath(cfg["output"].get("dir") or "outputs")
    return cfg


def _validate_config(cfg):
    ratios = ((cfg.get("split") or {}).get("train_val_test") or [0.7, 0.15, 0.15])
    if len(ratios) != 3:
        raise ValueError("split.train_val_test must have exactly 3 ratios")
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise ValueError(
            "split.train_val_test ratios must be >= 0 and sum to 1.0, got {}".format(ratios))
    sup = str((cfg.get("supervised") or {}).get("task", "auto")).strip().lower()
    unsup = (cfg.get("unsupervised") or {}).get("task")
    if sup not in ("auto", "") and unsup not in (None, "", "auto",
                                                 "AUTO", " Auto "):
        raise ValueError(
            "supervised.task ('{}') and unsupervised.task ('{}') conflict: "
            "set only one".format(sup, unsup))
    if ((cfg.get("split") or {}).get("stratify")
            and sup in ("regression", "regresion")):
        import warnings
        warnings.warn("split.stratify is ignored for regression tasks")
    model_type = (cfg.get("model") or {}).get("type")
    if model_type is not None and not isinstance(model_type, str):
        raise ValueError("model.type must be a string")


def make_run_dir(base_dir, prefix="runs"):
    """Central run-dir factory with microsecond stamp to avoid collisions."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base_dir, prefix, stamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir