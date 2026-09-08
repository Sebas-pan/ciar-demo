def build_tabular_metadata(
    bundle, task, model_type, hyperparameters,
    feature_names, classes, metrics, split_cfg,
    val_metrics=None,
):
    meta = {
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
    if val_metrics is not None:
        meta["val_metrics"] = val_metrics
    return meta