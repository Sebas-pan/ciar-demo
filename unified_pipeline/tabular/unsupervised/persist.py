"""Persist metadata for clustering branch."""

def build_clustering_metadata(
    bundle, task, model_type, hyperparameters,
    feature_names, metrics, split_cfg,
    val_metrics=None, centroids=None,
):
    meta = {
        "version": "1.0",
        "data_type": "tabular",
        "framework": "sklearn",
        "task": task,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "feature_names": feature_names,
        "metrics": metrics,
        "split": split_cfg,
        "source": bundle.source,
        "description": bundle.description,
    }
    if val_metrics is not None:
        meta["val_metrics"] = val_metrics
    if centroids is not None:
        meta["centroids"] = centroids
    return meta
