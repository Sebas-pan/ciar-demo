"""Clustering evaluation metrics."""
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _safe_silhouette(X, labels):
    try:
        if len(set(labels)) < 2 or len(set(labels)) >= len(X):
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def _safe_davies_bouldin(X, labels):
    try:
        if len(set(labels)) < 2:
            return None
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return None


def _safe_calinski_harabasz(X, labels):
    try:
        if len(set(labels)) < 2:
            return None
        return float(calinski_harabasz_score(X, labels))
    except Exception:
        return None


def evaluate_clustering(model, X):
    """Compute clustering metrics on transformed data X.

    Returns dict with inertia, silhouette, davies_bouldin, calinski_harabasz,
    n_clusters, cluster_sizes, n_samples.
    """
    if hasattr(model, "predict"):
        labels = model.predict(X)
    elif hasattr(model, "labels_"):
        labels = model.labels_
    else:
        raise ValueError("Model does not expose predict or labels_")

    # Ensure dense
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X)

    n_clusters = int(len(set(labels)))
    # cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

    metrics = {
        "n_clusters": n_clusters,
        "n_samples": int(X_arr.shape[0]),
        "cluster_sizes": cluster_sizes,
    }
    # inertia is specific to KMeans
    if hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)

    sil = _safe_silhouette(X_arr, labels)
    if sil is not None:
        metrics["silhouette"] = sil
    db = _safe_davies_bouldin(X_arr, labels)
    if db is not None:
        metrics["davies_bouldin"] = db
    ch = _safe_calinski_harabasz(X_arr, labels)
    if ch is not None:
        metrics["calinski_harabasz"] = ch

    return metrics
