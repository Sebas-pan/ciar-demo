"""Training for clustering models with optional k-search via silhouette."""
import numpy as np
from sklearn.metrics import silhouette_score

from .models import create_clustering_estimator


def _silhouette_for_k(X, labels):
    if len(set(labels)) < 2 or len(set(labels)) >= len(X):
        return -1
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1


def train_clustering_estimator(model_type, params, search, X_train, X_val=None):
    """Train clustering estimator.

    Args:
        model_type: e.g. 'kmeans' / 'k_means'
        params: base params dict
        search: None or dict e.g. {"n_clusters": [3,4,5,6]} -> picks best silhouette
        X_train: transformed array
        X_val: optional (unused for unsupervised but kept for API symmetry)

    Returns:
        (best_estimator, best_params, val_metrics)
        val_metrics is None (no supervised val) unless search performed.
    """
    # Ensure dense for sklearn
    if hasattr(X_train, "toarray"):
        X_arr = X_train.toarray()
    else:
        X_arr = np.asarray(X_train, dtype=float)

    if search and isinstance(search, dict) and "n_clusters" in search:
        candidates = search["n_clusters"]
        if not isinstance(candidates, (list, tuple)):
            candidates = [candidates]
        best_score = -2
        best_est = None
        best_params = None
        for k in candidates:
            p = dict(params or {})
            p["n_clusters"] = int(k)
            est = create_clustering_estimator(model_type, p)
            est.fit(X_arr)
            labels = est.labels_
            score = _silhouette_for_k(X_arr, labels)
            if score > best_score:
                best_score = score
                best_est = est
                best_params = p
        # val_metrics stores search ranking
        val_metrics = {"search_silhouette_best": float(best_score), "search_candidates": list(candidates)}
        return best_est, best_params, val_metrics

    # No search
    est = create_clustering_estimator(model_type, params or {})
    est.fit(X_arr)
    return est, None, None
