"""Clustering model registry — currently K-Means, extensible to GMM/DBSCAN."""

from sklearn.cluster import KMeans

# Canonical type names (normalize dash/underscore)
_ALIASES = {
    "kmeans": "kmeans",
    "k_means": "kmeans",
    "k-means": "kmeans",
    "k_means_clustering": "kmeans",
}

_MODELS = {
    "kmeans": KMeans,
}


def normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    return _ALIASES.get(key, key)


def create_clustering_estimator(model_type, params):
    norm = normalize_model_type(model_type)
    if norm not in _MODELS:
        raise ValueError(
            "Unknown clustering model type '{}' (normalized '{}'). "
            "Available: {}".format(model_type, norm, sorted(_MODELS.keys()))
        )
    cls = _MODELS[norm]
    merged = dict(params or {})
    # Ensure deterministic defaults if not provided
    # KMeans: n_init 10 is safe for sklearn 1.4+
    return cls(**merged)
