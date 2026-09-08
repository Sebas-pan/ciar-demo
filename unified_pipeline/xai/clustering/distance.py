"""Distance-to-centroid XAI for clustering."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...io import write_json


def run_distance(model, X, feature_names, out_dir, max_samples=500):
    """Compute per-sample distance to assigned centroid and all centroids.

    Args:
        model: fitted KMeans
        X: transformed array (n, p)
        feature_names: list
        out_dir: xai dir
        max_samples: cap for plot
    """
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X, dtype=float)

    n = min(len(X_arr), max_samples)
    X_sub = X_arr[:n]
    labels = model.predict(X_sub)
    # transform gives distance to each centroid
    try:
        dists_all = model.transform(X_sub)  # (n, k)
    except Exception:
        # fallback manual
        centroids = model.cluster_centers_
        dists_all = np.linalg.norm(X_sub[:, None, :] - centroids[None, :, :], axis=2)

    dist_assigned = dists_all[np.arange(n), labels]

    # Save JSON (first 20 for brevity, full stats)
    summary = {
        "n_samples": int(len(X_arr)),
        "n_clusters": int(model.n_clusters),
        "distance_stats": {
            "mean": float(np.mean(dist_assigned)),
            "std": float(np.std(dist_assigned)),
            "min": float(np.min(dist_assigned)),
            "max": float(np.max(dist_assigned)),
            "median": float(np.median(dist_assigned)),
        },
        "per_cluster": {},
        "samples": [],  # first 20
    }
    for c in range(model.n_clusters):
        mask = labels == c
        if np.sum(mask) > 0:
            d = dist_assigned[mask]
            summary["per_cluster"][str(c)] = {
                "count": int(mask.sum()),
                "mean_distance": float(d.mean()),
                "median_distance": float(np.median(d)),
                "std_distance": float(d.std()),
            }

    for i in range(min(20, n)):
        summary["samples"].append({
            "index": int(i),
            "cluster": int(labels[i]),
            "distance_to_centroid": float(dist_assigned[i]),
            "distances_to_all": [float(v) for v in dists_all[i]],
        })

    write_json(os.path.join(out_dir, "distance_to_centroid.json"), summary)

    # Plot: histogram + per-cluster box
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dist_assigned, bins=30, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Distance to assigned centroid")
    axes[0].set_xlabel("Euclidean distance (scaled space)")
    axes[0].set_ylabel("Count")

    # per cluster hist overlay
    for c in range(model.n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            axes[1].hist(dist_assigned[mask], bins=20, alpha=0.5, label=f"C{c}")
    axes[1].set_title("Distance per cluster")
    axes[1].set_xlabel("distance")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distance_to_centroid.png"), bbox_inches="tight")
    plt.close(fig)

    return summary
