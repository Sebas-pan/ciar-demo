"""Feature contribution to distance XAI."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...io import write_json


def run_contribution(model, X, feature_names, out_dir, top_k=5, max_samples=200):
    """Decompose distance to centroid per feature.

    contrib_ij = (x_ij - mu_cj)^2 / sum_j (x_ij - mu_cj)^2
    """
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X, dtype=float)

    n = min(len(X_arr), max_samples)
    X_sub = X_arr[:n]
    labels = model.predict(X_sub)
    centroids = model.cluster_centers_  # (k, p)

    # per-sample contributions (n, p)
    diff = X_sub - centroids[labels]  # (n, p)
    sq = diff ** 2
    denom = sq.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1e-12, denom)
    contrib = sq / denom  # (n, p)

    # Per-cluster mean contribution
    k = model.n_clusters
    p = len(feature_names)
    per_cluster = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            per_cluster[str(c)] = {f: 0.0 for f in feature_names}
            continue
        mean_c = contrib[mask].mean(axis=0)
        per_cluster[str(c)] = {f: float(v) for f, v in zip(feature_names, mean_c)}

    # Global mean
    global_mean = contrib.mean(axis=0)

    # Save JSON
    summary = {
        "n_samples": int(n),
        "n_clusters": int(k),
        "global_mean_contribution": {f: float(v) for f, v in zip(feature_names, global_mean)},
        "per_cluster_mean": per_cluster,
        "samples": [],  # first 10 with top features
    }
    for i in range(min(10, n)):
        # top contributors for this sample
        row = contrib[i]
        ranked = sorted(zip(feature_names, row), key=lambda t: -t[1])
        summary["samples"].append({
            "index": int(i),
            "cluster": int(labels[i]),
            "top_contributors": [[f, float(v)] for f, v in ranked[:top_k]],
        })

    write_json(os.path.join(out_dir, "feature_contribution.json"), summary)

    # Plot: heatmap per cluster
    import matplotlib.colors as mcolors
    matrix = np.array([[per_cluster[str(c)][f] for f in feature_names] for c in range(k)])
    fig, ax = plt.subplots(figsize=(max(8, p * 0.6), max(4, k * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(p))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c}" for c in range(k)])
    ax.set_title("Mean feature contribution to distance (per cluster)")
    fig.colorbar(im, ax=ax, label="mean contribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_contribution.png"), bbox_inches="tight")
    plt.close(fig)

    # Bar per cluster: top_k
    for c in range(k):
        vals = per_cluster[str(c)]
        sorted_items = sorted(vals.items(), key=lambda t: -t[1])[:top_k]
        feats, vs = zip(*sorted_items)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(feats)[::-1], list(vs)[::-1], color="#72b66f")
        ax.set_title(f"Top {top_k} contributors — Cluster {c}")
        ax.set_xlabel("mean contribution")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"feature_contribution_C{c}.png"), bbox_inches="tight")
        plt.close(fig)

    return summary
