"""Permutation-based cluster importance."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...io import write_json
from sklearn.metrics import silhouette_score


def _metrics(X, model):
    labels = model.predict(X)
    # inertia proxy: mean distance to centroid
    try:
        dists = model.transform(X)
        assigned = dists[np.arange(len(X)), labels]
        inertia_proxy = float(np.mean(assigned ** 2))
    except Exception:
        inertia_proxy = float("nan")
    # silhouette
    try:
        if len(set(labels)) > 1 and len(set(labels)) < len(X):
            sil = float(silhouette_score(X, labels))
        else:
            sil = float("nan")
    except Exception:
        sil = float("nan")
    return inertia_proxy, sil, labels


def run_permutation(model, X, feature_names, out_dir, n_repeats=5, random_state=42):
    """For each feature, permute column and measure delta in inertia and flip rate."""
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X, dtype=float)

    # Cap size for speed
    n_cap = min(len(X_arr), 1000)
    rng = np.random.RandomState(random_state)
    X_base = X_arr[:n_cap]
    base_inertia, base_sil, base_labels = _metrics(X_base, model)

    results = {}
    for j, fname in enumerate(feature_names):
        inertia_deltas = []
        sil_deltas = []
        flip_rates = []
        for r in range(n_repeats):
            X_perm = X_base.copy()
            perm = rng.permutation(X_perm[:, j])
            X_perm[:, j] = perm
            inertia_p, sil_p, labels_p = _metrics(X_perm, model)
            # delta inertia: increase means feature matters
            if np.isfinite(base_inertia) and np.isfinite(inertia_p):
                inertia_deltas.append(float(inertia_p - base_inertia))
            if np.isfinite(base_sil) and np.isfinite(sil_p):
                # drop in silhouette
                sil_deltas.append(float(base_sil - sil_p))
            # flip rate
            flip = float(np.mean(labels_p != base_labels))
            flip_rates.append(flip)

        results[fname] = {
            "mean_delta_inertia": float(np.mean(inertia_deltas)) if inertia_deltas else 0.0,
            "mean_delta_silhouette": float(np.mean(sil_deltas)) if sil_deltas else 0.0,
            "mean_flip_rate": float(np.mean(flip_rates)),
            "std_flip_rate": float(np.std(flip_rates)),
        }

    # Sort by flip_rate then delta_inertia
    sorted_feats = sorted(results.items(), key=lambda t: -(t[1]["mean_flip_rate"] + t[1]["mean_delta_inertia"]))

    summary = {
        "n_samples": int(n_cap),
        "n_repeats": int(n_repeats),
        "base_inertia": float(base_inertia) if np.isfinite(base_inertia) else None,
        "base_silhouette": float(base_sil) if np.isfinite(base_sil) else None,
        "importance": [
            {"feature": f, **vals} for f, vals in sorted_feats
        ],
    }
    write_json(os.path.join(out_dir, "permutation_importance.json"), summary)

    # Plot
    feats = [f for f, _ in sorted_feats]
    flips = [results[f]["mean_flip_rate"] for f in feats]
    deltas = [results[f]["mean_delta_inertia"] for f in feats]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feats)*0.4)))
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, flips, color="#4c78a8", label="flip rate")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("mean flip rate after permutation")
    ax.set_title("Permutation importance — cluster assignment flip")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "permutation_importance.png"), bbox_inches="tight")
    plt.close(fig)

    # Second plot delta inertia
    fig, ax = plt.subplots(figsize=(10, max(4, len(feats)*0.4)))
    ax.barh(y_pos, deltas, color="#e45756")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("delta inertia (perm - base)")
    ax.set_title("Permutation importance — inertia increase")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "permutation_delta_inertia.png"), bbox_inches="tight")
    plt.close(fig)

    return summary
