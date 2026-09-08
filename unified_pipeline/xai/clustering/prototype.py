"""Prototype / centroid analysis + PCA projection."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ...io import write_json


def run_prototype(model, X, feature_names, out_dir, preprocessor=None):
    """Centroid table, heatmap, PCA scatter."""
    if hasattr(X, "toarray"):
        X_arr = X.toarray()
    else:
        X_arr = np.asarray(X, dtype=float)

    centroids_scaled = model.cluster_centers_  # (k, p)
    k = int(model.n_clusters)
    labels = model.predict(X_arr)

    # Try inverse transform to original scale via preprocessor
    centroids_original = None
    if preprocessor is not None:
        try:
            # preprocessor is ColumnTransformer; inverse_transform returns original scale
            centroids_original = preprocessor.inverse_transform(centroids_scaled)
            # inverse_transform may return DataFrame-like; ensure array
            if isinstance(centroids_original, pd.DataFrame):
                centroids_original = centroids_original.values
        except Exception:
            centroids_original = None

    # Build table
    table_scaled = []
    for c in range(k):
        row = {"cluster": int(c), "count": int(np.sum(labels == c))}
        for j, f in enumerate(feature_names):
            row[f + "_scaled"] = float(centroids_scaled[c, j])
            if centroids_original is not None:
                try:
                    row[f] = float(centroids_original[c, j])
                except Exception:
                    pass
        table_scaled.append(row)

    # Heatmap scaled
    matrix = centroids_scaled  # (k, p)
    fig, ax = plt.subplots(figsize=(max(10, len(feature_names)*0.6), max(4, k*0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c} ({int(np.sum(labels==c))})" for c in range(k)])
    ax.set_title("Centroids (scaled space) — prototype heatmap")
    fig.colorbar(im, ax=ax, label="scaled value")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "centroids_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # If original scale available, second heatmap normalized per feature?
    if centroids_original is not None:
        try:
            # z-score per feature for visualization
            orig_arr = np.array([[r.get(f, 0) for f in feature_names] for r in table_scaled], dtype=float)
            # Save original centroids JSON
            orig_table = []
            for c in range(k):
                orig_table.append({f: float(orig_arr[c, j]) for j, f in enumerate(feature_names)})
            write_json(os.path.join(out_dir, "centroids_original.json"), orig_table)
        except Exception:
            pass

    write_json(os.path.join(out_dir, "centroids.json"), table_scaled)

    # PCA scatter
    # Cap to 2000 samples
    n_cap = min(len(X_arr), 2000)
    X_pc = X_arr[:n_cap]
    labels_pc = labels[:n_cap]
    pca = PCA(n_components=2, random_state=42)
    try:
        X_2d = pca.fit_transform(X_pc)
        centroids_2d = pca.transform(centroids_scaled)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pc, cmap="tab10", s=10, alpha=0.6)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="black", s=150, marker="*", edgecolor="white", linewidth=1, label="centroids")
        for c in range(k):
            ax.annotate(f"C{c}", centroids_2d[c], fontsize=9, weight="bold", ha="center", va="bottom")
        ax.set_title(f"PCA projection (explained {pca.explained_variance_ratio_.sum():.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        # colorbar for clusters
        legend = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "pca_scatter.png"), bbox_inches="tight")
        plt.close(fig)

        # Elbow curve if small k range
        ks = list(range(2, 11))
        inertias = []
        sils = []
        # Only compute if not too heavy; use X_pc
        from sklearn.metrics import silhouette_score
        for kk in ks:
            km = KMeans(n_clusters=kk, n_init=10, random_state=42)
            km.fit(X_pc)
            inertias.append(float(km.inertia_))
            try:
                sils.append(float(silhouette_score(X_pc, km.labels_)))
            except Exception:
                sils.append(float("nan"))
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(ks, inertias, marker="o", color="#4c78a8")
        ax1.set_xlabel("k")
        ax1.set_ylabel("inertia", color="#4c78a8")
        ax1.set_title("Elbow & Silhouette vs k")
        ax2 = ax1.twinx()
        ax2.plot(ks, sils, marker="s", color="#e45756")
        ax2.set_ylabel("silhouette", color="#e45756")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "elbow_silhouette.png"), bbox_inches="tight")
        plt.close(fig)

        elbow_json = {"k": ks, "inertia": inertias, "silhouette": sils}
        write_json(os.path.join(out_dir, "elbow.json"), elbow_json)
    except Exception as exc:
        write_json(os.path.join(out_dir, "pca_error.json"), {"error": str(exc)})

    return {"centroids": table_scaled}
