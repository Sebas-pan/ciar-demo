"""Unsupervised clustering pipeline — K-Means with 4 XAI methods."""
import datetime
import os

import numpy as np

from ...data.loader import load_dataset
from ...data.validation import validate
from ...io import write_json
from ...modelstore import save_artifact
from ..common.preprocess import build_preprocessor, transformed_feature_names
from .eda import run_eda_unsupervised
from .evaluate import evaluate_clustering
from .models import normalize_model_type
from .persist import build_clustering_metadata
from .split import split_unsupervised
from .trainer import train_clustering_estimator


def _normalize_task(cfg):
    task = (cfg.get("unsupervised") or {}).get("task", "auto")
    if task is None:
        return None
    t = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    # aliases
    if t in ("k_means", "kmeans", "k_means_clustering"):
        return "kmeans"
    return t


def run_clustering(cfg):
    # 1. Load
    bundle = load_dataset(cfg)
    df = bundle.df

    validation_report, warnings = validate(df)

    # Resolve task
    task = _normalize_task(cfg)
    if task is None:
        task = "kmeans"
    if task != "kmeans":
        raise ValueError(f"Unsupported unsupervised task '{task}': only kmeans/k_means is supported")

    split_cfg = cfg.get("split") or {"train_val_test": [0.85, 0.0, 0.15], "random_state": 42}

    # 2. Split (unsupervised — no y)
    parts = split_unsupervised(df, split_cfg)

    # 3. Preprocess (fit on train only)
    preprocessor, _, _ = build_preprocessor(parts["X_train"])
    X_train_t = preprocessor.transform(parts["X_train"])
    # handle empty val/test
    X_val_t = preprocessor.transform(parts["X_val"]) if len(parts["X_val"]) > 0 else np.empty((0, X_train_t.shape[1]))
    X_test_t = preprocessor.transform(parts["X_test"]) if len(parts["X_test"]) > 0 else X_train_t
    # For metrics we want test; if empty use train
    X_eval = X_test_t if len(X_test_t) > 0 else X_train_t
    feature_names = transformed_feature_names(preprocessor)

    # 4. Train
    model_cfg = cfg.get("model") or {}
    model_type_raw = model_cfg.get("type", "kmeans")
    model_type = normalize_model_type(model_type_raw)
    params = model_cfg.get("params") or {}
    # Ensure n_clusters default
    if "n_clusters" not in params and "n_clusters" not in (model_cfg.get("search") or {}):
        params = dict(params)
        params["n_clusters"] = 4
    search = model_cfg.get("search")

    estimator, best_params, val_metrics = train_clustering_estimator(
        model_type=model_type, params=params, search=search, X_train=X_train_t, X_val=X_val_t
    )
    if best_params is None:
        best_params = params

    # 5. Evaluate on eval set (test)
    metrics = evaluate_clustering(estimator, X_eval)

    # 6. Outputs
    outputs = cfg.get("output") or {}
    run_dir = os.path.join(
        outputs.get("dir", "unified_pipeline/outputs"),
        "runs",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    # Ensure abspath like config does
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    run_eda_unsupervised(df, run_dir)

    # Centroids for metadata
    centroids = None
    try:
        centroids = estimator.cluster_centers_.tolist()
    except Exception:
        centroids = None

    meta = build_clustering_metadata(
        bundle=bundle,
        task=task,
        model_type=model_type,
        hyperparameters=best_params,
        feature_names=feature_names,
        metrics=metrics,
        split_cfg=split_cfg,
        val_metrics=val_metrics,
        centroids=centroids,
    )
    if outputs.get("save_model", True):
        artifact_dir = os.path.join(run_dir, "model")
        save_artifact(artifact_dir, estimator, preprocessor, meta)
    else:
        write_json(os.path.join(run_dir, "metadata.json"), meta)

    # 7. XAI
    xai_dir = os.path.join(run_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    xai_cfg = cfg.get("xai") or {}
    # methods list controls which run; default all
    methods = xai_cfg.get("methods")
    if methods is None:
        methods = ["distance", "contribution", "permutation", "prototype"]
    # normalize
    methods = [str(m).lower() for m in methods]
    xai = {"distance": None, "contribution": None, "permutation": None, "prototype": None}

    # Use full data transformed for XAI (train+test) for better coverage — use train
    X_xai = X_train_t
    if len(X_xai) > 2000:
        # keep 2000 cap inside each XAI but pass all; functions handle cap
        pass

    if "distance" in methods or "distance_to_centroid" in methods:
        try:
            from ...xai.clustering.distance import run_distance
            out = run_distance(estimator, X_xai, feature_names, xai_dir)
            xai["distance"] = {"saved": True, "stats": out.get("distance_stats")}
        except Exception as exc:
            xai["distance"] = {"error": str(exc)}

    if "contribution" in methods or "feature_contribution" in methods:
        try:
            from ...xai.clustering.contribution import run_contribution
            out = run_contribution(estimator, X_xai, feature_names, xai_dir, top_k=int(xai_cfg.get("top_features", 5)))
            xai["contribution"] = {"saved": True}
        except Exception as exc:
            xai["contribution"] = {"error": str(exc)}

    if "permutation" in methods or "permutation_based" in methods:
        try:
            from ...xai.clustering.permutation import run_permutation
            out = run_permutation(estimator, X_xai, feature_names, xai_dir, n_repeats=int(xai_cfg.get("permutation_repeats", 5)))
            xai["permutation"] = {"saved": True}
        except Exception as exc:
            xai["permutation"] = {"error": str(exc)}

    if "prototype" in methods or "centroid" in methods or "prototype_analysis" in methods:
        try:
            from ...xai.clustering.prototype import run_prototype
            out = run_prototype(estimator, X_xai, feature_names, xai_dir, preprocessor=preprocessor)
            xai["prototype"] = {"saved": True}
        except Exception as exc:
            xai["prototype"] = {"error": str(exc)}

    # 8. Report (with optional LLM narration)
    if outputs.get("save_report", True):
        use_llm = bool(xai_cfg.get("use_llm", False))
        narrative = bool(xai_cfg.get("narrative", True))
        _llm_narrate_cluster = None
        _llm_narrate_profiles = None
        if use_llm:
            try:
                from ...xai.llm_narrator import narrate_clustering as _llm_c
                from ...xai.llm_narrator import narrate_cluster_profiles as _llm_p
                _llm_narrate_cluster = _llm_c
                _llm_narrate_profiles = _llm_p
            except Exception:
                pass
        _write_clustering_report(
            run_dir, xai_dir, metrics, task, model_type, feature_names, xai,
            narrative=narrative,
            narrate_cluster_func=_llm_narrate_cluster,
            narrate_profiles_func=_llm_narrate_profiles,
        )

    return {
        "data_type": "tabular",
        "task": task,
        "model": model_type,
        "target": None,
        "source": bundle.source,
        "split_shapes": {
            "train": int(X_train_t.shape[0]),
            "val": int(X_val_t.shape[0]) if hasattr(X_val_t, "shape") else 0,
            "test": int(X_eval.shape[0]),
        },
        "metrics": metrics,
        "val_metrics": val_metrics,
        "xai": xai,
        "run_dir": run_dir,
        "validation": validation_report,
        "warnings": warnings,
    }


def _write_clustering_report(run_dir, xai_dir, metrics, task, model_type, feature_names, xai,
                           narrative=True, narrate_cluster_func=None, narrate_profiles_func=None):
    lines = ["# Clustering Report", ""]
    lines.append(f"**Task:** {task} | **Model:** {model_type}")
    lines.append("")
    lines.append("## Metrics")
    for k, v in metrics.items():
        if k == "cluster_sizes":
            lines.append(f"- **{k}**: {v}")
        elif isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        elif isinstance(v, int):
            lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Narrative (LLM or deterministic)
    if narrative:
        lines.append("## What the model did")
        try:
            from ...xai.narrator import narrate_clustering as _det_c
            from ...xai.narrator import narrate_cluster_profiles as _det_p
            _narrate_c = narrate_cluster_func or _det_c
            _narrate_p = narrate_profiles_func or _det_p
            lines.append(_narrate_c(metrics, model_type, len(feature_names)))
            lines.append("")
            # Profiles — read centroids.json if available
            import json as _json
            import os as _os
            centroids_path = _os.path.join(xai_dir, "centroids.json")
            centroids_data = None
            if _os.path.exists(centroids_path):
                try:
                    with open(centroids_path, "r", encoding="utf-8") as fh:
                        centroids_data = _json.load(fh)
                except Exception:
                    centroids_data = None
            profiles_txt = _narrate_p(centroids_path, centroids_data, top_k=3)
            if profiles_txt:
                lines.append(profiles_txt)
                lines.append("")
        except Exception:
            # Fallback inline (previous behaviour)
            n_c = metrics.get("n_clusters")
            sil = metrics.get("silhouette")
            inertia = metrics.get("inertia")
            if n_c:
                lines.append(f"The model discovered **{n_c} clusters** via K-Means on {len(feature_names)} scaled features.")
            if sil is not None:
                qual = "well-separated" if sil > 0.5 else "moderately separated" if sil > 0.25 else "weakly separated / overlapping"
                lines.append(f"Silhouette = **{sil:.3f}** ({qual}). Higher is better (range -1..1).")
            if inertia is not None:
                lines.append(f"Inertia (within-cluster sum of squares) = **{inertia:.1f}** — lower means tighter clusters.")
            cs = metrics.get("cluster_sizes") or {}
            if cs:
                total = sum(cs.values())
                for c, cnt in sorted(cs.items()):
                    lines.append(f"- Cluster {c}: {cnt} samples ({cnt/total:.1%})")
            lines.append("")

    # XAI sections
    lines.append("## XAI — Cluster explanations")
    for name in ["prototype", "distance", "contribution", "permutation"]:
        status = xai.get(name)
        if status is None:
            continue
        if "error" in status:
            lines.append(f"### {name} — ERROR: {status['error']}")
        elif status.get("saved"):
            lines.append(f"### {name} — OK")
            if name == "prototype":
                lines.append("- `xai/centroids.json` (scaled centroids), `xai/centroids_heatmap.png`, `xai/pca_scatter.png`, `xai/elbow_silhouette.png`")
                lines.append("- Centroids are prototypes; heatmap shows which features are high/low per cluster.")
            elif name == "distance":
                lines.append("- `xai/distance_to_centroid.json` + `xai/distance_to_centroid.png` — per-sample distance to assigned centroid; outliers have large distances.")
            elif name == "contribution":
                lines.append("- `xai/feature_contribution.json` + `xai/feature_contribution.png` — decomposes distance into per-feature contribution; high contribution means that feature pushed sample away from centroid.")
            elif name == "permutation":
                lines.append("- `xai/permutation_importance.json` + `xai/permutation_importance.png` — shuffling a feature and measuring flip-rate / inertia increase; high flip means feature drives assignment.")
        lines.append("")

    lines.append("## How to interpret")
    lines.append("- **Prototype**: Compare centroids across clusters to name segments (e.g., high `PURCHASES` vs high `CASH_ADVANCE`).")
    lines.append("- **Distance**: Samples far from centroid are atypical for that segment.")
    lines.append("- **Contribution**: For a sample, top contributors explain *why* it is far/near.")
    lines.append("- **Permutation**: Global ranking of which features most affect clustering.")
    lines.append("")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    # Also save JSON summary for machine use
    write_json(os.path.join(xai_dir, "xai_summary.json"), xai)
