import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from ..io import write_json


def is_tree_based(model):
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )

    if isinstance(
        model,
        (RandomForestClassifier, RandomForestRegressor,
         DecisionTreeClassifier, DecisionTreeRegressor),
    ):
        return True
    # XGBoost models use the fast TreeExplainer path when installed.
    try:
        import xgboost as _xgb

        if isinstance(model, (_xgb.XGBClassifier, _xgb.XGBRegressor)):
            return True
    except Exception:
        pass
    return False


def run_shap(model, X_train, X_test, feature_names, out_dir,
             sample_limit=200, num_instances=3, top_n=8, task="classification",
             random_state=42):
    rng = np.random.default_rng(random_state)
    if is_tree_based(model):
        explainer = _safe_tree_explainer(model)
    else:
        n_bg = min(100, X_train.shape[0])
        bg_idx = rng.choice(X_train.shape[0], n_bg, replace=False)
        background = X_train[bg_idx] if hasattr(X_train, "__getitem__") else X_train[:n_bg]
        if hasattr(background, "toarray"):
            background = background.toarray()
        explainer = _safe_kernel_explainer(model, background, task=task)

    n_eval = min(sample_limit, X_test.shape[0])
    eval_idx = rng.choice(X_test.shape[0], n_eval, replace=False)
    X_eval = X_test[eval_idx] if hasattr(X_test, "__getitem__") else X_test[:n_eval]
    if hasattr(X_eval, "toarray"):
        X_eval = X_eval.toarray()
    raw = explainer.shap_values(np.asarray(X_eval, dtype=float))

    if isinstance(raw, list):
        arr = np.array(raw[-1], dtype=float)
    else:
        arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., -1]

    global_import = np.abs(arr).mean(axis=0)
    table = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_import,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = table.head(15).iloc[::-1]
    ax.barh(plot_data["feature"], plot_data["mean_abs_shap"])
    ax.set_title("SHAP Global Importance (mean |SHAP|)")
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_global.png"), bbox_inches="tight")
    plt.close(fig)

    write_json(os.path.join(out_dir, "shap_global.json"),
               table.to_dict(orient="records"))

    instances = []
    for i in range(min(num_instances, len(arr))):
        ranked = sorted(
            zip(feature_names, arr[i]), key=lambda t: -abs(t[1]))
        instances.append({
            "index": i,
            "top_features": [
                [str(f), round(float(v), 4)] for f, v in ranked[:top_n]
            ],
        })
    return {"instances": instances}


def _safe_tree_explainer(model):
    import shap

    return shap.TreeExplainer(model)


def _safe_kernel_explainer(model, background, task="classification"):
    import shap

    if task == "classification" and hasattr(model, "predict_proba"):
        def predict_fn(X):
            return model.predict_proba(np.asarray(X, dtype=float))
    elif task == "classification" and hasattr(model, "decision_function"):
        def predict_fn(X):
            return model.decision_function(np.asarray(X, dtype=float))
    else:
        def predict_fn(X):
            return model.predict(np.asarray(X, dtype=float))
    return shap.KernelExplainer(predict_fn, np.asarray(background, dtype=float))