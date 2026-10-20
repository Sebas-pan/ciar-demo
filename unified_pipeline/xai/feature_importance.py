import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def global_importance(model, feature_names, X=None, y=None, n_repeats=5,
                      random_state=42):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = abs(coef[0]) if coef.ndim > 1 else abs(coef)
        method = "abs(coef_)"
    else:
        # Fallback for models without native importance (e.g. SVM RBF,
        # CalibratedClassifierCV(SVC)): permutation importance on held-out data.
        if X is None or y is None:
            raise ValueError(
                "Model does not expose feature importance or coefficients "
                "(pass X, y for permutation fallback)")
        from sklearn.inspection import permutation_importance

        r = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        values = r.importances_mean
        method = "permutation"
    table = pd.DataFrame({
        "feature": feature_names,
        "importance": values,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return table, method


def plot_importance(table, out_path, top_n=15):
    data = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["feature"], data["importance"])
    ax.set_title("Global Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)