import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(
            y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(
            y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0)),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    return metrics


def _regression_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def evaluate(model, task, X_test, y_test):
    if task == "classification":
        return _classification_metrics(model, X_test, y_test)
    return _regression_metrics(model, X_test, y_test)