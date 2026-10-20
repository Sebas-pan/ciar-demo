from sklearn.model_selection import GridSearchCV
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
)

from .models import create_estimator


def _classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_estimator(model_type, task, params, search, X_train, y_train, X_val=None, y_val=None, cv=3):
    from sklearn.model_selection import StratifiedKFold
    estimator = create_estimator(model_type, task, params)
    if search:
        if task == "classification":
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        else:
            cv_obj = cv
            scoring = "r2"
        grid = GridSearchCV(estimator, search, cv=cv_obj, scoring=scoring, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        best_params = dict(grid.best_params_)
    else:
        try:
            estimator.fit(X_train, y_train)
        except ValueError as exc:
            # e.g. CalibratedClassifierCV(cv=3) on tiny/rare classes:
            # fall back to uncalibrated SVC so the run survives.
            if model_type == "svm" and task == "classification":
                import warnings
                warnings.warn(
                    "CalibratedClassifierCV failed ({}); "
                    "falling back to plain SVC".format(exc))
                from sklearn.svm import SVC
                fb_params = dict(params or {})
                fb_params.pop("calibration_cv", None)
                estimator = SVC(**fb_params)
                estimator.fit(X_train, y_train)
            else:
                raise
        best_estimator = estimator
        best_params = None

    # Compute validation metrics if val data provided and non-empty
    val_metrics = None
    if (X_val is not None and y_val is not None
            and len(X_val) > 0 and len(y_val) > 0):
        y_val_pred = best_estimator.predict(X_val)
        if task == "classification":
            val_metrics = _classification_metrics(y_val, y_val_pred)
        else:
            val_metrics = _regression_metrics(y_val, y_val_pred)

    return best_estimator, best_params, val_metrics