import importlib.util

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

_MODELS = {
    "classification": {
        "logistic": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "xgboost": "XGBClassifier",
    },
    "regression": {
        "logistic": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "svm": SVR,
        "xgboost": "XGBRegressor",
    },
}

_DEFAULTS = {
    "classification": {
        "logistic": {"max_iter": 1000},
    },
    "regression": {},
}


def _xgb_class(model_name):
    if importlib.util.find_spec("xgboost") is None:
        raise ImportError(
            "xgboost is not installed in the environment; "
            "choose another model type or install xgboost")
    import xgboost

    return getattr(xgboost, model_name)


def create_estimator(model_type, task, params):
    table = _MODELS[task]
    if model_type not in table:
        raise ValueError(
            "Unknown model type '{}' for task '{}'".format(model_type, task))
    merged = dict(_DEFAULTS.get(task, {}).get(model_type, {}))
    merged.update(params or {})
    if model_type == "svm" and task == "classification":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        return CalibratedClassifierCV(
            SVC(**merged), method="sigmoid", cv=3)
    entry = table[model_type]
    if isinstance(entry, str):
        cls = _xgb_class(entry)
    else:
        cls = entry
    return cls(**merged)