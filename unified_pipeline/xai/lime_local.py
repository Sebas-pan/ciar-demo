import os

import numpy as np
from lime import lime_tabular

from ..io import write_json


def run_lime(model, X_train, y_train, X_test, y_test, feature_names,
             class_names, num_features=10, num_instances=3,
             random_state=42, out_dir=None, mode="classification"):
    training = X_train
    if hasattr(training, "toarray"):
        training = training.toarray()
    if mode == "regression":
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(training),
            feature_names=list(feature_names),
            mode="regression",
            random_state=random_state,
        )

        def predict_fn(X):
            return model.predict(np.asarray(X, dtype=float))

        coerce_true = float
    else:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(training),
            feature_names=list(feature_names),
            class_names=list(class_names) if class_names is not None else None,
            mode="classification",
            random_state=random_state,
        )

        def predict_fn(X):
            return model.predict_proba(np.asarray(X, dtype=float))

        coerce_true = int

    explanations = []
    for i in range(min(num_instances, X_test.shape[0])):
        data_row = X_test[i]
        if hasattr(data_row, "toarray"):
            data_row = data_row.toarray().reshape(-1)
        exp = explainer.explain_instance(
            data_row=np.asarray(data_row, dtype=float),
            predict_fn=predict_fn,
            num_features=num_features,
        )
        explanations.append({
            "position": i,
            "index": int(y_test.index[i]),
            "true": coerce_true(y_test.iloc[i]),
            "as_list": exp.as_list(),
        })
    if out_dir:
        write_json(os.path.join(out_dir, "lime_local.json"), explanations)
    return explanations