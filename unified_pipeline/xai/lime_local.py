import os

import numpy as np
from lime import lime_tabular

from ..tabular.persist import write_json


def run_lime(model, X_train, y_train, X_test, y_test, feature_names,
             class_names, num_features=10, num_instances=3,
             random_state=42, out_dir=None):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        random_state=random_state,
    )

    def predict_fn(X):
        return model.predict_proba(np.asarray(X, dtype=float))

    explanations = []
    for i in range(min(num_instances, len(X_test))):
        exp = explainer.explain_instance(
            data_row=np.asarray(X_test[i], dtype=float),
            predict_fn=predict_fn,
            num_features=num_features,
        )
        explanations.append({
            "position": i,
            "index": int(y_test.index[i]),
            "true": int(y_test.iloc[i]),
            "as_list": exp.as_list(),
        })
    if out_dir:
        write_json(os.path.join(out_dir, "lime_local.json"), explanations)
    return explanations