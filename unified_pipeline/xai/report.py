import os

import numpy as np

from ..tabular.persist import write_json


def build_report(model, task, X_test, y_test, class_names,
                 num_instances, xai):
    instances = []
    lime = xai.get("lime") or {}
    shap_data = xai.get("shap") or {}
    lime_instances = lime.get("instances") or []
    shap_instances = shap_data.get("instances") or []

    for i in range(min(num_instances, len(X_test))):
        rec = {
            "position": i,
            "index": int(y_test.index[i]),
            "true": int(y_test.iloc[i]),
        }
        row = np.asarray(X_test[i], dtype=float).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            rec["probabilities"] = {
                str(c): round(float(p), 4)
                for c, p in zip(model.classes_, proba)
            }
            rec["prediction"] = int(model.classes_[int(np.argmax(proba))])
        else:
            rec["prediction"] = int(model.predict(row)[0])
        if i < len(lime_instances):
            rec["top_features_lime"] = lime_instances[i].get("as_list", [])
        if i < len(shap_instances):
            rec["top_shap"] = shap_instances[i].get("top_features", [])
        instances.append(rec)
    return instances


def write_report(instances, xai_dir, run_dir):
    write_json(os.path.join(xai_dir, "explanatory_instances.json"), instances)
    lines = [
        "# Instancia explicativa",
        "",
    ]
    for rec in instances:
        lines.append("## Muestra (index {})".format(rec["index"]))
        lines.append("- True: {} | Pred: {}".format(
            rec["true"], rec["prediction"]))
        if "probabilities" in rec:
            lines.append("- Probabilidades: {}".format(rec["probabilities"]))
        if "top_features_lime" in rec:
            lines.append("- Top features LIME:")
            for feat, weight in rec["top_features_lime"]:
                lines.append("    - {} -> {:.4f}".format(feat, weight))
        if "top_shap" in rec:
            lines.append("- Top SHAP:")
            for feat, value in rec["top_shap"]:
                lines.append("    - {} -> {:.4f}".format(feat, value))
        lines.append("")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))