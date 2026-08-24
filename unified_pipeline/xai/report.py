import json
import os

import numpy as np

from ..io import write_json
from .narrator import narrate_instance, narrate_model, narrate_top_features


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_report(model, task, X_test, y_test, class_names,
                 num_instances, xai):
    instances = []
    lime = xai.get("lime") or {}
    shap_data = xai.get("shap") or {}
    lime_instances = lime.get("instances") or []
    shap_instances = shap_data.get("instances") or []

    def _coerce(value):
        return float(value) if task == "regression" else int(value)

    for i in range(min(num_instances, X_test.shape[0])):
        rec = {
            "position": i,
            "index": int(y_test.index[i]),
            "true": _coerce(y_test.iloc[i]),
        }
        row = X_test[i]
        if hasattr(row, "toarray"):
            row = row.toarray()
        row = np.asarray(row, dtype=float).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            rec["probabilities"] = {
                str(c): round(float(p), 4)
                for c, p in zip(model.classes_, proba)
            }
            rec["prediction"] = int(model.classes_[int(np.argmax(proba))])
        else:
            rec["prediction"] = _coerce(model.predict(row)[0])
        if i < len(lime_instances):
            rec["top_features_lime"] = lime_instances[i].get("as_list", [])
        if i < len(shap_instances):
            rec["top_shap"] = shap_instances[i].get("top_features", [])
        instances.append(rec)
    return instances


def write_report(instances, xai_dir, run_dir, metrics=None, task=None,
                 model_type=None, target=None, top_n=15, narrative=True):
    write_json(os.path.join(xai_dir, "explanatory_instances.json"), instances)
    lines = [
        "# Instancia explicativa",
        "",
    ]

    if narrative:
        lines.append("## ¿Qué está haciendo este modelo?")
        lines.append("")
        if metrics and task:
            lines.append(narrate_model(metrics, task, model_type, target))
            lines.append("")
        fi_records = _read_json(os.path.join(xai_dir, "feature_importance.json"))
        shap_records = _read_json(os.path.join(xai_dir, "shap_global.json"))
        top = narrate_top_features(fi_records, shap_records, top_n=top_n)
        if top:
            lines.append(top)
            lines.append("")

        lines.append("## ¿Por qué el modelo predijo lo que predijo?")
        lines.append("")
        for rec in instances:
            narrative_txt = narrate_instance(rec, task or "classification")
            if narrative_txt:
                lines.append("### Muestra (index {})".format(rec["index"]))
                lines.append(narrative_txt)
                lines.append("")

    for rec in instances:
        lines.append("## Detalle técnico — Muestra (index {})".format(rec["index"]))
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