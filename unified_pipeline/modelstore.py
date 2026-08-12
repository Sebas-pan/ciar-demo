import datetime
import json
import os

import joblib


def save_artifact(out_dir, model, preprocessor, meta):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    preproc_path = os.path.join(out_dir, "preprocessor.pkl")
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)
    meta["model_path"] = "model.pkl"
    meta["preprocessor_path"] = "preprocessor.pkl"
    meta["created_at"] = datetime.datetime.utcnow().isoformat()
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
    return out_dir


def load_artifact(run_dir):
    with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = joblib.load(os.path.join(run_dir, meta["model_path"]))
    preprocessor = joblib.load(os.path.join(run_dir, meta["preprocessor_path"]))
    return model, preprocessor, meta


def identify_model_type(meta):
    if meta.get("framework") == "torch" or meta.get("data_type") == "image":
        return "image_dl"
    return "tabular_ml"