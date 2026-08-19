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