"""Persist metadata for classification."""
import datetime

def build_classification_metadata(dataset_dir, data_spec, variant, weights, metrics, train_cfg, device, source, model_type="vgg16"):
    return {
        "version": "1.0",
        "data_type": "image",
        "framework": "torch",
        "task": "classification",
        "model_type": model_type,
        "variant": variant,
        "weights": weights,
        "nc": data_spec.get("nc", 0),
        "names": data_spec.get("names", []),
        "metrics": metrics,
        "train": {k:v for k,v in (train_cfg or {}).items()},
        "device": str(device),
        "dataset_dir": dataset_dir,
        "source": source,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
