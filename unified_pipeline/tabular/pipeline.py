import datetime
import os

from ..data.loader import load_dataset
from ..data.validation import validate
from ..io import write_json
from ..modelstore import save_artifact
from ..xai.feature_importance import global_importance, plot_importance
from ..xai.lime_local import run_lime
from ..xai.report import build_report, write_report
from ..xai.shap_local_global import run_shap
from .eda import run_eda
from .evaluate import evaluate
from .persist import build_tabular_metadata
from .preprocess import build_preprocessor, transformed_feature_names
from .split import split_by_task
from .target_task import resolve_target, resolve_task
from .trainer import train_estimator


def run_tabular(cfg):
    bundle = load_dataset(cfg)
    df = bundle.df

    validation_report, warnings = validate(df)
    target = resolve_target(df, cfg["data"].get("target", "auto"))
    task = resolve_task(cfg, df[target])
    split_cfg = cfg["split"]
    parts = split_by_task(df, target, task, split_cfg)

    preprocessor, _, _ = build_preprocessor(parts["X_train"])
    X_train_t = preprocessor.transform(parts["X_train"])
    X_test_t = preprocessor.transform(parts["X_test"])
    X_val_t = preprocessor.transform(parts["X_val"])
    feature_names = transformed_feature_names(preprocessor)

    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "random_forest")
    initial_params = model_cfg.get("params") or {}
    search = model_cfg.get("search")
    estimator, best_params, val_metrics = train_estimator(
        model_type=model_type,
        task=task,
        params=initial_params,
        search=search,
        X_train=X_train_t,
        y_train=parts["y_train"],
        X_val=X_val_t,
        y_val=parts["y_val"],
    )
    if best_params is None:
        best_params = initial_params

    metrics = evaluate(estimator, task, X_test_t, parts["y_test"])
    classes = (
        [str(c) for c in estimator.classes_] #take each class and convert it to string
        if task == "classification"
        else None
    )

    outputs = cfg["output"]
    run_dir = os.path.join(
        outputs["dir"],
        "runs",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    run_eda(df, target, run_dir)

    meta = build_tabular_metadata(
        bundle=bundle,
        task=task,
        model_type=model_type,
        hyperparameters=best_params,
        feature_names=feature_names,
        classes=classes,
        metrics=metrics,
        split_cfg=split_cfg,
        val_metrics=val_metrics,
    )
    if outputs.get("save_model", True):
        artifact_dir = os.path.join(run_dir, "model")
        save_artifact(artifact_dir, estimator, preprocessor, meta)
    else:
        write_json(os.path.join(run_dir, "metadata.json"), meta)

    xai_dir = os.path.join(run_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)
    xai_cfg = cfg["xai"]
    xai = {"feature_importance": None, "shap": None, "lime": None}

    if xai_cfg.get("feature_importance", True):
        try:
            table, method = global_importance(estimator, feature_names)
            plot_importance(
                table, os.path.join(xai_dir, "feature_importance.png"))
            write_json(
                os.path.join(xai_dir, "feature_importance.json"),
                table.to_dict(orient="records"))
            xai["feature_importance"] = {"method": method, "saved": True}
        except Exception as exc:
            xai["feature_importance"] = {"error": str(exc)}

    if xai_cfg.get("shap", True):
        try:
            shap_out = run_shap(
                model=estimator,
                X_train=X_train_t,
                X_test=X_test_t,
                feature_names=feature_names,
                out_dir=xai_dir,
            )
            xai["shap"] = {"saved": True, "instances": shap_out["instances"]}
        except Exception as exc:
            xai["shap"] = {"error": str(exc)}

    if xai_cfg.get("lime", True):
        try:
            lime_out = run_lime(
                model=estimator,
                X_train=X_train_t,
                y_train=parts["y_train"],
                X_test=X_test_t,
                y_test=parts["y_test"],
                feature_names=feature_names,
                class_names=classes,
                num_features=xai_cfg.get("num_features", 10),
                num_instances=xai_cfg.get("num_instances", 3),
                random_state=split_cfg.get("random_state", 42),
                out_dir=xai_dir,
                mode=task,
            )
            xai["lime"] = {"saved": True, "instances": lime_out}
        except Exception as exc:
            xai["lime"] = {"error": str(exc)}

    instances = build_report(
        model=estimator,
        task=task,
        X_test=X_test_t,
        y_test=parts["y_test"],
        class_names=classes,
        num_instances=xai_cfg.get("num_instances", 3),
        xai=xai,
    )
    if outputs.get("save_report", True):
        write_report(
            instances, xai_dir, run_dir,
            metrics=metrics, task=task, model_type=model_type, target=target,
            top_n=int(xai_cfg.get("top_features", 15)),
            narrative=bool(xai_cfg.get("narrative", True)),
        )

    return {
        "data_type": "tabular",
        "task": task,
        "model": model_type,
        "target": target,
        "source": bundle.source,
        "split_shapes": {
            "train": int(X_train_t.shape[0]),
            "val": int(len(parts["X_val"])),
            "test": int(X_test_t.shape[0]),
        },
        "metrics": metrics,
        "val_metrics": val_metrics,
        "xai": xai,
        "run_dir": run_dir,
        "validation": validation_report,
        "warnings": warnings,
    }