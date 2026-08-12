# AGENTS.md

## Repo shape
- Not a git repo. No README, configs, tests, or CI. The project is 3 Jupyter notebooks + a `unified_pipeline/` package + `requirements.txt`.
- All data is sklearn's built-in `breast_cancer` (loaded in-notebook); there are no data files or loaders.

## Environment (Windows)
- Use the repo-root venv: `.venv\Scripts\python` (Python 3.14.3). Notebook kernel: `.venv (3.14.3)`.
- `requirements.txt` is **UTF-16 LE encoded** — Read/linter tools may report it as binary. Read/parse it with pandas or `codecs.open('utf-16')`; never rewrite it to UTF-8 without intent.

## Execution flow / entrypoints
- `unified_pipeline/` — parametrized train+explain pipeline, driven by CLI + YAML config:
  `.venv\Scripts\python -m unified_pipeline run --config configs/tabular.yaml`.
  Config keys: `data.source` (`sklearn:breast_cancer` or a path), `model.type` (`logistic|decision_tree|random_forest|svm|xgboost`), `split.train_val_test`, `xai.*`. Outputs land in `unified_pipeline/outputs/runs/<timestamp>/` (EDA, metrics, `model/` artifact with `metadata.json` + preprocessor, `xai/` with SHAP/LIME/feature-importance, `report.md`).
  SHAP 0.52 returns classifier SHAP as `(n, features, n_classes)` — the code slices `arr[..., -1]`.
  `xgboost` is optional and guarded in `tabular/models.py`.
- `models/RForest.ipynb` — load data, 80/20 stratified split (`test_size=0.2, random_state=42, stratify=y`), train
  `RandomForestClassifier(300, gini, max_features='sqrt', random_state=42, n_jobs=-1)`, save with `joblib.dump("./saved_models/RF_model.pkl")` (path is notebook-relative → `models/saved_models/`).
- `xai/lime.ipynb` — loads model as `../models/saved_models/RF_model.pkl`, explains individual predictions with LIME.
- `models/test.ipynb` — audit; loads `models/saved_models/RF_model.pkl` (repo-root relative). Model paths are inconsistent across notebooks — match the notebook's existing convention.

## Known issues / gotchas
- `LimeTabularExplainer(training_data=X_train, ...)` with a pandas DataFrame raises `InvalidIndexError`; pass numpy (`.values`). The explainer cell in `lime.ipynb` still contains this bug (the `unified_pipeline/xai/lime_local.py` is the correct pattern).
- `xai/lime.ipynb` is ~3.5 MB / ~74k lines: saved outputs embed d3 webpack JS. Prefer targeted source-only edits (grep `"source"`, skim via offset reads) over full-file reads/writes.
- No tests or lint config; "verification" = running notebook cells with the venv kernel, or `python -m unified_pipeline run --config configs/tabular.yaml`.