# AGENTS.md

## Repo shape
- Git repo (`Explainable_AI_CIAR`). No tests or CI. The project is `unified_pipeline/` (a parametrized train+explain package) + `configs/` + a couple of exploratory notebooks in `models/` and `xai/` + `requirements.txt`.
- Two pipeline branches driven by the same CLI (dispatch via `detect_data_type` on `data.source`):
  - **Tabular** (sklearn ML + SHAP/LIME/feature-importance) — `data.source` = `sklearn:breast_cancer` or a tabular file path.
  - **Image** (YOLO11 detection + Grad-CAM/Occlusion/LIME-Image) — `data.source` = a YOLO-format dataset dir.

## Environment (Windows)
- Use the repo-root venv: `.venv\Scripts\python` (Python 3.14.3). Notebook kernel: `.venv (3.14.3)`.
- `requirements.txt` is **UTF-16 LE encoded** — Read/linter tools may report it as binary. Read/parse it with pandas or `codecs.open('utf-16')`; never rewrite it to UTF-8 without intent.
- Deep-learning stack IS installed in the venv: `torch 2.11.0+cu128`, `torchvision`, `ultralytics 8.4.118`, `opencv-python`. RTX 5070 (Blackwell sm_120) requires the **cu128** torch build — cu121/cu124 do NOT support sm_120. The venv is Python 3.14 (cp314 wheels exist for torch cu128).

## Execution flow / entrypoints
- `unified_pipeline/` — parametrized train+explain pipeline, driven by CLI + YAML config:
  - Tabular: `.venv\Scripts\python -m unified_pipeline run --config configs/tabular.yaml`
  - Image: `.venv\Scripts\python -m unified_pipeline run --config configs/image_detection.yaml` (train+val on `CatsDogs_YOLO11`)
  - Outputs land in `unified_pipeline/outputs/runs/<timestamp>/`:
    - Image branch: `eda_image/`, `data.yaml` (absolute), `train/`, `val/` (mAP metrics + curves), `model/best.pt` + `metadata.json`, `predictions/`, `xai/` (gradcam/occlusion/lime_image PNG + npy), `report.md`.
    - Tabular branch: EDA, `model/` artifact (metadata.json + model.pkl + preprocessor.pkl), `xai/`, `report.md`.
- `data/image_loader.py`: `build_yolo_data_spec()` resolves the dataset by physical layout `<dir>/<split>/images` (Roboflow `data.yaml` `../train/images` paths are unreliable). `write_yolo_data_yaml()` writes an absolute data.yaml — ultralytics requires a **str path**, NOT a dict, in `model.train/val(data=...)` for this version.
- YOLO11 weights are named `yolo11n.pt` (NOT `yolov11n.pt`). `images/models.py` downloads variants to `unified_pipeline/weights/` using absolute paths (ultralytics' global `weights_dir` is set elsewhere on this machine and is not reliable).
- SHAP 0.52 returns classifier SHAP as `(n, features, n_classes)` — the code slices `arr[..., -1]`.
- `xgboost` is optional and guarded in `tabular/models.py`.

## XAI image internals (gotchas)
- `xai/gradcam_image.py` (detection-aware Grad-CAM): the YOLO11 `Detect` head creates tensors under `torch.inference_mode()` (crashes if you wrap the full forward in `torch.enable_grad`; `.detach()` does NOT clear the inference flag — use a fresh `torch.empty_like.copy_`). It captures head input features via a forward hook under `inference_mode`, then builds the CAM directly on `head.cv3` (no backward pass) by **confidence-weighted activation**: `cam = ReLU(sum_c(w_c·act)) · sigmoid(logits_clase)` with `w_c = mean_spatial(sigmoid(logits_clase)·act)` per scale. (A previous BCE-gradient + `abs()` variant inverted the attribution — background lit red, object yellow.) Without detections, the class is inferred from global class-logit sums. `images/trainer.py:resolve_device` normalizes numeric device strings like `"0"` to `int` because raw `torch .to('0')` raises `Invalid device string`.
- `xai/occlusion_image.py` and `xai/lime_image.py` are model-agnostic. LIME's classifier = aggregated per-class detection confidence (temp files are written for inference because `predict_image` takes a path; cleaned up by the pipeline).

## Known issues / gotchas
- `LimeTabularExplainer(training_data=X_train, ...)` with a pandas DataFrame raises `InvalidIndexError`; pass numpy (`.values`). The explainer cell in `lime.ipynb` still contains this bug (the `unified_pipeline/xai/lime_local.py` is the correct pattern).
- `xai/lime.ipynb` is ~3.5 MB / ~74k lines: saved outputs embed d3 webpack JS. Prefer targeted source-only edits (grep `"source"`, skim via offset reads) over full-file reads/writes.
- The dataset has a `masks/` folder (Roboflow leftovers) — ignored for detection. `labels.cache` files are stale numpy caches, harmless. Test splits are FIXED by dataset folders (no random split as in tabular).
- No tests or lint config; "verification" = running notebook cells with the venv kernel, or `python -m unified_pipeline run --config configs/<branch>.yaml`.