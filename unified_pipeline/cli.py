import sys

from .config import load_config
from .data.type_detector import detect_data_type
from .images.pipeline import run_image
from .tabular.supervised.pipeline import run_tabular


def main(argv=None):
    import argparse #uso de comandos como input para el pipeline

    parser = argparse.ArgumentParser(prog="unified_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full training + XAI pipeline")
    run.add_argument("--config", required=True, help="Path to the YAML config")

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        # Unsupervised tabular takes precedence if configured
        unsup_task = (cfg.get("unsupervised") or {}).get("task")
        if unsup_task not in (None, "", "auto"):
            from .tabular.unsupervised.pipeline import run_clustering
            summary = run_clustering(cfg)
        else:
            data_type = detect_data_type(cfg["data"]["source"])
            if data_type == "tabular":
                summary = run_tabular(cfg)
            else:
                summary = run_image(cfg)
        _print_summary(summary)


def _print_summary(summary):
    print("\n=== UNIFIED PIPELINE: {} | {} | {} ===".format(
        summary["data_type"], summary["task"], summary["model"]))
    print("Source: {} | Target: {}".format(summary["source"], summary["target"]))
    # Device banner — show CUDA vs CPU prominently
    dev = summary.get("device")
    if dev:
        try:
            import torch
            if "cuda" in str(dev).lower() and torch.cuda.is_available():
                try:
                    name = torch.cuda.get_device_name(torch.device(dev) if isinstance(dev, str) and ":" in dev else 0)
                    print("Device: {} ({}) | CUDA available: True".format(dev, name))
                except Exception:
                    print("Device: {} | CUDA available: True".format(dev))
            else:
                print("Device: {} | CUDA available: {}".format(dev, torch.cuda.is_available() if 'torch' in sys.modules else "unknown"))
        except Exception:
            print("Device: {}".format(dev))
    else:
        # Fallback for pipelines that didn't expose device
        try:
            import torch
            print("Device: CUDA available: {} (no explicit device in summary)".format(torch.cuda.is_available()))
        except Exception:
            pass
    splits = summary.get("split_shapes", {})
    # Support both 'val' and 'valid' keys (classification used 'valid' before)
    train_n = splits.get("train", "?")
    val_n = splits.get("val", splits.get("valid", "?"))
    test_n = splits.get("test", "?")
    print("Splits (train/val/test): {} / {} / {}".format(train_n, val_n, test_n))
    print("--- Metrics ---")
    for key, value in summary["metrics"].items():
        if key in ("confusion_matrix", "classification_report", "cluster_sizes"):
            if key == "cluster_sizes":
                print("  {}: {}".format(key, value))
            continue
        if isinstance(value, float):
            print("  {}: {:.4f}".format(key, value))
        elif isinstance(value, int):
            print("  {}: {}".format(key, value))
        elif isinstance(value, dict):
            print("  {}: {}".format(key, value))
    print("--- XAI ---")
    for key, value in summary["xai"].items():
        if value is None or value == {}:
            status = "skipped"
        elif "error" in value:
            status = "ERROR: {}".format(value["error"])
        else:
            status = "OK"
        print("  {}: {}".format(key, status))
    print("--- Output ---")
    print("  run_dir: {}".format(summary["run_dir"]))


if __name__ == "__main__":
    main()