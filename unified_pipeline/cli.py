import sys

from .config import load_config
from .data.type_detector import detect_data_type
from .images.pipeline import run_image
from .tabular.pipeline import run_tabular


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="unified_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full training + XAI pipeline")
    run.add_argument("--config", required=True, help="Path to the YAML config")
    run.add_argument("--verbose", action="store_true", help="Print extended detail")

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
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
    print("Splits (train/val/test): {} / {} / {}".format(
        summary["split_shapes"]["train"],
        summary["split_shapes"]["val"],
        summary["split_shapes"]["test"]))
    print("--- Metrics ---")
    for key, value in summary["metrics"].items():
        if key in ("confusion_matrix", "classification_report"):
            continue
        if isinstance(value, float):
            print("  {}: {:.4f}".format(key, value))
        elif isinstance(value, int):
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