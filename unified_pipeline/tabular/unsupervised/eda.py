"""Unsupervised EDA — no target column."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...io import write_json


def run_eda_unsupervised(df, out_dir):
    summary = {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_total": int(df.isnull().sum().sum()),
        "missing_per_column": {col: int(v) for col, v in df.isnull().sum().items() if v > 0},
        "duplicated_rows": int(df.duplicated().sum()),
        "numeric_summary": {},
        "outliers_iqr": {},
    }
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col]
        summary["numeric_summary"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((series < lo) | (series > hi)).sum())
        summary["outliers_iqr"][col] = outliers

    # Save JSON
    write_json(os.path.join(out_dir, "eda.json"), summary)

    # Optional: missing bar plot if needed (skip to keep minimal)
    return summary
