import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .persist import write_json


def run_eda(df, target, out_dir):
    features = df.drop(columns=[target])
    y = df[target]
    summary = {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_total": int(df.isnull().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "target_type": str(y.dtype),
        "target_distribution": {
            str(k): int(v) for k, v in y.value_counts().items()
        },
    }
    outliers = {}
    for col in features.select_dtypes(include="number").columns:
        q1, q3 = features[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[col] = int(((features[col] < lo) | (features[col] > hi)).sum())
    summary["outliers_iqr"] = outliers
    write_json(os.path.join(out_dir, "eda.json"), summary)
    return summary