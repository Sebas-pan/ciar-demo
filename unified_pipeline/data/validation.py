def validate(df):
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_per_column": {
            col: int(v) for col, v in df.isnull().sum().items() if v > 0
        },
    }
    warnings = []
    if report["missing_total"]:
        warnings.append("Dataset contains missing values")
    if report["duplicated_rows"]:
        warnings.append("Dataset contains duplicated rows")
    return report, warnings