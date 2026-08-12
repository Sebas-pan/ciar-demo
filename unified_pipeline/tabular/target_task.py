import pandas as pd

TARGET_HINTS = ("target", "label", "class", "y", "clase", "etiqueta", "objetivo")


def resolve_target(df, target_cfg):
    if isinstance(target_cfg, str) and target_cfg not in ("auto", ""):
        if target_cfg not in df.columns:
            raise ValueError(
                "target column '{}' not found in dataframe".format(target_cfg))
        return target_cfg
    for hint in TARGET_HINTS:
        for col in df.columns:
            if col.strip().lower() == hint:
                return col
    return df.columns[-1]


def detect_task(y):
    if (pd.api.types.is_object_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 10 and pd.api.types.is_integer_dtype(y):
        return "classification"
    return "regression"