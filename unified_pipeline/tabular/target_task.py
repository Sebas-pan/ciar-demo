import pandas as pd

TARGET_HINTS = ("target", "label", "class", "y", "clase", "etiqueta", "objetivo")

_ALLOWED_TASKS = {"classification", "clasificacion", "regression", "regresion"}

#Target searcher 
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
    raise ValueError(
        "No se ha podido encontrar la columna objetivo (target) en el dataframe."
        "Por favor, especifiquela en $data.target"
    )
    # return df.columns[-1]


# def detect_task(y):
#     if (pd.api.types.is_object_dtype(y) #text
#             or pd.api.types.is_categorical_dtype(y) #Low/medium/high
#             or pd.api.types.is_bool_dtype(y)):
#         return "classification"
#     nunique = y.nunique(dropna=True)
#     if nunique <= 10 and pd.api.types.is_integer_dtype(y):
#         return "classification"
#     return "regression"


def _detect_task(y):
    if (pd.api.types.is_object_dtype(y)  # text
            or pd.api.types.is_string_dtype(y)  # pandas 3.x usa dtype 'str'
            or pd.api.types.is_categorical_dtype(y)  # Low/medium/high
            or pd.api.types.is_bool_dtype(y)):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 10 and pd.api.types.is_integer_dtype(y):
        return "classification"
    return "regression"


def resolve_task(cfg, y):
    task = (cfg.get("supervised") or {}).get("task", "auto")
    task = str(task).strip().lower()
    if task in ("auto", ""):
        return _detect_task(y)
    if task in ("classification", "clasificacion"):
        return "classification"
    if task in ("regression", "regresion"):
        return "regression"
    raise ValueError(
        "Unknown supervised task '{}': use one of {} or 'auto'".format(
            task, ", ".join(sorted(_ALLOWED_TASKS))))