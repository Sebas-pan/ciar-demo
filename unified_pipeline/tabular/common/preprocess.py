"""Shared preprocessing for both supervised and unsupervised tabular branches."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train):
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # bools are numeric for scaling, not categorical
    numeric_cols = sorted(set(numeric_cols) | set(bool_cols))
    categorical_cols = [c for c in X_train.columns
                        if c not in numeric_cols
                        and pd.api.types.is_datetime64_any_dtype(X_train[c])
                        and not np.issubdtype(X_train[c].dtype, np.datetime64)]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(X_train)
    return preprocessor, numeric_cols, categorical_cols


def transformed_feature_names(preprocessor):
    return list(preprocessor.get_feature_names_out())
