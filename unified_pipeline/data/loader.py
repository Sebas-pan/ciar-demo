import os

import pandas as pd
from sklearn.datasets import load_breast_cancer


class TabularBundle:
    def __init__(self, df, target, feature_names, source, description):
        self.df = df
        self.target = target
        self.feature_names = feature_names
        self.source = source
        self.description = description

#Cambiar si tienes un dataset del modulo de scikit-learn 
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
def _load_sklearn(name):
    if name != "breast_cancer":
        raise ValueError("Unsupported sklearn dataset: {}".format(name))
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return TabularBundle(
        df=df,
        target="target",
        feature_names=list(data.feature_names),
        source="sklearn:" + name,
        description="Breast Cancer Wisconsin (diagnostic)",
    )


def _load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported tabular file extension: {}".format(ext))
    return df


def load_dataset(cfg):
    source = cfg["data"]["source"]
    if source.startswith("sklearn:"):
        return _load_sklearn(source.split(":", 1)[1])
    df = _load_file(source)
    return TabularBundle(
        df=df,
        target=None,
        feature_names=None,
        source=source,
        description=os.path.basename(source),
    )