import os

TABULAR_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".npy"}


def detect_data_type(source):
    source = str(source)
    if source.startswith("sklearn:"):
        return "tabular"
    if os.path.isdir(source):
        return "image"
    ext = os.path.splitext(source)[1].lower()
    if ext in TABULAR_EXT:
        return "tabular"
    if ext in IMAGE_EXT:
        return "image"
    raise ValueError("Could not detect data type for source: {}".format(source))