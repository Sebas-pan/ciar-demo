import os

TABULAR_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".npy"}


def _dir_has_images(dir_path):
    """True if dir_path contains any image file (case-insensitive)."""
    try:
        for f in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, f)) \
                    and os.path.splitext(f)[1].lower() in IMAGE_EXT:
                return True
    except OSError:
        return False
    return False


def detect_data_type(source):
    if source is None:
        raise ValueError("Could not detect data type: source is None")
    source = str(source).strip()
    if not source:
        raise ValueError("Could not detect data type: empty source")
    if source.startswith("sklearn:"):
        return "tabular"
    if os.path.isdir(source):
        # Inspect directory contents instead of assuming image.
        try:
            entries = os.listdir(source)
        except OSError:
            raise ValueError("Could not detect data type for source: {}".format(source))
        # YOLO layout: data.yaml or <split>/images
        if "data.yaml" in entries:
            return "image"
        for sub in ("train", "valid", "val", "test"):
            if os.path.isdir(os.path.join(source, sub, "images")):
                return "image"
            # Split classification layout: <split>/<class>/*.jpg
            # (e.g. train/Cat, valid/Dog)
            split_dir = os.path.join(source, sub)
            if os.path.isdir(split_dir):
                try:
                    for cls in os.listdir(split_dir):
                        cls_dir = os.path.join(split_dir, cls)
                        if os.path.isdir(cls_dir) and _dir_has_images(cls_dir):
                            return "image"
                except OSError:
                    pass
        # Flat ImageFolder layout: <class>/*.jpg (e.g. PetImages Cat/Dog)
        for e in entries:
            if os.path.isdir(os.path.join(source, e)) and _dir_has_images(
                    os.path.join(source, e)):
                return "image"
        # Tabular folder: any tabular file inside
        for e in entries:
            if os.path.splitext(e)[1].lower() in TABULAR_EXT:
                return "tabular"
        raise ValueError(
            "Could not detect data type for directory '{}': "
            "expected data.yaml or <split>/images (image) or tabular files; "
            "set data_type explicitly".format(source))
    ext = os.path.splitext(source)[1].lower()
    if ext in TABULAR_EXT:
        return "tabular"
    if ext in IMAGE_EXT:
        return "image"
    raise ValueError("Could not detect data type for source: {}".format(source))