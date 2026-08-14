import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import yaml

TABULAR_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".npy"}


@dataclass
class ImageSplitBundle:
    images: List[Path]  # rutas absolutas a imágenes
    labels: List[Path]  # rutas absolutas a etiquetas .txt
    nc: int  # número de clases
    names: List[str]  # nombres de clase ['Cat', 'Dog', ...]
    split: str  # 'train' | 'valid' | 'test'


def detect_data_type(source: str) -> str:
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


def _read_data_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_yolo_data_spec(dataset_dir: str) -> dict:
    """Construye el dict de datos absoluto para la detección.

    Los data.yaml de Roboflow suelen usar rutas relativas poco fiables
    (p. ej. '../train/images'), por lo que aquí se resuelve a partir del
    layout físico: <dataset_dir>/<split>/images (split= train/valid/test).
    nc y names se toman del data.yaml del dataset si existe.
    """
    base = Path(dataset_dir).resolve()
    yaml_path = base / "data.yaml"

    names = ["class_0", "class_1"]
    nc = 0
    if yaml_path.is_file():
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        nc = int(raw.get("nc", 0))
        raw_names = raw.get("names")
        if isinstance(raw_names, list):
            names = [str(n) for n in raw_names]
        elif isinstance(raw_names, dict):
            names = [str(raw_names[k]) for k in sorted(raw_names)] or names

    spec = {
        "path": str(base),
        "nc": nc,
        "names": names,
    }
    spec_split_names = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    for yaml_key, spec_key in spec_split_names.items():
        img_dir = base / yaml_key / "images"
        if img_dir.is_dir():
            spec[spec_key] = str(img_dir.resolve())
    if nc == 0:
        # inferir nc desde las anotaciones si no viene en data.yaml
        seen = set()
        for split in ("train", "valid", "test"):
            lbl_dir = base / split / "labels"
            if not lbl_dir.is_dir():
                continue
            for lbl in lbl_dir.glob("*.txt"):
                try:
                    with open(lbl, "r", encoding="utf-8") as fh:
                        for line in fh:
                            parts = line.split()
                            if parts:
                                seen.add(int(float(parts[0])))
                except Exception:
                    pass
        if seen:
            nc = max(seen) + 1
            spec["nc"] = nc
            spec["names"] = names if len(names) >= nc else [f"class_{i}" for i in range(nc)]
    return spec


def write_yolo_data_yaml(data_spec: dict, out_path: str) -> str:
    """Escribe un data.yaml con rutas absolutas usable por ultralytics
    (acepta str en model.train/val, no dict, en versiones actuales)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": data_spec.get("path", ""),
        "nc": data_spec.get("nc", 0),
        "names": data_spec.get("names", []),
    }
    for key in ("train", "val", "valid", "test"):
        if data_spec.get(key):
            payload[key] = data_spec[key]
    if "val" not in payload and "valid" in payload:
        payload["val"] = payload["valid"]
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True)
    return str(out)


def _count_labels_per_image(label_dir: Path) -> int:
    """Retorna número de labels (cajas) por imagen como referencia; no estricto."""
    count = 0
    if label_dir.exists():
        for lbl in label_dir.glob("*.txt"):
            try:
                with open(lbl, "r") as f:
                    n = len(f.read().strip().split())
                    if n > 0:
                        count += 1
            except Exception:
                pass
    return count


def _collect_split_images_labels(base_dir: Path, split_name: str) -> Tuple[List[Path], List[Path]]:
    """Colecta imágenes y labels para un split (train/valid/test).
    Asume estructura: base_dir/split/images/*.jpg, *.png, etc.
    y base_dir/split/labels/*.txt correspondientes 1-a-1.
    """
    img_dir = base_dir / split_name / "images"
    lbl_dir = base_dir / split_name / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"No images dir: {img_dir}")

    img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tif", "*.tiff", "*.webp"]
    images: List[Path] = []
    for ext in img_exts:
        images.extend(img_dir.glob(ext))
    # sort for deterministic order
    images = sorted(images, key=lambda p: p.name)

    # Corresponding labels: replace images path with labels path, keep same stem
    labels: List[Path] = []
    for img_path in images:
        lbl_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if lbl_dir.exists() and not lbl_path.exists():
            # fallback: some datasets have .png labels etc.; try to find any label
            found = False
            if lbl_dir.exists():
                for lbl_try in lbl_dir.glob("*.txt"):
                    if lbl_try.stem == img_path.stem:
                        lbl_path = lbl_try
                        found = True
                        break
            if not found:
                lbl_path = Path("")  # placeholder
        labels.append(lbl_path)

    return images, labels


def load_yolo_dataset(dataset_dir: str,
                      splits: Tuple[str, ...] = ("train", "valid", "test")) -> List[ImageSplitBundle]:
    """Carga un dataset YOLO detection.

    dataset_dir: ruta a la carpeta que contiene data.yaml y subcarpetas train/valid/test
    con images/ y labels/.
    Retorna lista de ImageSplitBundle, una por split pedido.
    """
    base = Path(dataset_dir).resolve()

    # Buscar data.yaml: o está dentro de dataset_dir, o el user pasa la ruta absoluta
    yaml_path = None
    candidates = [base / "data.yaml", base.parent / "data.yaml"]
    for c in candidates:
        if c.is_file():
            yaml_path = c
            break
    if yaml_path is None:
        raise FileNotFoundError(
            f"data.yaml not found under {base}. "
            "Place data.yaml at the root of the YOLO dataset directory."
        )

    yaml_data = _read_data_yaml(yaml_path)
    nc = yaml_data.get("nc", 0)
    names = yaml_data.get("names", [f"class_{i}" for i in range(nc)])

    # Asegurar nombres consistentes (si data.yaml tiene nc pero no names, rellenar)
    if len(names) < nc:
        names = names + [f"class_{i}" for i in range(len(names), nc)]

    bundles: List[ImageSplitBundle] = []
    for split in splits:
        images, labels = _collect_split_images_labels(base, split)
        # Filtrar labels vacíos o sin archivo
        valid_imgs = []
        valid_lbls = []
        for img, lbl in zip(images, labels):
            if img == Path("") or not img.exists():
                continue
            # Normalizar label path: quitar placeholder vacío
            lbl_path = Path(str(lbl)) if lbl else Path("")
            if not lbl_path.exists():
                # count empty labels but keep image; o podríamos drop, pero mantengamos para máximo info
                valid_imgs
                # keep image even if label missing
            valid_imgs.append(img)
            valid_lbls.append(lbl_path)
        bundles.append(ImageSplitBundle(
            images=valid_imgs,
            labels=valid_lbls,
            nc=nc,
            names=names,
            split=split,
        ))
    return bundles


# Export bundle for quick use
def quick_bundle(dataset_dir: str, split: str = "train") -> ImageSplitBundle:
    bundles = load_yolo_dataset(dataset_dir, splits=(split,))
    if bundles:
        return bundles[0]
    raise ValueError(f"No bundle for split '{split}' under {dataset_dir}")