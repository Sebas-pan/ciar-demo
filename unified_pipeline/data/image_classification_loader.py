"""Loader for image classification (ImageFolder) datasets.

Supports:
  - Flat layout: <root>/Cat/*.jpg , <root>/Dog/*.jpg  (PetImages)
  - Split layout: <root>/train/Cat/*.jpg , <root>/valid/... , <root>/test/...
Falls back to flat + stratified 70/15/15 split if no split subdirs.

Usage: build_classification_spec + load_classification_dataset + get_dataloaders
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import random

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class ClassificationSplitBundle:
    images: List[Path]
    labels: List[int]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    names: List[str]
    nc: int
    split: str


def _scan_flat_dataset(root: Path) -> Tuple[List[Path], List[int], Dict[str,int], List[str]]:
    """Scan flat Cat/Dog layout: root/<class>/*.jpg"""
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    # ignore hidden / non-class dirs (e.g. masks, labels leftover)
    valid = []
    for d in class_dirs:
        # check if contains images
        has_img = any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png"))
        if has_img:
            valid.append(d)
    class_names = sorted([d.name for d in valid])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    images = []
    labels = []
    for cname in class_names:
        cdir = root / cname
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in cdir.glob(ext):
                images.append(p)
                labels.append(class_to_idx[cname])
        # also search case-insensitive? keep simple
    return images, labels, class_to_idx, class_names


def _scan_split_dataset(root: Path, split: str) -> Tuple[List[Path], List[int], Dict[str,int], List[str]]:
    split_dir = root / split
    if not split_dir.is_dir():
        return [], [], {}, []
    # ImageFolder style: split/class/*.jpg
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs if any(d.glob("*.jpg")) or any(d.glob("*.png"))])
    if not class_names:
        return [], [], {}, []
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    images = []
    labels = []
    for cname in class_names:
        cdir = split_dir / cname
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in cdir.glob(ext):
                images.append(p)
                labels.append(class_to_idx[cname])
    return images, labels, class_to_idx, class_names


def build_classification_spec(dataset_dir: str, split_ratios=(0.7,0.15,0.15), seed=42) -> Dict:
    base = Path(dataset_dir).resolve()
    # Check if split layout exists
    has_split = (base/"train").is_dir() and any((base/"train").iterdir())
    if has_split:
        total = 0
        nc = 0
        names = []
        for s in ("train","valid","test","val"):
            imgs, labs, c2i, nms = _scan_split_dataset(base, s)
            if imgs:
                total += len(imgs)
                if not names:
                    names = nms
                    nc = len(nms)
        return {"path": str(base), "nc": nc, "names": names, "layout": "split", "total": total}
    # flat
    imgs, labs, c2i, names = _scan_flat_dataset(base)
    return {"path": str(base), "nc": len(names), "names": names, "layout": "flat", "total": len(imgs)}


def load_classification_dataset(dataset_dir: str, split_ratios=(0.7,0.15,0.15), seed=42, valid_name="valid"):
    """Return list of ClassificationSplitBundle for train/valid/test.

    For flat datasets, does stratified shuffle split.
    For split datasets, loads each split folder directly (maps valid<->val).
    """
    base = Path(dataset_dir).resolve()
    # Try split layout first
    split_candidates = ["train","valid","val","test"]
    has_split = (base/"train").is_dir()
    if has_split:
        bundles = []
        for split in ("train","valid","test"):
            # map valid<->val
            actual = split
            if split=="valid" and not (base/"valid").is_dir() and (base/"val").is_dir():
                actual = "val"
            imgs, labs, c2i, names = _scan_split_dataset(base, actual)
            if not imgs and split=="valid":
                # try val
                imgs, labs, c2i, names = _scan_split_dataset(base, "val")
            if imgs:
                idx_to_class = {v:k for k,v in c2i.items()}
                bundles.append(ClassificationSplitBundle(
                    images=imgs, labels=labs, class_to_idx=c2i, idx_to_class=idx_to_class,
                    names=names, nc=len(names), split=split
                ))
        if bundles:
            return bundles

    # Flat -> stratified split
    images, labels, class_to_idx, names = _scan_flat_dataset(base)
    if not images:
        raise FileNotFoundError(f"No images found in {base} (expected <class>/*.jpg or train/<class>/*.jpg)")
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    import numpy as np
    labels_np = np.array(labels)
    # first split train vs temp
    train_ratio, valid_ratio, test_ratio = split_ratios
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(images, labels_np))
    # temp -> valid/test
    valid_test_ratio = valid_ratio + test_ratio
    if valid_ratio==0:
        valid_idx = np.array([], dtype=int)
        test_idx = temp_idx
    else:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/valid_test_ratio, random_state=seed)
        # need to map temp_idx to local indices
        temp_labels = labels_np[temp_idx]
        # create dummy indices for split
        local_valid, local_test = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
        valid_idx = temp_idx[local_valid]
        test_idx = temp_idx[local_test]

    def bundle_from_indices(idxs, split_name):
        imgs = [images[i] for i in idxs]
        labs = [labels[i] for i in idxs]
        return ClassificationSplitBundle(
            images=imgs, labels=labs, class_to_idx=class_to_idx, idx_to_class=idx_to_class,
            names=names, nc=len(names), split=split_name
        )
    bundles = [
        bundle_from_indices(train_idx, "train"),
        bundle_from_indices(valid_idx, "valid"),
        bundle_from_indices(test_idx, "test"),
    ]
    return bundles
