import os
from collections import Counter

from PIL import Image

from ..io import write_json

_CLASS_COLORS = [
    (0, 255, 0),   # green
    (255, 0, 0),   # blue
    (0, 0, 255),   # red
    (255, 255, 0), # cyan
    (0, 255, 255), # yellow
    (255, 0, 255), # magenta
]


def _parse_label_file(path):
    ids = []
    if not path or not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                try:
                    ids.append(int(float(parts[0])))
                except ValueError:
                    pass
    return ids


def _parse_label_boxes(path, w, h):
    """Retorna lista de (cid, x1, y1, x2, y2) en píxeles desde un label YOLO."""
    boxes = []
    if not path or not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((
                cid,
                int((xc - bw / 2) * w), int((yc - bh / 2) * h),
                int((xc + bw / 2) * w), int((yc + bh / 2) * h),
            ))
    return boxes


def run_eda(bundles, out_dir):
    """EDA del dataset YOLO: distribución por clase y split, stats de tamaño,
    grid de muestras con cajas (train). Guarda eda.json + PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    matplotlib_agg()

    split_counts = {}
    size_stats = {"count": 0, "min_w": None, "max_w": 0, "min_h": None, "max_h": 0}
    for bundle in bundles:
        counts = Counter()
        for label_path in bundle.labels:
            for cid in _parse_label_file(label_path):
                counts[cid] += 1
        split_counts[bundle.split] = {
            bundle.names[cid] if cid < len(bundle.names) else "class_{}".format(cid): int(cnt)
            for cid, cnt in counts.items()
        }

    # stats de tamaño sobre el primer split (train)
    train = next((b for b in bundles if b.split == "train"), bundles[0])
    widths, heights = [], []
    for img_path in train.images[:200]:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass
    if widths:
        size_stats.update({
            "count": len(widths),
            "min_w": min(widths), "max_w": max(widths),
            "min_h": min(heights), "max_h": max(heights),
            "mean_w": round(sum(widths) / len(widths), 1),
            "mean_h": round(sum(heights) / len(heights), 1),
        })

    summary = {
        "splits": {b.split: len(b.images) for b in bundles},
        "class_counts_per_split": split_counts,
        "image_size_stats": size_stats,
    }
    for b in bundles:
        summary["splits"][b.split] = {"images": len(b.images), "labels": len(b.labels)}
    write_json(os.path.join(out_dir, "eda_image.json"), summary)

    _plot_class_distribution(split_counts, os.path.join(out_dir, "class_distribution.png"))
    _plot_sample_grid(train, os.path.join(out_dir, "sample_grid.jpg"), n=6)
    return summary


def matplotlib_agg():
    import matplotlib

    matplotlib.use("Agg")


def _plot_class_distribution(split_counts, out_path):
    import matplotlib.pyplot as plt

    splits = list(split_counts.keys())
    all_classes = sorted({c for counts in split_counts.values() for c in counts})
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(all_classes)), 4))
    width = 0.8 / max(len(splits), 1)
    for i, split in enumerate(splits):
        vals = [split_counts[split].get(cls, 0) for cls in all_classes]
        xs = [j + i * width for j in range(len(all_classes))]
        ax.bar(xs, vals, width, label=split)
    ax.set_xticks([j + width * (len(splits) - 1) / 2 for j in range(len(all_classes))])
    ax.set_xticklabels(all_classes)
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de instancias por clase y split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_sample_grid(bundle, out_path, n=6):
    import matplotlib.pyplot as plt

    import cv2

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for idx in range(min(n, len(bundle.images))):
        img_path = str(bundle.images[idx])
        label_path = str(bundle.labels[idx]) if idx < len(bundle.labels) else ""
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        for cid, x1, y1, x2, y2 in _parse_label_boxes(label_path, w, h):
            color = _CLASS_COLORS[cid % len(_CLASS_COLORS)]
            name = bundle.names[cid] if cid < len(bundle.names) else "class_{}".format(cid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(y1 - 4, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].axis("off")
        axes[idx].set_title(os.path.basename(img_path)[:24])
    for idx in range(min(n, len(bundle.images)), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Muestras train con cajas (color por clase)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)