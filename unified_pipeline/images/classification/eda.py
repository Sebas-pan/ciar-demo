"""EDA for classification (no boxes)."""
import os
from collections import Counter
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ...io import write_json


def run_eda_classification(bundles, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_counts = {}
    names = bundles[0].names if bundles else []
    for b in bundles:
        cnt = Counter(b.labels)
        split_counts[b.split] = {b.names[i]: cnt.get(i,0) for i in range(b.nc)}
    # image size stats from train
    train = next((b for b in bundles if b.split=="train"), bundles[0])
    widths, heights = [], []
    for p in train.images[:200]:
        try:
            with Image.open(p) as im:
                w,h = im.size
                widths.append(w); heights.append(h)
        except Exception:
            pass
    size_stats = {}
    if widths:
        size_stats = {"count": len(widths), "min_w": min(widths), "max_w": max(widths), "min_h": min(heights), "max_h": max(heights), "mean_w": round(sum(widths)/len(widths),1), "mean_h": round(sum(heights)/len(heights),1)}
    summary = {"splits": {b.split: len(b.images) for b in bundles}, "class_counts_per_split": split_counts, "image_size_stats": size_stats, "names": names, "nc": len(names)}
    write_json(os.path.join(out_dir, "eda_classification.json"), summary)

    # plot class distribution
    try:
        splits = list(split_counts.keys())
        all_classes = names
        fig, ax = plt.subplots(figsize=(max(6,2*len(all_classes)),4))
        width = 0.8 / max(len(splits),1)
        for i, split in enumerate(splits):
            vals = [split_counts[split].get(cls,0) for cls in all_classes]
            xs = [j + i*width for j in range(len(all_classes))]
            ax.bar(xs, vals, width, label=split)
        ax.set_xticks([j+width*(len(splits)-1)/2 for j in range(len(all_classes))])
        ax.set_xticklabels(all_classes)
        ax.set_ylabel("Images")
        ax.set_title("Class distribution per split")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=100)
        plt.close(fig)
    except Exception:
        pass

    # sample grid without boxes
    try:
        import matplotlib.pyplot as plt
        cols=3
        n=6
        rows=(n+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
        axes = axes.flatten()
        for idx in range(min(n, len(train.images))):
            p = str(train.images[idx])
            lab = train.names[train.labels[idx]] if idx < len(train.labels) else ""
            img = Image.open(p).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(f"{os.path.basename(p)[:18]} ({lab})", fontsize=8)
        for idx in range(min(n, len(train.images)), len(axes)):
            axes[idx].axis("off")
        fig.suptitle("Sample train images")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sample_grid.jpg"), dpi=100)
        plt.close(fig)
    except Exception:
        pass

    return summary
