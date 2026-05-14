import os, shutil, random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CLASSES     = ["wrinkles", "dark spots", "puffy eyes", "clear skin"]
RAW_DIR     = "DATASET"
DATASET_DIR = "dataset_split"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

def create_folders():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            Path(os.path.join(DATASET_DIR, split, cls)).mkdir(parents=True, exist_ok=True)
    print("[✓] Folders created")

def split_dataset():
    summary = {}
    for cls in CLASSES:
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(cls_path):
            print(f"[!] Not found: {cls_path}")
            continue
        imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                if Path(f).suffix.lower() in IMAGE_EXTS]
        random.shuffle(imgs)
        n       = len(imgs)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits  = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train+n_val],
            "test":  imgs[n_train+n_val:]
        }
        summary[cls] = {s: len(v) for s, v in splits.items()}
        for split, files in splits.items():
            for fp in files:
                shutil.copy2(fp, os.path.join(DATASET_DIR, split, cls))
        print(f"  ✅ {cls}: {n} images split")
    plot_distribution(summary)

def plot_distribution(summary):
    os.makedirs("logs", exist_ok=True)
    x = np.arange(len(CLASSES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (split, color) in enumerate(zip(["train","val","test"],
                                            ["#3B82F6","#10B981","#EF4444"])):
        counts = [summary.get(cls, {}).get(split, 0) for cls in CLASSES]
        bars = ax.bar(x + i*w, counts, w, label=split.title(), color=color, alpha=0.85)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(c), ha="center", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([c.title() for c in CLASSES])
    ax.set_title("DermalScan – Class Distribution")
    ax.set_ylabel("Number of Images")
    ax.legend()
    plt.tight_layout()
    plt.savefig("logs/class_distribution.png", dpi=150)
    plt.close()
    print("[✓] Plot saved → logs/class_distribution.png")

if __name__ == "__main__":
    create_folders()
    split_dataset()
    print("[✓] Dataset split complete!")