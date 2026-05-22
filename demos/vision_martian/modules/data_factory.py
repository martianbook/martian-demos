"""
modules/data_factory.py — Synthetic image dataset generator.

Generates a mini dataset of geometric shapes (circles, triangles, squares)
with noise, used as a toy classification benchmark.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon

ARTIFACT_DIR = ".martian/artifacts"
IMG_SIZE = 32   # pixels per side
N_SAMPLES = 300 # total samples
CLASSES = ["circle", "triangle", "square"]
SEED = 42


def _draw_circle(ax, rng):
    cx, cy = rng.uniform(0.25, 0.75, 2)
    r = rng.uniform(0.15, 0.30)
    color = rng.choice(["#7dd3fc", "#a78bfa", "#f9a8d4"])
    ax.add_patch(plt.Circle((cx, cy), r, color=color, zorder=2))


def _draw_triangle(ax, rng):
    cx, cy = rng.uniform(0.3, 0.7, 2)
    r = rng.uniform(0.18, 0.32)
    color = rng.choice(["#6ee7b7", "#fcd34d", "#fb923c"])
    tri = RegularPolygon((cx, cy), numVertices=3, radius=r,
                         orientation=rng.uniform(0, np.pi),
                         color=color, zorder=2)
    ax.add_patch(tri)


def _draw_square(ax, rng):
    side = rng.uniform(0.20, 0.38)
    x = rng.uniform(0.15, 0.65)
    y = rng.uniform(0.15, 0.65)
    angle = rng.uniform(-20, 20)
    color = rng.choice(["#f87171", "#34d399", "#818cf8"])
    sq = patches.FancyBboxPatch(
        (x, y), side, side,
        boxstyle="square,pad=0",
        color=color, zorder=2,
        transform=ax.transData
    )
    ax.add_patch(sq)


_DRAWERS = {"circle": _draw_circle, "triangle": _draw_triangle, "square": _draw_square}


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> dict:
    """
    Generates a synthetic dataset of 32×32 RGB images, each containing
    a random geometric shape (circle, triangle, or square) rendered on a
    dark background with Gaussian noise.  Returns a dict with keys
    `images` (N,3,32,32 float32 in [0,1]) and `labels` (N int64).
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    per_class = n_samples // len(CLASSES)

    for class_idx, class_name in enumerate(CLASSES):
        for _ in range(per_class):
            fig, ax = plt.subplots(figsize=(1, 1), dpi=IMG_SIZE)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal"); ax.axis("off")
            bg_gray = rng.uniform(0.05, 0.15)
            fig.patch.set_facecolor((bg_gray, bg_gray, bg_gray))
            ax.set_facecolor((bg_gray, bg_gray, bg_gray))
            _DRAWERS[class_name](ax, rng)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(IMG_SIZE, IMG_SIZE, 4)[:, :, :3]
            plt.close(fig)
            img = buf.astype(np.float32) / 255.0
            noise = rng.normal(0, 0.04, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
            images.append(img.transpose(2, 0, 1))  # CHW
            labels.append(class_idx)

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    idx = rng.permutation(len(images))
    return {"images": images[idx], "labels": labels[idx]}


def plot_sample_grid(dataset: dict, n_per_class: int = 6) -> None:
    """
    Saves a visual grid of sample images (n_per_class per class) to the
    artifact directory, colour-coded by class label.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images, labels = dataset["images"], dataset["labels"]
    fig, axes = plt.subplots(
        len(CLASSES), n_per_class,
        figsize=(n_per_class * 1.4, len(CLASSES) * 1.4 + 0.6),
        facecolor="#0f0f1a"
    )
    border_colors = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]

    for row, (cls_name, bcol) in enumerate(zip(CLASSES, border_colors)):
        cls_imgs = images[labels == row]
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.imshow(cls_imgs[col].transpose(1, 2, 0), interpolation="nearest")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(bcol)
                spine.set_linewidth(2)
                spine.set_visible(True)
            if col == 0:
                ax.set_ylabel(cls_name, color=bcol,
                              fontsize=9, fontweight="bold",
                              labelpad=4)
                ax.yaxis.set_label_position("left")

    plt.suptitle("Synthetic Shape Dataset — Sample Grid",
                 color="#e2e8f0", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    fig.savefig(f"{ARTIFACT_DIR}/01_sample_grid.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    print(f"Saved sample grid → {ARTIFACT_DIR}/01_sample_grid.png")


def plot_class_distribution(dataset: dict) -> None:
    """
    Saves a styled bar chart showing the class balance of the generated
    dataset — a quick sanity check before training begins.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    labels = dataset["labels"]
    counts = [int((labels == i).sum()) for i in range(len(CLASSES))]
    palette = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    bars = ax.bar(CLASSES, counts, color=palette,
                  edgecolor="#1e1e3a", linewidth=1.2, width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom",
                color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution", color="#e2e8f0", fontsize=11, pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    fig.savefig(f"{ARTIFACT_DIR}/02_class_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    print(f"Saved distribution chart → {ARTIFACT_DIR}/02_class_distribution.png")
