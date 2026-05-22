"""
modules/vision_features.py — Classical computer vision feature extraction.

Applies Sobel edge detection, computes per-channel histograms, and
visualises gradient magnitude maps — all before any neural network runs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

ARTIFACT_DIR = ".martian/artifacts"

# Sobel kernels
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T


def _sobel_magnitude(img_chw: np.ndarray) -> np.ndarray:
    """img_chw: (3,H,W) float32 → (H,W) gradient magnitude."""
    gray = img_chw.mean(axis=0)
    gx = convolve(gray, SOBEL_X)
    gy = convolve(gray, SOBEL_Y)
    return np.sqrt(gx**2 + gy**2)


def compute_edge_maps(dataset: dict, n_samples: int = 9) -> dict:
    """
    Computes Sobel gradient-magnitude edge maps for the first n_samples
    images per class.  Returns a dict mapping class index → list of
    (image_chw, edge_map_hw) tuples.
    """
    images, labels = dataset["images"], dataset["labels"]
    classes = sorted(set(labels.tolist()))
    edge_data = {}
    for cls in classes:
        cls_imgs = images[labels == cls][:n_samples]
        edges = [_sobel_magnitude(img) for img in cls_imgs]
        edge_data[cls] = list(zip(cls_imgs, edges))
    print(f"Computed edge maps for {sum(len(v) for v in edge_data.values())} images "
          f"across {len(classes)} classes.")
    return edge_data


def plot_edge_comparison(edge_data: dict, class_names: list) -> None:
    """
    Saves a side-by-side grid: original image | Sobel edge map, for 4
    samples per class.  Reveals how each shape's boundary signature
    looks to a classical filter bank.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    n_show = 4
    n_classes = len(class_names)
    accent = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]

    fig, axes = plt.subplots(
        n_classes, n_show * 2,
        figsize=(n_show * 2 * 1.3, n_classes * 1.45 + 0.5),
        facecolor="#0a0a14"
    )
    for row, cls in enumerate(sorted(edge_data)):
        samples = edge_data[cls][:n_show]
        for col, (img, edge) in enumerate(samples):
            # original
            ax_img = axes[row][col * 2]
            ax_img.imshow(img.transpose(1, 2, 0), interpolation="nearest")
            ax_img.axis("off")
            # edge map
            ax_edge = axes[row][col * 2 + 1]
            ax_edge.imshow(edge, cmap="inferno", interpolation="nearest")
            ax_edge.axis("off")
            if col == 0:
                ax_img.set_ylabel(class_names[cls], color=accent[row],
                                  fontsize=9, fontweight="bold")
    # column headers
    for col in range(n_show):
        axes[0][col * 2].set_title("RGB", color="#94a3b8", fontsize=7, pad=2)
        axes[0][col * 2 + 1].set_title("Sobel ∇", color="#fb923c", fontsize=7, pad=2)

    plt.suptitle("Classical Edge Detection — RGB vs Sobel Gradient Magnitude",
                 color="#e2e8f0", fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.25)
    fig.savefig(f"{ARTIFACT_DIR}/03_edge_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved edge comparison → {ARTIFACT_DIR}/03_edge_comparison.png")


def plot_mean_edge_maps(edge_data: dict, class_names: list) -> None:
    """
    Averages all edge maps within each class and plots them side by side.
    The mean edge map is the 'canonical boundary signature' of each shape
    — useful for understanding what a linear classifier would see.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    n_classes = len(class_names)
    accent = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]
    cmaps = ["cool", "summer", "spring"]

    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes * 2.8, 2.8),
                             facecolor="#0a0a14")
    for cls in sorted(edge_data):
        edges = np.stack([e for _, e in edge_data[cls]])
        mean_edge = edges.mean(axis=0)
        ax = axes[cls]
        im = ax.imshow(mean_edge, cmap=cmaps[cls], interpolation="bilinear")
        ax.set_title(f"⌀ {class_names[cls]}", color=accent[cls],
                     fontsize=10, fontweight="bold", pad=5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#94a3b8")

    plt.suptitle("Mean Sobel Edge Maps per Class",
                 color="#e2e8f0", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{ARTIFACT_DIR}/04_mean_edge_maps.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved mean edge maps → {ARTIFACT_DIR}/04_mean_edge_maps.png")


def plot_pixel_histograms(dataset: dict, class_names: list) -> None:
    """
    Plots per-class pixel intensity histograms across the R, G, B channels.
    Dense shapes push brightness right; backgrounds stay near zero — this
    distribution gap is exactly what the neural network will exploit.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images, labels = dataset["images"], dataset["labels"]
    channel_colors = ["#f87171", "#6ee7b7", "#60a5fa"]
    channel_names = ["R", "G", "B"]
    accent = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]

    fig, axes = plt.subplots(1, len(class_names),
                             figsize=(len(class_names) * 3.5, 3),
                             facecolor="#0a0a14")
    for cls, ax in enumerate(axes):
        ax.set_facecolor("#0f0f1a")
        cls_imgs = images[labels == cls]   # (N,3,H,W)
        for ch in range(3):
            vals = cls_imgs[:, ch, :, :].ravel()
            ax.hist(vals, bins=40, color=channel_colors[ch], alpha=0.65,
                    label=channel_names[ch], density=True)
        ax.set_title(class_names[cls], color=accent[cls],
                     fontsize=10, fontweight="bold")
        ax.tick_params(colors="#475569", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        if cls == 0:
            ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")
        ax.set_xlabel("intensity", color="#64748b", fontsize=8)

    plt.suptitle("Per-Class RGB Pixel Intensity Histograms",
                 color="#e2e8f0", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{ARTIFACT_DIR}/05_pixel_histograms.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved pixel histograms → {ARTIFACT_DIR}/05_pixel_histograms.png")
