"""
modules/evaluation.py — Model evaluation, feature map visualisation, and saliency.

Three windows into the trained network:
  1. Confusion matrix — where does it stumble?
  2. First-layer filter visualisation — what did it learn to detect?
  3. Gradient saliency maps — which pixels drove each prediction?
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ARTIFACT_DIR = ".martian/artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def evaluate(model, val_loader, class_names: list) -> dict:
    """
    Runs inference on the validation set and returns a results dict with
    all_preds, all_labels, and overall accuracy.  Used by downstream plots.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Val accuracy: {accuracy:.1%}  ({int(accuracy*len(all_labels))}/{len(all_labels)})")
    return {"preds": all_preds, "labels": all_labels, "accuracy": accuracy}


def plot_confusion_matrix(results: dict, class_names: list) -> None:
    """
    Saves a colour-coded confusion matrix for the validation predictions.
    Diagonal cells show correct classifications; off-diagonal cells reveal
    which class pairs the model confuses most often.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    cm = confusion_matrix(results["labels"], results["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#0a0a14")
    ax.set_facecolor("#0a0a14")
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            raw = cm[i, j]
            color = "white" if val > 0.5 else "#94a3b8"
            ax.text(j, i, f"{val:.2f}\n({raw})", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color="#94a3b8", fontsize=9)
    ax.set_yticklabels(class_names, color="#94a3b8", fontsize=9)
    ax.set_xlabel("Predicted", color="#64748b", fontsize=9)
    ax.set_ylabel("True", color="#64748b", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix — Accuracy {results['accuracy']:.1%}",
                 color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(f"{ARTIFACT_DIR}/07_confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved confusion matrix → {ARTIFACT_DIR}/07_confusion_matrix.png")


# ─── First-layer filters ──────────────────────────────────────────────────────

def plot_first_layer_filters(model) -> None:
    """
    Visualises the 32 learned convolutional filters from the network's
    first layer.  Each filter is a 3×3 RGB kernel that acts as a
    directional edge detector, colour selector, or texture probe.
    After training, these filters should show structured patterns rather
    than random noise — a sign the network learned meaningful features.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    weights = model.features[0].weight.data.cpu().numpy()  # (32, 3, 3, 3)
    n_filters = weights.shape[0]
    n_cols = 8
    n_rows = n_filters // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.1, n_rows * 1.1),
                             facecolor="#0a0a14")
    for idx in range(n_filters):
        ax = axes[idx // n_cols][idx % n_cols]
        f = weights[idx].transpose(1, 2, 0)  # (3,3,3) → HWC
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, interpolation="nearest")
        ax.axis("off")

    plt.suptitle("Learned First-Layer Conv Filters (32 × 3×3 RGB)",
                 color="#e2e8f0", fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.2)
    fig.savefig(f"{ARTIFACT_DIR}/08_conv_filters.png", dpi=180,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved conv filters → {ARTIFACT_DIR}/08_conv_filters.png")


# ─── Gradient saliency ────────────────────────────────────────────────────────

def compute_saliency(model, images_chw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Computes vanilla gradient saliency maps for a batch of images.

    Gradient saliency: ∂loss/∂input — which pixels, if perturbed slightly,
    would change the model's prediction the most?  High-gradient regions
    are the areas the model 'looks at' when it classifies.

    Returns: (N, H, W) float32 saliency maps, max-normalised per image.
    """
    model.eval()
    # Create x on the target device directly so it stays a leaf tensor,
    # then call retain_grad() so .grad is populated even in eval mode.
    x = torch.tensor(images_chw, dtype=torch.float32).to(DEVICE)
    x.requires_grad_(True)
    x.retain_grad()
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    with torch.enable_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

    saliency = x.grad.abs().cpu().numpy()        # (N,3,H,W)
    saliency = saliency.max(axis=1)              # (N,H,W) — max over channels
    # normalise each map independently
    mn = saliency.min(axis=(1, 2), keepdims=True)
    mx = saliency.max(axis=(1, 2), keepdims=True)
    return (saliency - mn) / (mx - mn + 1e-8)


def plot_saliency_maps(model, dataset: dict, class_names: list, n_per_class: int = 4) -> None:
    """
    For each class, picks n_per_class correctly-classified examples and
    plots the original image alongside its gradient saliency map.
    Hot regions in the saliency map show where the network 'looked'.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images, labels = dataset["images"], dataset["labels"]
    n_classes = len(class_names)
    accent = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]

    fig, axes = plt.subplots(
        n_classes, n_per_class * 2,
        figsize=(n_per_class * 2 * 1.3, n_classes * 1.45 + 0.6),
        facecolor="#0a0a14"
    )

    for cls in range(n_classes):
        cls_imgs = images[labels == cls][:n_per_class]
        cls_lbls = np.full(len(cls_imgs), cls, dtype=np.int64)
        sal = compute_saliency(model, cls_imgs, cls_lbls)

        for col in range(len(cls_imgs)):
            ax_img = axes[cls][col * 2]
            ax_sal = axes[cls][col * 2 + 1]

            ax_img.imshow(cls_imgs[col].transpose(1, 2, 0), interpolation="nearest")
            ax_img.axis("off")
            ax_sal.imshow(sal[col], cmap="magma", interpolation="bilinear")
            ax_sal.axis("off")

            if col == 0:
                ax_img.set_ylabel(class_names[cls], color=accent[cls],
                                  fontsize=9, fontweight="bold")

    # headers
    for col in range(n_per_class):
        axes[0][col * 2].set_title("input", color="#94a3b8", fontsize=7, pad=2)
        axes[0][col * 2 + 1].set_title("saliency", color="#fb923c", fontsize=7, pad=2)

    plt.suptitle("Gradient Saliency Maps — Where the Network Looks",
                 color="#e2e8f0", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.25)
    fig.savefig(f"{ARTIFACT_DIR}/09_saliency_maps.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved saliency maps → {ARTIFACT_DIR}/09_saliency_maps.png")