"""
modules/model.py — Tiny convolutional neural network for shape classification.

Defines ShapeCNN, a lightweight 3-conv-block network, plus training and
evaluation logic.  Designed to be interpretable, not merely accurate.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT_DIR = ".martian/artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Architecture ────────────────────────────────────────────────────────────

class ShapeCNN(nn.Module):
    """
    A three-block convolutional network for 32×32 RGB input.

    Block layout:
        Conv2d → BatchNorm → GELU → MaxPool (×3 blocks)
        Flatten → Linear(256) → Dropout(0.35) → Linear(n_classes)

    GELU activations give smoother gradients than ReLU for small networks.
    BatchNorm before activation keeps training stable without LR tuning.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 3→32 channels, 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2 — 32→64 channels, 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3 — 64→128 channels, 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def make_loaders(dataset: dict, val_frac: float = 0.2, batch_size: int = 32):
    """
    Splits the dataset into train / validation and wraps them in DataLoaders.
    Returns (train_loader, val_loader).
    """
    images = torch.tensor(dataset["images"], dtype=torch.float32)
    labels = torch.tensor(dataset["labels"], dtype=torch.long)
    n = len(images)
    n_val = int(n * val_frac)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(images[train_idx], labels[train_idx])
    val_ds   = TensorDataset(images[val_idx],   labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Batch size: {batch_size} | Device: {DEVICE}")
    return train_loader, val_loader


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(
    train_loader,
    val_loader,
    n_epochs: int = 25,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
) -> tuple[ShapeCNN, dict]:
    """
    Trains ShapeCNN with AdamW and cosine-annealing LR scheduling.

    Returns the trained model and a history dict with keys:
      train_loss, val_loss, train_acc, val_acc (all lists, one value/epoch).

    AdamW decouples weight decay from the adaptive LR — important for Conv
    networks with BatchNorm where L2 interacts poorly with scale invariance.
    Cosine annealing smoothly reduces LR over training, improving final
    convergence without any manual LR schedule tuning.
    """
    model = ShapeCNN().to(DEVICE)
    print(f"ShapeCNN — {model.param_count():,} trainable parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += len(xb)
        scheduler.step()

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(xb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(xb)

        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct / v_total)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} — "
                  f"train loss {history['train_loss'][-1]:.4f} "
                  f"acc {history['train_acc'][-1]:.3f} | "
                  f"val loss {history['val_loss'][-1]:.4f} "
                  f"acc {history['val_acc'][-1]:.3f}")

    final_acc = history["val_acc"][-1]
    print(f"\n✓ Training complete — final val accuracy: {final_acc:.1%}")
    return model, history


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """
    Plots training & validation loss and accuracy over epochs.
    The dual-panel chart makes it easy to spot overfitting: loss curves
    diverging while accuracy plateaus is the tell-tale sign.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="#0a0a14")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f0f1a")
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        ax.tick_params(colors="#64748b", labelsize=8)

    # Loss
    ax1.plot(epochs, history["train_loss"], color="#7dd3fc", lw=2, label="train")
    ax1.plot(epochs, history["val_loss"],   color="#f9a8d4", lw=2, linestyle="--", label="val")
    ax1.set_title("Loss", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax1.set_xlabel("epoch", color="#64748b", fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.15, labelcolor="white")

    # Accuracy
    ax2.plot(epochs, [a*100 for a in history["train_acc"]], color="#6ee7b7", lw=2, label="train")
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   color="#fb923c", lw=2, linestyle="--", label="val")
    ax2.set_title("Accuracy (%)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax2.set_xlabel("epoch", color="#64748b", fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.15, labelcolor="white")

    plt.suptitle("ShapeCNN Training Curves", color="#e2e8f0",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{ARTIFACT_DIR}/06_training_curves.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved training curves → {ARTIFACT_DIR}/06_training_curves.png")
