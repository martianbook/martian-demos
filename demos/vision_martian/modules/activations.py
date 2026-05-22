"""
modules/activations.py — Feature map visualisation via forward hooks.

Registers PyTorch forward hooks to intercept intermediate activations,
then plots the activation heatmaps at each convolutional block.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT_DIR = ".martian/artifacts"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivationRecorder:
    """
    Attaches forward hooks to named layers and records their output tensors.

    Usage:
        recorder = ActivationRecorder(model, ["features.3", "features.7"])
        with recorder:
            model(x)
        maps = recorder.activations
    """
    def __init__(self, model, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook


def plot_activation_maps(model, dataset: dict, class_names: list) -> None:
    """
    For one representative image per class, plots the mean activation map
    at each of the three convolutional blocks (after GELU, before pooling).

    Mean-pooling across the channel dimension collapses the feature map
    cube to a single spatial heatmap — hot regions indicate where the
    network 'activated' most strongly at each resolution level.
    The resolution halves at each block (32 → 16 → 8 → 4 px).
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    images, labels = dataset["images"], dataset["labels"]

    # Layers: index 2 = GELU after block1, 6 = after block2, 10 = after block3
    layer_names = ["features.2", "features.6", "features.10"]
    layer_labels = ["Block 1 (32×32→16×16)", "Block 2 (16×16→8×8)", "Block 3 (8×8→4×4)"]
    accent = ["#7dd3fc", "#6ee7b7", "#f9a8d4"]
    n_classes = len(class_names)
    n_layers = len(layer_names)

    fig, axes = plt.subplots(
        n_classes, n_layers + 1,
        figsize=((n_layers + 1) * 2, n_classes * 2 + 0.5),
        facecolor="#0a0a14"
    )

    for cls in range(n_classes):
        # pick first sample of this class
        img = images[labels == cls][0]
        x = torch.tensor(img[None], dtype=torch.float32).to(DEVICE)

        recorder = ActivationRecorder(model, layer_names)
        model.eval()
        with recorder:
            with torch.no_grad():
                model(x)

        # col 0: original image
        ax0 = axes[cls][0]
        ax0.imshow(img.transpose(1, 2, 0), interpolation="nearest")
        ax0.axis("off")
        ax0.set_ylabel(class_names[cls], color=accent[cls],
                       fontsize=9, fontweight="bold")

        # cols 1…: activation heatmaps
        for col, lname in enumerate(layer_names):
            act = recorder.activations[lname][0]  # (C, H, W)
            heatmap = act.mean(0).numpy()          # (H, W)
            ax = axes[cls][col + 1]
            ax.imshow(heatmap, cmap="inferno", interpolation="bilinear")
            ax.axis("off")

    # column headers
    axes[0][0].set_title("input", color="#94a3b8", fontsize=8, pad=3)
    for col, ll in enumerate(layer_labels):
        axes[0][col + 1].set_title(ll, color="#fb923c", fontsize=7.5, pad=3)

    plt.suptitle("Conv Activation Maps — Mean Channel Response per Block",
                 color="#e2e8f0", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    fig.savefig(f"{ARTIFACT_DIR}/10_activation_maps.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"Saved activation maps → {ARTIFACT_DIR}/10_activation_maps.png")
