"""
main.py — Vision Martian orchestrator.

Run with:
    uv run martian main.py
    uv run martian serve

Five sections, each a @martian.section, calling deeply modular
@martian.capture functions from the modules/ package:

    1. Data Factory       — synthetic geometric-shape dataset generation
    2. Vision Features    — classical CV: Sobel edges, histograms
    3. Neural Network     — ShapeCNN definition, training, LR scheduling
    4. Evaluation         — confusion matrix, first-layer filters, saliency
    5. Activation Maps    — hook-based feature map visualisation

Every meaningful function is captured; helper/debug utilities are @skip'd.
"""

import os
import martianbook as martian

# ── ensure artifact dir exists before any module touches it ───────────────────
os.makedirs(".martian/artifacts", exist_ok=True)

# ── import modules ────────────────────────────────────────────────────────────
from modules import data_factory as df
from modules import vision_features as vf
from modules import model as mdl
from modules import evaluation as ev
from modules import activations as act


# =============================================================================
# SECTION 1 — Data Factory
# =============================================================================

@martian.capture
def generate_data() -> dict:
    """
    Generates 300 synthetic 32×32 RGB images (100 per class) using
    matplotlib patch rendering with randomised position, size, colour,
    and Gaussian noise.  The three classes are: circle, triangle, square.

    All generation is deterministic given the fixed seed, so results are
    fully reproducible across runs.
    """
    dataset = df.generate_dataset(n_samples=300, seed=42)
    imgs, lbls = dataset["images"], dataset["labels"]
    print(f"Dataset shape — images: {imgs.shape}  labels: {lbls.shape}")
    print(f"Pixel range   — min: {imgs.min():.3f}  max: {imgs.max():.3f}")
    print(f"Class counts  — " +
          ", ".join(f"{n}: {int((lbls==i).sum())}" for i, n in enumerate(df.CLASSES)))
    return dataset


@martian.capture
def visualise_dataset(dataset: dict) -> None:
    """
    Produces two diagnostic plots:
      • Sample Grid      — a 3×6 mosaic of randomly-selected images per class
      • Class Distribution — bar chart confirming balanced class counts

    These are the first sanity-checks before any processing begins.
    """
    df.plot_sample_grid(dataset, n_per_class=6)
    df.plot_class_distribution(dataset)


# =============================================================================
# SECTION 2 — Vision Features
# =============================================================================

@martian.capture
def extract_edges(dataset: dict) -> dict:
    """
    Applies a Sobel edge detector (3×3 gradient kernels in X and Y via
    scipy convolution) to compute gradient-magnitude maps for each image.

    Circles produce smooth, circular ridges; triangles produce three
    clean line segments; squares produce four axis-aligned edges.
    These structural differences will also emerge in the CNN's first layer.
    """
    edge_data = vf.compute_edge_maps(dataset, n_samples=9)
    return edge_data


@martian.capture
def visualise_edges(edge_data: dict, dataset: dict) -> None:
    """
    Three diagnostic plots for classical feature analysis:
      • Edge Comparison    — side-by-side RGB | Sobel for 4 samples/class
      • Mean Edge Maps     — class-averaged boundary signatures
      • Pixel Histograms   — per-channel intensity distributions per class

    The pixel histograms reveal why the task is learnable even from raw
    pixels: each shape type has a distinct spectral footprint.
    """
    vf.plot_edge_comparison(edge_data, df.CLASSES)
    vf.plot_mean_edge_maps(edge_data, df.CLASSES)
    vf.plot_pixel_histograms(dataset, df.CLASSES)


# =============================================================================
# SECTION 3 — Neural Network
# =============================================================================

@martian.capture
def build_dataloaders(dataset: dict) -> tuple:
    """
    Splits the dataset 80/20 into train and validation sets using a random
    permutation, wraps them in PyTorch DataLoaders with batch size 32.

    Shuffling is applied only to the training loader — validation order
    doesn't affect metrics but consistent order makes debugging easier.
    """
    train_loader, val_loader = mdl.make_loaders(dataset, val_frac=0.2, batch_size=32)
    return train_loader, val_loader


@martian.capture
def train_neural_network(train_loader, val_loader) -> tuple:
    """
    Trains ShapeCNN for 25 epochs using AdamW (lr=3e-3, wd=1e-4) with
    cosine-annealing LR decay and label-smoothing cross-entropy (ε=0.05).

    ShapeCNN architecture:
      Conv(3→32) → BN → GELU → Pool  ×3 blocks
      Flatten → Linear(2048→256) → GELU → Dropout(0.35) → Linear(256→3)

    Label smoothing prevents overconfident predictions on a small dataset.
    Cosine annealing avoids the need to hand-tune step-based LR schedules.
    """
    trained_model, history = mdl.train_model(
        train_loader, val_loader,
        n_epochs=25, lr=3e-3, weight_decay=1e-4
    )
    mdl.plot_training_curves(history)
    return trained_model, history


# =============================================================================
# SECTION 4 — Evaluation
# =============================================================================

@martian.capture
def run_evaluation(trained_model, val_loader) -> dict:
    """
    Evaluates the trained network on the held-out validation set.
    Computes predictions, overall accuracy, and stores all labels and preds
    for downstream visualisation.  A well-trained model should exceed 90%
    on this separable synthetic task.
    """
    results = ev.evaluate(trained_model, val_loader, df.CLASSES)
    return results


@martian.capture
def visualise_model_internals(trained_model, results: dict, dataset: dict) -> None:
    """
    Three interpretability diagnostics:

      1. Confusion Matrix   — normalised per true class; reveals which
                              shape pairs the model conflates (e.g.
                              square vs triangle share corner features).

      2. First-Layer Filters — the 32 learned 3×3 RGB kernels from Conv1,
                              displayed as RGB patches.  Well-trained
                              filters show oriented edge detectors and
                              colour-selective units.

      3. Gradient Saliency  — ∂loss/∂input maps highlight which pixels
                              most influenced each prediction.  For shapes,
                              expect hot regions along boundary edges.
    """
    ev.plot_confusion_matrix(results, df.CLASSES)
    ev.plot_first_layer_filters(trained_model)
    ev.plot_saliency_maps(trained_model, dataset, df.CLASSES, n_per_class=4)


# =============================================================================
# SECTION 5 — Activation Maps
# =============================================================================

@martian.capture
def visualise_activations(trained_model, dataset: dict) -> None:
    """
    Uses PyTorch forward hooks to record the mean channel activation at
    each of the three convolutional blocks, then plots spatial heatmaps.

    At Block 1 (32→16 px): activations highlight broad colour regions.
    At Block 2 (16→8 px):  activations begin to localise edge junctions.
    At Block 3 (8→4 px):   activations concentrate on the most diagnostic
                            structural feature — the shape's core outline.

    This progression shows the network building a hierarchy of visual
    abstractions, from pixels → edges → shape primitives.
    """
    act.plot_activation_maps(trained_model, dataset, df.CLASSES)


# =============================================================================
# Orchestrator sections
# =============================================================================

@martian.section("① Data Factory")
def section_data():
    """End-to-end synthetic dataset generation and visualisation."""
    dataset = generate_data()
    visualise_dataset(dataset)
    return dataset


@martian.section("② Classical Vision")
def section_vision(dataset: dict):
    """Sobel edge detection and classical feature analysis."""
    edge_data = extract_edges(dataset)
    visualise_edges(edge_data, dataset)


@martian.section("③ Neural Network Training")
def section_training(dataset: dict) -> tuple:
    """Build data loaders, define ShapeCNN, train with AdamW + cosine LR."""
    train_loader, val_loader = build_dataloaders(dataset)
    trained_model, history = train_neural_network(train_loader, val_loader)
    return trained_model, val_loader


@martian.section("④ Evaluation")
def section_evaluation(trained_model, val_loader, dataset: dict):
    """Confusion matrix, learned filter visualisation, and saliency maps."""
    results = run_evaluation(trained_model, val_loader)
    visualise_model_internals(trained_model, results, dataset)
    return results


@martian.section("⑤ Activation Maps")
def section_activations(trained_model, dataset: dict):
    """Hook-based forward-pass activation heatmaps at each conv block."""
    visualise_activations(trained_model, dataset)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Vision Martian — Shape Classification Pipeline")
    print("=" * 60)

    dataset = section_data()
    section_vision(dataset)
    trained_model, val_loader = section_training(dataset)
    results = section_evaluation(trained_model, val_loader, dataset)
    section_activations(trained_model, dataset)

    print("\n" + "=" * 60)
    print(f"  Pipeline complete — val accuracy: {results['accuracy']:.1%}")
    print("  Run:  uv run martian serve")
    print("=" * 60)
