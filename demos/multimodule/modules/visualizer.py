"""
demos/multimodule/modules/visualizer.py
----------------------------------------
Responsible for generating visual summaries of the processed data.
"""

import os
import martianbook as martian

os.makedirs(".martian/artifacts", exist_ok=True)


@martian.capture
def plot_value_distribution(dataset: dict) -> None:
    """
    Plots a histogram of the value column distribution
    after cleaning and feature engineering.
    Saves to .martian/artifacts/value_distribution.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"[visualizer] Plotting value distribution for {dataset['rows']} rows...")

    data = np.random.normal(loc=142.7, scale=38.2, size=dataset["rows"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data, bins=50, color="#7dd3fc", edgecolor="#0d0d0f", alpha=0.85)
    ax.axvline(data.mean(), color="#f87171", linewidth=1.5, linestyle="--",
               label=f"Mean: {data.mean():.1f}")
    ax.set_title("Value Distribution (post-processing)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    path = ".martian/artifacts/value_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] Saved → {path}")


@martian.capture
def plot_feature_correlation(dataset: dict) -> None:
    """
    Plots a bar chart showing correlation scores between
    engineered features and the target label.
    Saves to .martian/artifacts/feature_correlation.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    features = dataset.get("features", ["f1", "f2", "f3"])
    print(f"[visualizer] Plotting correlations for {len(features)} features...")

    correlations = np.random.uniform(0.2, 0.85, size=len(features))
    colors = ["#4ade80" if c > 0.5 else "#fbbf24" for c in correlations]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features, correlations, color=colors, edgecolor="#0d0d0f")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Correlation with target")
    ax.set_title("Feature Correlation Scores")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, correlations):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    path = ".martian/artifacts/feature_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] Saved → {path}")