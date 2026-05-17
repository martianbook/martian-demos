"""
demos/plotting/charts.py
------------------------
Demonstrates MartianBook artifact capture with matplotlib.
Three plots are generated and embedded inline in the exported HTML.

Run from your sandbox (not from this repo directly — see demos/README.md):
    uv run martian plotting/charts.py
    uv run martian export
    open martianbook.html
"""

import os
import martianbook as martian

os.makedirs(".martian/artifacts", exist_ok=True)


@martian.capture
def generate_distribution():
    """
    Generates a normal distribution and plots a histogram.
    Saves to .martian/artifacts/distribution.png.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.random.normal(loc=100, scale=15, size=1000)

    print(f"Generated {len(data)} samples")
    print(f"Mean:  {data.mean():.2f}")
    print(f"Std:   {data.std():.2f}")
    print(f"Min:   {data.min():.2f}")
    print(f"Max:   {data.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=40, color="#7dd3fc", edgecolor="#0d0d0f", alpha=0.85)
    ax.set_title("Distribution of Generated Data")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)

    path = ".martian/artifacts/distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return {"mean": float(data.mean()), "std": float(data.std()), "n": len(data)}


@martian.capture
def generate_scatter(stats: dict):
    """
    Generates a scatter plot of two correlated variables.
    Saves to .martian/artifacts/scatter.png.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 300
    x = np.random.normal(0, 1, n)
    y = 0.7 * x + np.random.normal(0, 0.5, n)
    corr = float(np.corrcoef(x, y)[0, 1])
    print(f"Correlation: {corr:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.5, color="#c4b5fd", edgecolors="#7c3aed", linewidths=0.4, s=30)
    ax.set_title(f"Scatter Plot (r = {corr:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.spines[["top", "right"]].set_visible(False)

    path = ".martian/artifacts/scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return {"correlation": corr, "n": n}


@martian.capture
def generate_timeseries():
    """
    Generates a synthetic time series (random walk) and plots it.
    Saves to .martian/artifacts/timeseries.png.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps  = 200
    series = np.cumsum(np.random.randn(steps))

    print(f"Steps:      {steps}")
    print(f"Final val:  {series[-1]:.3f}")
    print(f"Peak:       {series.max():.3f}")
    print(f"Trough:     {series.min():.3f}")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series, color="#4ade80", linewidth=1.5)
    ax.axhline(0, color="#444", linewidth=0.5, linestyle="--")
    ax.fill_between(range(steps), series, 0, where=(series > 0), alpha=0.15, color="#4ade80")
    ax.fill_between(range(steps), series, 0, where=(series < 0), alpha=0.15, color="#f87171")
    ax.set_title("Synthetic Time Series (Random Walk)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.spines[["top", "right"]].set_visible(False)

    path = ".martian/artifacts/timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return {"steps": steps, "final": float(series[-1])}


@martian.section("Plotting Mission")
def run():
    """
    Runs three visualization functions and captures their plots
    as artifacts linked to the functions that produced them.
    """
    stats   = generate_distribution()
    scatter = generate_scatter(stats)
    ts      = generate_timeseries()
    return {"distribution": stats, "scatter": scatter, "timeseries": ts}


if __name__ == "__main__":
    run()