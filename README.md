# MartianBook Demos

These demos show MartianBook working across different scenarios —
a simple pipeline, matplotlib artifact capture, and a multi-module project.

---

## ⚠️ You cannot run these directly from this repo

The demos depend on packages like `numpy` and `matplotlib` which are
**not** listed as dependencies of `martianbook` itself. This is intentional —
martianbook never forces its own deps on your project.

If you try to run a demo from inside `~/martianbook` you will get:

```
ModuleNotFoundError: No module named 'numpy'
```

---

## How to run the demos

Create a separate project that installs martianbook alongside the
packages the demos need:

```bash
# 1. Create a new project
uv init my-martian-sandbox
cd my-martian-sandbox

# 2. Set up pyproject.toml
```

```toml
[project]
name = "my-martian-sandbox"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "martianbook",
    "numpy",
    "matplotlib",
]

[tool.uv.sources]
martianbook = { path = "../martianbook", editable = true }
```

```bash
# 3. Install
uv sync

# 4. Copy the demo you want to run into your sandbox
cp -r ../martianbook/demos/multimodule ./multimodule

# 5. Run it
uv run martian multimodule/run.py
uv run martian serve
```

---

## Demos

### `basic/pipeline.py`

A simple linear data pipeline across a single file.
Shows `@martian.capture`, `@martian.skip`, and `@martian.section`.
Good starting point.

**Requires:** nothing beyond martianbook itself.

```bash
uv run martian basic/pipeline.py
```

### `plotting/charts.py`

Three matplotlib charts saved as artifacts and embedded inline
in the exported MartianBook HTML.
Shows how artifact detection and image embedding work.

**Requires:** `numpy`, `matplotlib`

```bash
uv run martian plotting/charts.py
uv run martian export
open martianbook.html
```

### `multimodule/run.py`

A project split across three modules in a `modules/` subdirectory.
Shows that `@martian.capture` works across files — functions in
`ingestor.py`, `processor.py`, and `visualizer.py` all appear in
the same MartianBook with correct module attribution and call tree.

**Requires:** `numpy`, `matplotlib`

```bash
uv run martian multimodule/run.py --inspect
uv run martian serve
```

### `vision_martian/main.py`

A full modular computer vision + neural network pipeline spanning
multiple files and execution stages.

Generates a synthetic geometric shape dataset, applies classical
computer vision (Sobel edge detection), trains a CNN, and visualizes
what the network learned using saliency maps, learned filters, and
forward-hook activation maps.

Shows that MartianBook can scale beyond small scripts into a realistic
multi-module ML workflow with artifact-heavy reports.

**Requires:** `torch`, `torchvision`, `numpy`, `matplotlib`,
`scikit-learn`, `scipy`, `Pillow`

```bash
uv run martian vision_martian/main.py
uv run martian serve
```

Expected output:

- sample dataset grid
- class distribution chart
- Sobel edge comparisons
- mean edge maps
- RGB histograms
- training curves
- confusion matrix
- learned convolution filters
- gradient saliency maps
- activation heatmaps

**10 embedded artifact plots total**


---

## What each demo tests

| Demo | Decorators | Cross-file | Artifacts | Scale |
|---|---:|---:|---:|---:|
| `basic/pipeline.py` | ✓ | ✗ | ✗ | minimal |
| `plotting/charts.py` | ✓ | ✗ | ✓ PNG | small |
| `multimodule/run.py` | ✓ | ✓ 3 modules | ✓ PNG | medium |
| `vision_martian/main.py` | ✓ | ✓ 5 modules | ✓ 10 plots | large ML |
