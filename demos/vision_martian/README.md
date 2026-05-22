# Vision Martian 🔭

A **modular computer vision + neural network pipeline** built for [MartianBook](https://martianbook.dev).  
Generates a synthetic shape dataset, runs classical edge detection, trains a tiny CNN, then explains what the network learned — all rendered as a gorgeous interactive execution report.

---

## What it does

| Section | Module | Artifacts |
|---------|--------|-----------|
| ① Data Factory | `modules/data_factory.py` | Sample grid, class distribution chart |
| ② Classical Vision | `modules/vision_features.py` | Edge comparison, mean edge maps, pixel histograms |
| ③ Neural Network | `modules/model.py` | Training loss & accuracy curves |
| ④ Evaluation | `modules/evaluation.py` | Confusion matrix, learned filters, saliency maps |
| ⑤ Activation Maps | `modules/activations.py` | Per-block forward-hook heatmaps |

**10 artifact plots** total, all embedded inline in the MartianBook HTML report.

---

## Quick start

```bash
# 1 — install deps (first run only, ~2 min)
uv sync

# 2 — run the pipeline
uv run martian main.py

# 3 — open the report in your browser
uv run martian serve

# 4 — (optional) export a standalone HTML file
uv run martian export -o vision_report.html
```

Add the alias to your shell for convenience:
```bash
alias martian="uv run martian"
```

---

## Project structure

```
vision_martian/
├── main.py                  ← orchestrator + @martian.section wiring
├── pyproject.toml           ← uv project (all deps declared here)
├── modules/
│   ├── __init__.py
│   ├── data_factory.py      ← synthetic dataset generation
│   ├── vision_features.py   ← Sobel edges, histograms
│   ├── model.py             ← ShapeCNN, training loop, AdamW + cosine LR
│   ├── evaluation.py        ← confusion matrix, filter viz, saliency
│   └── activations.py       ← forward-hook activation maps
└── .martian/
    ├── report.json          ← execution IR (auto-generated)
    └── artifacts/           ← all 10 PNG plots (cleared each run)
```

---

## ShapeCNN architecture

```
Input (3, 32, 32)
  │
  ├─ Conv2d(3→32, 3×3) → BatchNorm → GELU → MaxPool  →  (32, 16, 16)
  ├─ Conv2d(32→64, 3×3) → BatchNorm → GELU → MaxPool  →  (64, 8, 8)
  ├─ Conv2d(64→128, 3×3) → BatchNorm → GELU → MaxPool →  (128, 4, 4)
  │
  └─ Flatten → Linear(2048→256) → GELU → Dropout(0.35) → Linear(256→3)

~360K parameters  |  AdamW lr=3e-3  |  Cosine LR  |  Label smoothing ε=0.05
```

Expected val accuracy: **≥ 90%** after 25 epochs on CPU (~60 s).

---

## Extending this project

- **Real images**: swap `data_factory.generate_dataset()` for a torchvision dataset loader.
- **Deeper network**: add more blocks to `ShapeCNN.features` — the hook-based activation visualiser requires no changes.
- **More augmentation**: add `torchvision.transforms` inside `make_loaders()`.
- **GradCAM**: replace vanilla gradient saliency in `evaluation.py` with a GradCAM implementation using the existing `ActivationRecorder` pattern.

---

Built with MartianBook 0.2.2 · PyTorch · scikit-learn · matplotlib
