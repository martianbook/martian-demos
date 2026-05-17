"""
demos/multimodule/run.py
------------------------
Entry point for the multi-module data pipeline demo.

Imports from three separate modules in modules/:
  - ingestor   → load and validate raw data
  - processor  → clean and engineer features
  - visualizer → generate plots

Run from your sandbox (not from this repo directly — see demos/README.md):
    uv run martian multimodule/run.py
    uv run martian serve
"""

import martianbook as martian

from modules.ingestor   import load_csv, validate_schema
from modules.processor  import clean, engineer_features, _debug_print_sample
from modules.visualizer import plot_value_distribution, plot_feature_correlation


@martian.section("Full Pipeline")
def run():
    """
    Orchestrates the full data pipeline across three modules:
    ingestion, processing, and visualization.
    Each step is captured and linked in MartianBook.
    """
    # --- Ingestion (modules/ingestor.py) ---
    raw   = load_csv("data/sales_2026.csv")
    valid = validate_schema(raw)

    # --- Processing (modules/processor.py) ---
    cleaned = clean(valid)
    _debug_print_sample(cleaned)   # skipped — never appears in MartianBook
    final   = engineer_features(cleaned)

    # --- Visualization (modules/visualizer.py) ---
    plot_value_distribution(final)
    plot_feature_correlation(final)

    return final


if __name__ == "__main__":
    result = run()
    print(f"\nPipeline complete. Final dataset: {result['rows']} rows × {result['cols']} cols.")