"""
demos/basic/pipeline.py
-----------------------
A simple linear data pipeline instrumented with MartianBook.

Run with:
    uv run martian pipeline.py
    uv run martian serve
"""

import martianbook as martian


@martian.capture
def load_data(path: str):
    """Loads raw CSV data and performs an initial row count."""
    print(f"Loading dataset from {path}...")
    print("1200 rows found across 8 columns.")
    return {"rows": 1200, "columns": 8}


@martian.capture
def clean_data(dataset: dict):
    """Removes null rows and validates required columns."""
    print(f"Cleaning {dataset['rows']} rows...")
    print("Removed 14 null rows. 1186 rows remaining.")
    return {**dataset, "rows": dataset["rows"] - 14}


@martian.capture
def compute_statistics(dataset: dict):
    """Computes descriptive statistics across numeric columns."""
    print("Mean price: 142.7")
    print("Std dev:    38.2")
    return {"mean": 142.7, "std": 38.2}


@martian.skip
def internal_debug_helper():
    print("DEBUG: this should not be captured")


@martian.section("Full Pipeline")
def run_pipeline():
    """Orchestrates the full pipeline from ingestion to output."""
    data  = load_data("data/raw.csv")
    clean = clean_data(data)
    stats = compute_statistics(clean)
    internal_debug_helper()
    return stats


if __name__ == "__main__":
    run_pipeline()