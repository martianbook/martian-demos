"""
demos/multimodule/modules/ingestor.py
--------------------------------------
Responsible for loading and validating raw data.
"""

import martianbook as martian


@martian.capture
def load_csv(path: str) -> dict:
    """
    Loads a CSV file from the given path and returns a
    summary dict with row count, column count, and path.
    Simulates real IO with printed progress.
    """
    print(f"[ingestor] Opening {path}...")
    print("[ingestor] Reading headers...")
    print("[ingestor] Counting rows...")
    rows = 2400
    cols = 12
    print(f"[ingestor] Found {rows} rows × {cols} columns.")
    return {"path": path, "rows": rows, "cols": cols}


@martian.capture
def validate_schema(dataset: dict) -> dict:
    """
    Validates that required columns are present and
    that row count meets the minimum threshold.
    Raises ValueError if validation fails.
    """
    print(f"[ingestor] Validating schema for {dataset['path']}...")
    required_cols = ["id", "timestamp", "value", "label"]
    print(f"[ingestor] Checking required columns: {required_cols}")
    # simulate all present
    print("[ingestor] All required columns present. ✓")
    print(f"[ingestor] Row count {dataset['rows']} >= minimum 100. ✓")
    return {**dataset, "valid": True}