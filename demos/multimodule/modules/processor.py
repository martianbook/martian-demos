"""
demos/multimodule/modules/processor.py
---------------------------------------
Responsible for cleaning, transforming, and engineering features.
"""

import martianbook as martian


@martian.capture
def clean(dataset: dict) -> dict:
    """
    Removes null rows, deduplicates records, and strips
    whitespace from string columns. Reports rows removed.
    """
    print(f"[processor] Cleaning {dataset['rows']} rows...")
    removed_nulls = 38
    removed_dupes = 12
    final_rows    = dataset["rows"] - removed_nulls - removed_dupes
    print(f"[processor] Removed {removed_nulls} null rows.")
    print(f"[processor] Removed {removed_dupes} duplicate rows.")
    print(f"[processor] {final_rows} rows remaining after cleaning.")
    return {**dataset, "rows": final_rows, "cleaned": True}


@martian.capture
def engineer_features(dataset: dict) -> dict:
    """
    Derives new features from existing columns:
    rolling averages, lag features, and normalized value scores.
    Prints a summary of features added.
    """
    print(f"[processor] Engineering features on {dataset['rows']} rows...")
    features_added = ["value_rolling_7d", "value_lag_1", "value_normalized"]
    for f in features_added:
        print(f"[processor]   + {f}")
    print(f"[processor] {len(features_added)} features engineered.")
    return {
        **dataset,
        "cols": dataset["cols"] + len(features_added),
        "features": features_added,
    }


@martian.skip
def _debug_print_sample(dataset: dict) -> None:
    """Internal debug helper — never shown in MartianBook."""
    print(f"DEBUG SAMPLE: {dataset}")