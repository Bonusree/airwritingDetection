#!/usr/bin/env python3
"""Run flattened 256-step sequence nested baselines."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from ml_utils import load_sample_splits, load_sequence_npz, run_nested_cv  # noqa: E402


def main() -> int:
    feature_dir = SCRIPT_PATH.parents[1] / "02_feature_extraction" / "artifacts"
    fold_dir = SCRIPT_PATH.parents[1] / "01_fold_assignments" / "artifacts"
    output_dir = SCRIPT_PATH.parent / "artifacts"
    feature_table = load_sequence_npz(feature_dir / "sequence_features_256.npz")
    splits = load_sample_splits(fold_dir / "sample_splits.csv")
    run_nested_cv(
        baseline_name="sequence_256_flattened",
        feature_table=feature_table,
        splits=splits,
        output_dir=output_dir,
        # High-dimensional flattened input (256*11=2816-D): nearest centroid only,
        # matching procedure1's choice to skip ridge/kNN grid search here.
        model_configs=[{"model": "nearest_centroid"}],
    )
    print(f"Wrote sequence baseline artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
