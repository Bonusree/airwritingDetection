#!/usr/bin/env python3
"""Run flattened 128-step sequence nested baselines."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from ml_utils import load_augmented_sequence_npz, load_sample_splits, load_sequence_npz, run_nested_cv  # noqa: E402


def main() -> int:
    repo_root = SCRIPT_PATH.parents[3]
    feature_dir = SCRIPT_PATH.parents[1] / "01_feature_extraction" / "artifacts"
    output_dir = SCRIPT_PATH.parent / "artifacts"
    feature_table = load_sequence_npz(feature_dir / "sequence_features_128.npz")
    augmented = load_augmented_sequence_npz(feature_dir / "augmented_sequence_features_128.npz")
    splits = load_sample_splits(repo_root / "procedure1" / "artifacts" / "sample_splits.csv")
    run_nested_cv(
        baseline_name="sequence_128_flattened_with_train_augmentation",
        feature_table=feature_table,
        augmented=augmented,
        splits=splits,
        output_dir=output_dir,
        model_configs=[{"model": "nearest_centroid"}],
    )
    print(f"Wrote sequence baseline artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
