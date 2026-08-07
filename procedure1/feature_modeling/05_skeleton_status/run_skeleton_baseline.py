#!/usr/bin/env python3
"""Run skeleton-summary nested baselines."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from ml_utils import load_feature_csv, load_sample_splits, run_nested_cv  # noqa: E402


def main() -> int:
    repo_root = SCRIPT_PATH.parents[3]
    output_dir = SCRIPT_PATH.parent / "artifacts"
    feature_table = load_feature_csv(output_dir / "skeleton_features.csv")
    splits = load_sample_splits(repo_root / "procedure1" / "artifacts" / "sample_splits.csv")
    run_nested_cv(
        baseline_name="skeleton_landmark_summary",
        feature_table=feature_table,
        splits=splits,
        output_dir=output_dir,
    )
    print(f"Wrote skeleton baseline artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
