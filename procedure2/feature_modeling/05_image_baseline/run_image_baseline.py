#!/usr/bin/env python3
"""Run trajectory-image descriptor nested baselines.

26 of the 172 accepted samples have no trajectory PNG (see
`procedure2/preprocess/artifacts/sample_warnings.csv`, all `missing_png`),
and all 4 of user5's samples are among them, so `sample_splits.csv` is
filtered down to the samples `02_feature_extraction` could actually build an
image for before running nested CV. That leaves the user5 outer fold with an
empty test set (0 macro-F1 by definition, not a modeling failure) since
user5 has no image data at all.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from ml_utils import load_feature_csv, load_sample_splits, run_nested_cv  # noqa: E402


def main() -> int:
    feature_dir = SCRIPT_PATH.parents[1] / "02_feature_extraction" / "artifacts"
    fold_dir = SCRIPT_PATH.parents[1] / "01_fold_assignments" / "artifacts"
    output_dir = SCRIPT_PATH.parent / "artifacts"
    feature_table = load_feature_csv(feature_dir / "image_features.csv")
    splits = load_sample_splits(fold_dir / "sample_splits.csv")
    splits = [row for row in splits if row["sample_id"] in feature_table.X_by_id]
    run_nested_cv(
        baseline_name="trajectory_image_descriptors",
        feature_table=feature_table,
        splits=splits,
        output_dir=output_dir,
    )
    print(f"Wrote image baseline artifacts to {output_dir}")
    print(f"Samples with images: {len(feature_table.X_by_id)} of 172 accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
