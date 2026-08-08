# Feature Modeling (procedure2)

Feature extraction and baseline modeling on top of `procedure2/preprocess/`'s
paper-faithful preprocessing (172 accepted Phase 1 samples: `অ`, `আ`, `ই`,
`ঈ`; causal 8-D features; T=256 arc-length resampling; leave-one-writer-out
folds — see `procedure2/preprocess/preprocess.py`).

Each step is kept in its own numbered subfolder, mirroring
`procedure1/feature_modeling/`.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 procedure2/feature_modeling/01_fold_assignments/build_folds.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure2/feature_modeling/02_feature_extraction/build_features.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure2/feature_modeling/03_summary_baseline/run_summary_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure2/feature_modeling/04_sequence_baseline/run_sequence_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure2/feature_modeling/05_image_baseline/run_image_baseline.py
```

Step folders:

- `01_fold_assignments`: expands the flat leave-one-writer-out folds in
  `procedure2/preprocess/artifacts/lowo_folds.csv` into nested outer/inner
  splits (`nested_fold_assignments.csv`, `sample_splits.csv`), using the same
  count-balanced grouping `procedure1/preprocess_clean_augment.py` uses, so
  every downstream baseline gets model selection folds that never touch the
  outer test writer.
- `02_feature_extraction`: derived-dynamics summary features from the native
  (variable-length) causal trajectories, a `[172, 256, 11]` sequence tensor
  from the resampled trajectories (2 coordinates + normalized stroke id + the
  paper's 8-D causal feature vector), and 224x224 trajectory-image descriptor
  features for the 146 of 172 samples that have a PNG.
- `03_summary_baseline`: NumPy baselines (nearest centroid, ridge, k-NN,
  Gaussian NB) over the 114 derived trajectory/dynamics features.
- `04_sequence_baseline`: nearest-centroid baseline over the flattened
  `[256, 11]` sequence tensor (2,816-D), matching procedure1's choice to skip
  the wider model grid on high-dimensional flattened input.
- `05_image_baseline`: NumPy baselines over trajectory-image descriptors.

As in procedure1, baselines are NumPy-only (no `sklearn`/`torch` in this
environment) and use `common/ml_utils.py` (a self-contained copy of
`procedure1/feature_modeling/common/ml_utils.py`) for the shared nested-CV
harness, metrics, and lightweight model implementations.

## Differences from procedure1

- No augmentation stage exists yet for procedure2 (`procedure2/preprocess.py`
  only implements the paper's preprocessing, not procedure1's separate
  augmentation step), so there are no `augmented_*` counterparts here and the
  baselines train on the outer-fold development set only.
- No skeleton baseline yet (procedure1's `05_skeleton_status`) — re-extracting
  hand landmarks from video is unported.
- 26 of the 172 accepted samples have no trajectory PNG, and **all 4 of
  user5's samples are among them**, so the image baseline's user5 outer fold
  has zero test samples. `common/ml_utils.py` records that fold with
  `test_samples=0` in `outer_fold_metrics.csv` and excludes it from the
  aggregated mean in `aggregate_metrics.json` (`outer_folds_scored: 4`)
  rather than letting a spurious 0.0 drag the average down.

## Current run

- 172 clean samples, 5 writers (`user1`..`user5`), 4 classes.
- Leave-one-writer-out outer folds, 3 grouped inner validation folds per
  outer fold (2,580 `sample_splits.csv` rows).
- 114 trajectory summary features.
- Sequence tensor shape: `[172, 256, 11]`.
- Image tensor shape: `[146, 224, 224]` (146 of 172 samples have a PNG).
- Aggregate test macro-F1 (mean across outer folds, `±std`):
  - summary: `0.625 ± 0.183`
  - sequence: `0.528 ± 0.103`
  - image: `0.487 ± 0.147` (4 of 5 folds scored; user5 has no images)

These are LOWO macro-F1 numbers on a paper-faithful causal-feature
recomputation, not directly comparable to procedure1's numbers (different
preprocessing, T=128 vs. T=256, and a fixed outer/inner split policy applied
to a different accepted-sample set — procedure1 excluded the 26 missing-PNG
samples outright instead of keeping them for non-image baselines).

## Next steps (per `Airwriting Detection_ Concise Next-Step Plan.pdf`)

- Port the skeleton stream (re-extract 21 hand landmarks from video) as a
  fourth baseline.
- Add fold-scoped trajectory augmentation (rotation/scale/translation/jitter/
  dropout/temporal warp) as procedure1 does, applied only after this stage's
  user-level splitting.
- Combined report (`06_modeling_report` in procedure1) and the extended
  evaluation protocol (ROC-AUC, calibration/ECE, accuracy-coverage,
  user-bootstrap confidence intervals) from `07_evaluation_protocol`.
- The proposed cross-view stroke-masked transformer, once these baselines are
  in place.
