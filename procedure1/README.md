# Procedure 1: Preprocessing, Cleaning, and Augmentation

This folder contains the first two steps from `Airwriting Detection_ Concise Next-Step Plan.pdf`.

Feature extraction and baseline modeling are in `feature_modeling/`, with each modeling step kept in its own subfolder.

Run from the repository root:

```bash
python3 procedure1/preprocess_clean_augment.py
```

Outputs are written to `procedure1/artifacts/`.

Current artifact run:

- 173 JSON samples inventoried.
- 147 clean JSON/PNG/video sample triplets retained.
- 26 samples excluded, all because the trajectory PNG was missing.
- 5 clean users assigned to leave-one-user-out outer test folds.
- 3 grouped inner validation folds per outer fold.
- 2,205 sample split rows: 1,176 train, 588 validation, and 441 test.
- 3,528 train-only augmented trajectory rows generated.

Key files:

- `manifest.csv`: one row per JSON sample with matched PNG/video paths and parsed device.
- `excluded_samples.csv`: missing files, invalid metadata, duplicate IDs, corrupt trajectories, and samples with fewer than 10 valid points.
- `class_user_counts.csv`: per-user/per-label counts before and after cleaning.
- `fold_assignments.csv`: fixed user-level folds. With five or more users this is leave-one-user-out.
- `nested_fold_assignments.csv`: user-level outer test folds plus inner train/validation folds.
- `sample_splits.csv`: sample-level train/validation/test role for every outer and inner fold.
- `clean_manifest.csv`: cleaned samples with point counts, cleaning stats, normalization scale, and the fold where each user is held out.
- `cleaned_trajectories.jsonl`: cleaned variable-length trajectories plus bbox-normalized and 128-point resampled sequences.
- `training_sample_weights.csv`: fold-specific class/user weights for weighted sampling without copying users across folds.
- `augmented_manifest.csv`: fold-scoped train-only trajectory augmentation metadata.
- `augmented_train_trajectories.jsonl`: augmented trajectory sequences generated only for training roles after user-level splitting.
- `video_augmentation_plan.csv`: deterministic video augmentation parameters for the same train-only augmented rows.
- `duplicates_report.csv`: exact hash duplicates and near duplicates detected by DTW within each user/label group.
- `run_summary.json`: counts and run settings.

The script uses only the Python standard library. It does not alter files under `output/`.

Video augmentation is recorded as per-row parameters because this repository does not include a local video processing dependency. Apply those parameters in the eventual video dataloader or install a video stack such as OpenCV/ffmpeg for materialized augmented clips.
