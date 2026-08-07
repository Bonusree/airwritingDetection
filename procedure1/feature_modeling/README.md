# Feature Modeling

This folder keeps each feature/modeling step separate.

Run the steps from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/01_feature_extraction/build_features.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/02_summary_baseline/run_summary_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/03_sequence_baseline/run_sequence_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/04_image_baseline/run_image_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/05_skeleton_status/check_skeleton_readiness.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/05_skeleton_status/extract_skeleton_features.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/05_skeleton_status/run_skeleton_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/06_modeling_report/build_report.py
PYTHONDONTWRITEBYTECODE=1 python3 procedure1/feature_modeling/07_evaluation_protocol/build_evaluation_protocol.py
```

Step folders:

- `01_feature_extraction`: trajectory summary features, 128-step sequence tensors, 224x224 one-channel trajectory images, and image descriptor features.
- `02_summary_baseline`: NumPy baselines over derived trajectory/dynamics features.
- `03_sequence_baseline`: NumPy baselines over flattened `[128, 11]` sequence tensors.
- `04_image_baseline`: NumPy baselines over trajectory-image descriptors.
- `05_skeleton_status`: checks skeleton readiness, extracts 21-landmark skeleton tensors, and runs skeleton-summary baselines.
- `06_modeling_report`: combined metrics table, evaluation figures, and caveats.
- `07_evaluation_protocol`: ROC-AUC, calibration/ECE, accuracy-coverage, user-bootstrap confidence intervals, and efficiency summaries.

The runnable baselines use NumPy only because this environment currently has no `sklearn` or `torch`. The summary, image, and skeleton candidates are nearest centroid, ridge classifier, k-NN, and Gaussian Naive Bayes. The high-dimensional flattened sequence baseline uses nearest centroid only so it finishes quickly without deep-learning dependencies. Model selection uses the inner validation folds from `procedure1/artifacts/sample_splits.csv`; test metrics use the outer held-out users.

Current run:

- 147 clean samples.
- 3,528 train-only augmented trajectory rows.
- 114 trajectory summary features.
- Sequence tensor shape: `[147, 128, 11]`.
- Image tensor shape: `[147, 224, 224]`.
- Skeleton tensor shape: `[147, 128, 21, 3]`, using at most 48 sampled frames per video.
- Aggregate test macro-F1: summary `0.448`, sequence `0.443`, image `0.390`, skeleton `0.279`.

Open `06_modeling_report/artifacts/feature_modeling_report.md` for the combined metrics table and generated figures.
Open `07_evaluation_protocol/artifacts/evaluation_protocol_report.md` for the extended evaluation protocol outputs.
