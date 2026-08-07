# Procedure 1 Feature Modeling Report

All reported test scores use outer user-held-out folds. Model choices are selected from inner validation folds only.

## Aggregate Metrics

| Baseline | Accuracy | Balanced Accuracy | Macro-F1 | Top-2 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Summary Features | 0.416 | 0.440 | 0.448 | 0.631 |
| Sequence 128 Flattened | 0.447 | 0.452 | 0.443 | 0.679 |
| Image Descriptors | 0.441 | 0.402 | 0.390 | 0.664 |
| Skeleton Landmarks | 0.283 | 0.298 | 0.279 | 0.459 |

## Evaluation Figures

- [Aggregate metric comparison](figures/aggregate_metric_comparison.png)
- [Outer-fold metric stability](figures/outer_fold_metric_stability.png)

| Baseline | Confusion Matrix | Per-Class Precision/Recall/F1 |
| --- | --- | --- |
| Summary Features | [PNG](figures/confusion_matrix_02-summary.png) | [PNG](figures/per_class_metrics_02-summary.png) |
| Sequence 128 Flattened | [PNG](figures/confusion_matrix_03-sequence.png) | [PNG](figures/per_class_metrics_03-sequence.png) |
| Image Descriptors | [PNG](figures/confusion_matrix_04-image.png) | [PNG](figures/per_class_metrics_04-image.png) |
| Skeleton Landmarks | [PNG](figures/confusion_matrix_05-skeleton.png) | [PNG](figures/per_class_metrics_05-skeleton.png) |

## Important Caveat

The current clean dataset has only five labels and uneven user/label coverage. Some held-out users contain labels that do not appear in the corresponding training users, so those outer folds are intentionally marked in the aggregate JSON files.

## Artifacts

- `01_feature_extraction/artifacts/summary_features.csv`
- `01_feature_extraction/artifacts/sequence_features_128.npz`
- `01_feature_extraction/artifacts/image_224_uint8.npz`
- `02_summary_baseline/artifacts/outer_fold_metrics.csv`
- `03_sequence_baseline/artifacts/outer_fold_metrics.csv`
- `04_image_baseline/artifacts/outer_fold_metrics.csv`
- `05_skeleton_status/artifacts/skeleton_landmarks_128.npz`
- `05_skeleton_status/artifacts/skeleton_features.csv`
- `05_skeleton_status/artifacts/outer_fold_metrics.csv`
- `06_modeling_report/artifacts/figures/*.png`
