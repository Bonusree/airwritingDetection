# Extended Evaluation Protocol Report

Metric values are means across held-out users. Confidence intervals use user-level bootstrap resampling.

## User-Bootstrap 95% Confidence Intervals

| Baseline | Accuracy | Balanced Accuracy | Macro-F1 | Top-2 Accuracy | Macro ROC-AUC | ECE, 10 bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image Descriptors | 0.441 [0.220, 0.622] | 0.402 [0.181, 0.584] | 0.390 [0.174, 0.580] | 0.664 [0.325, 0.885] | 0.692 [0.560, 0.803] | 0.358 [0.286, 0.467] |
| Sequence 128 Flattened | 0.447 [0.222, 0.604] | 0.452 [0.222, 0.612] | 0.443 [0.225, 0.600] | 0.679 [0.343, 0.892] | 0.813 [0.746, 0.879] | 0.264 [0.162, 0.400] |
| Skeleton Landmarks | 0.283 [0.117, 0.440] | 0.298 [0.127, 0.494] | 0.279 [0.120, 0.484] | 0.470 [0.210, 0.695] | 0.600 [0.524, 0.721] | 0.443 [0.224, 0.662] |
| Summary Features | 0.416 [0.174, 0.620] | 0.440 [0.203, 0.620] | 0.448 [0.217, 0.607] | 0.631 [0.308, 0.831] | 0.817 [0.739, 0.887] | 0.334 [0.249, 0.438] |

## Efficiency Summary

| Baseline | Median Inference ms/sample | Mean Parameters | Mean Estimated Model Memory KB |
| --- | ---: | ---: | ---: |
| Summary Features | 0.038 | 552.000 | 6.094 |
| Sequence 128 Flattened | 0.065 | 6758.400 | 74.800 |
| Image Descriptors | 0.126 | 9076.800 | 71.973 |
| Skeleton Landmarks | 0.141 | 7045.600 | 59.055 |

## Figures

- [ROC-AUC by class](figures/roc_auc_by_class.png)
- [Calibration reliability](figures/calibration_reliability.png)
- [Accuracy-coverage curves](figures/accuracy_coverage.png)

## Caveat

The current dataset has only five clean labels and five effective held-out users. Treat the intervals and calibration estimates as early diagnostics, not final claims.

## Artifacts

- `user_bootstrap_metric_ci.csv`
- `per_user_metrics.csv`
- `roc_auc_by_class.csv`
- `calibration_bins.csv`
- `accuracy_coverage.csv`
- `efficiency_summary.csv`
- `figures/*.png`
