#!/usr/bin/env python3
"""Build extended evaluation-protocol artifacts from baseline predictions."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


BASELINES = (
    ("Summary Features", "02_summary_baseline"),
    ("Sequence 128 Flattened", "03_sequence_baseline"),
    ("Image Descriptors", "04_image_baseline"),
    ("Skeleton Landmarks", "05_skeleton_status"),
)

BOOTSTRAP_METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("top2_accuracy", "Top-2 Accuracy"),
    ("macro_roc_auc", "Macro ROC-AUC"),
    ("ece_10_bin", "ECE, 10 bins"),
)

COVERAGE_POINTS = tuple(index / 10.0 for index in range(1, 11))


def main() -> int:
    script_path = Path(__file__).resolve()
    feature_root = script_path.parents[1]
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_baseline_artifacts(feature_root)
    if not artifacts:
        raise SystemExit("No baseline artifacts found. Run the baseline scripts first.")

    summary_rows = []
    roc_rows = []
    calibration_rows = []
    coverage_rows = []
    efficiency_rows = []
    per_user_rows = []

    for artifact in artifacts:
        display_name = str(artifact["display_name"])
        labels = artifact["labels"]  # type: ignore[assignment]
        prediction_rows = artifact["predictions"]  # type: ignore[assignment]
        metrics = compute_prediction_metrics(prediction_rows, labels)
        user_metrics = compute_user_metrics(prediction_rows, labels)
        per_user_rows.extend({"baseline": display_name, **row} for row in user_metrics)
        user_means = mean_user_metrics(user_metrics)
        bootstrap = bootstrap_ci_by_user_metrics(user_metrics)

        for metric_key, metric_label in BOOTSTRAP_METRICS:
            ci = bootstrap[metric_key]
            summary_rows.append(
                {
                    "baseline": display_name,
                    "metric": metric_label,
                    "value": round_float(user_means[metric_key]),
                    "ci95_low": round_float(ci["low"]),
                    "ci95_high": round_float(ci["high"]),
                    "bootstrap_unit": "user",
                    "bootstrap_repeats": ci["repeats"],
                }
            )

        for row in metrics["roc_auc_by_class"]:
            roc_rows.append({"baseline": display_name, **row})
        for row in metrics["calibration_bins"]:
            calibration_rows.append({"baseline": display_name, **row})
        for row in metrics["accuracy_coverage"]:
            coverage_rows.append({"baseline": display_name, **row})

        efficiency = summarize_efficiency(artifact["outer_metrics"])  # type: ignore[arg-type]
        if efficiency:
            efficiency_rows.append({"baseline": display_name, **efficiency})

    write_csv(output_dir / "user_bootstrap_metric_ci.csv", summary_rows)
    write_csv(output_dir / "per_user_metrics.csv", per_user_rows)
    write_csv(output_dir / "roc_auc_by_class.csv", roc_rows)
    write_csv(output_dir / "calibration_bins.csv", calibration_rows)
    write_csv(output_dir / "accuracy_coverage.csv", coverage_rows)
    write_csv(output_dir / "efficiency_summary.csv", efficiency_rows)
    write_report(output_dir / "evaluation_protocol_report.md", summary_rows, efficiency_rows)
    build_figures(output_dir, summary_rows, roc_rows, calibration_rows, coverage_rows)

    print(f"Wrote extended evaluation artifacts to {output_dir}")
    return 0


def load_baseline_artifacts(feature_root: Path) -> list[dict[str, object]]:
    artifacts = []
    for display_name, folder in BASELINES:
        artifact_dir = feature_root / folder / "artifacts"
        aggregate_path = artifact_dir / "aggregate_metrics.json"
        predictions_path = artifact_dir / "test_predictions.csv"
        outer_metrics_path = artifact_dir / "outer_fold_metrics.csv"
        if not aggregate_path.exists() or not predictions_path.exists():
            continue
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        prediction_rows = read_csv_rows(predictions_path)
        labels = aggregate.get("labels") or infer_labels_from_score_columns(prediction_rows)
        require_score_columns(prediction_rows, labels, predictions_path)
        artifacts.append(
            {
                "display_name": display_name,
                "folder": folder,
                "labels": labels,
                "predictions": prediction_rows,
                "outer_metrics": read_csv_rows(outer_metrics_path) if outer_metrics_path.exists() else [],
            }
        )
    return artifacts


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_labels_from_score_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return sorted(column.removeprefix("score_") for column in rows[0] if column.startswith("score_"))


def require_score_columns(rows: list[dict[str, str]], labels: list[str], path: Path) -> None:
    if not rows:
        raise SystemExit(f"No prediction rows found in {path}")
    missing = [f"score_{label}" for label in labels if f"score_{label}" not in rows[0]]
    if missing:
        raise SystemExit(
            f"{path} is missing full probability columns: {', '.join(missing)}. "
            "Rerun the baseline scripts after updating ml_utils.py."
        )


def compute_prediction_metrics(rows: list[dict[str, str]], labels: list[str]) -> dict[str, Any]:
    y_true = [row["true_label"] for row in rows]
    scores = score_matrix(rows, labels)
    predicted_indices = np.argmax(scores, axis=1)
    y_pred = [labels[index] for index in predicted_indices]
    confusion = confusion_matrix(y_true, y_pred, labels)
    class_rows = per_class_from_confusion(confusion, labels)

    supports = np.array([row["support"] for row in class_rows], dtype=np.float64)
    present = supports > 0
    recall_values = np.array([row["recall"] for row in class_rows], dtype=np.float64)
    f1_values = np.array([row["f1"] for row in class_rows], dtype=np.float64)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    top2 = top_k_accuracy(y_true, scores, labels, k=2)
    roc_rows = roc_auc_by_class(y_true, scores, labels)
    valid_auc_values = [row["roc_auc"] for row in roc_rows if not math.isnan(row["roc_auc"])]
    calibration_bins, ece = calibration_metrics(y_true, scores, labels, n_bins=10)
    coverage_rows = accuracy_coverage(y_true, scores, labels)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall_values[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1_values[present].mean()) if present.any() else 0.0,
        "top2_accuracy": top2,
        "macro_roc_auc": float(np.mean(valid_auc_values)) if valid_auc_values else math.nan,
        "ece_10_bin": ece,
        "roc_auc_by_class": roc_rows,
        "calibration_bins": calibration_bins,
        "accuracy_coverage": coverage_rows,
    }


def score_matrix(rows: list[dict[str, str]], labels: list[str]) -> np.ndarray:
    return np.array([[float(row[f"score_{label}"]) for label in labels] for row in rows], dtype=np.float64)


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        confusion[label_to_index[truth], label_to_index[prediction]] += 1
    return confusion


def per_class_from_confusion(confusion: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(labels):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "support": int(confusion[index, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def top_k_accuracy(y_true: list[str], scores: np.ndarray, labels: list[str], k: int) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    top_indices = np.argsort(-scores, axis=1)[:, : min(k, len(labels))]
    hits = sum(1 for row_index, label in enumerate(y_true) if label_to_index[label] in top_indices[row_index])
    return hits / max(1, len(y_true))


def roc_auc_by_class(y_true: list[str], scores: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for label_index, label in enumerate(labels):
        binary_truth = np.array([1 if truth == label else 0 for truth in y_true], dtype=np.int64)
        auc = binary_roc_auc(binary_truth, scores[:, label_index])
        rows.append(
            {
                "label": label,
                "positive_support": int(binary_truth.sum()),
                "negative_support": int((1 - binary_truth).sum()),
                "roc_auc": round_float(auc),
            }
        )
    return rows


def binary_roc_auc(binary_truth: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(binary_truth.sum())
    negative_count = int(len(binary_truth) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return math.nan
    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[binary_truth == 1].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def calibration_metrics(
    y_true: list[str],
    scores: np.ndarray,
    labels: list[str],
    n_bins: int,
) -> tuple[list[dict[str, Any]], float]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_indices = np.array([label_to_index[label] for label in y_true], dtype=np.int64)
    predicted_indices = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    correct = (predicted_indices == true_indices).astype(np.float64)

    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_index in range(n_bins):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        if bin_index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else math.nan
        avg_confidence = float(confidences[mask].mean()) if count else math.nan
        gap = abs(accuracy - avg_confidence) if count else math.nan
        ece += (count / max(1, len(y_true))) * (gap if count else 0.0)
        rows.append(
            {
                "bin_index": bin_index,
                "confidence_low": round_float(lower),
                "confidence_high": round_float(upper),
                "count": count,
                "accuracy": round_float(accuracy),
                "avg_confidence": round_float(avg_confidence),
                "calibration_gap": round_float(gap),
            }
        )
    return rows, float(ece)


def accuracy_coverage(y_true: list[str], scores: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_indices = np.array([label_to_index[label] for label in y_true], dtype=np.int64)
    predicted_indices = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    correct = predicted_indices == true_indices
    order = np.argsort(-confidences)
    rows = []
    for coverage in COVERAGE_POINTS:
        keep_count = max(1, int(math.ceil(coverage * len(y_true))))
        kept = order[:keep_count]
        rows.append(
            {
                "coverage": round_float(keep_count / max(1, len(y_true))),
                "threshold": round_float(float(confidences[kept].min())),
                "kept_samples": keep_count,
                "accuracy": round_float(float(correct[kept].mean())),
            }
        )
    return rows


def compute_user_metrics(rows: list[dict[str, str]], labels: list[str]) -> list[dict[str, Any]]:
    rows_by_user: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_user.setdefault(row["user_id"], []).append(row)

    user_rows = []
    for user_id in sorted(rows_by_user):
        user_predictions = rows_by_user[user_id]
        metrics = compute_prediction_metrics(user_predictions, labels)
        row = {
            "user_id": user_id,
            "samples": len(user_predictions),
            "labels_present": "|".join(sorted({item["true_label"] for item in user_predictions})),
        }
        for metric_key, _metric_label in BOOTSTRAP_METRICS:
            row[metric_key] = round_float(metrics[metric_key])
        user_rows.append(row)
    return user_rows


def mean_user_metrics(user_rows: list[dict[str, Any]]) -> dict[str, float]:
    means = {}
    for metric_key, _metric_label in BOOTSTRAP_METRICS:
        values = [float(row[metric_key]) for row in user_rows if not math.isnan(float(row[metric_key]))]
        means[metric_key] = float(np.mean(values)) if values else math.nan
    return means


def bootstrap_ci_by_user_metrics(
    user_rows: list[dict[str, Any]],
    repeats: int = 1000,
    seed: int = 20260806,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values_by_metric = {metric_key: [] for metric_key, _label in BOOTSTRAP_METRICS}

    for _repeat in range(repeats):
        sampled_rows = rng.choice(user_rows, size=len(user_rows), replace=True)
        for metric_key in values_by_metric:
            sampled_values = [
                float(row[metric_key]) for row in sampled_rows if not math.isnan(float(row[metric_key]))
            ]
            if sampled_values:
                values_by_metric[metric_key].append(float(np.mean(sampled_values)))

    return {
        metric_key: {
            "low": percentile_or_nan(values, 2.5),
            "high": percentile_or_nan(values, 97.5),
            "repeats": repeats,
        }
        for metric_key, values in values_by_metric.items()
    }


def percentile_or_nan(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.array(values, dtype=np.float64), percentile))


def summarize_efficiency(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows or "median_inference_ms" not in rows[0]:
        return {}
    inference_values = [float(row["median_inference_ms"]) for row in rows if row.get("median_inference_ms")]
    parameter_values = [float(row["model_parameters"]) for row in rows if row.get("model_parameters")]
    memory_values = [
        float(row["estimated_model_memory_bytes"]) for row in rows if row.get("estimated_model_memory_bytes")
    ]
    return {
        "outer_folds": len(rows),
        "median_inference_ms": round_float(float(np.median(inference_values))) if inference_values else math.nan,
        "mean_model_parameters": round_float(float(np.mean(parameter_values))) if parameter_values else math.nan,
        "mean_estimated_model_memory_kb": round_float(float(np.mean(memory_values) / 1024.0)) if memory_values else math.nan,
    }


def write_report(path: Path, summary_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, Any]]) -> None:
    baselines = sorted({row["baseline"] for row in summary_rows})
    metric_labels = [label for _key, label in BOOTSTRAP_METRICS]
    by_baseline_metric = {(row["baseline"], row["metric"]): row for row in summary_rows}

    lines = [
        "# Extended Evaluation Protocol Report",
        "",
        "Metric values are means across held-out users. Confidence intervals use user-level bootstrap resampling.",
        "",
        "## User-Bootstrap 95% Confidence Intervals",
        "",
        "| Baseline | " + " | ".join(metric_labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(metric_labels)) + " |",
    ]
    for baseline in baselines:
        cells = [baseline]
        for metric_label in metric_labels:
            row = by_baseline_metric[(baseline, metric_label)]
            cells.append(format_ci_cell(row))
        lines.append("| " + " | ".join(cells) + " |")

    if efficiency_rows:
        lines.extend(
            [
                "",
                "## Efficiency Summary",
                "",
                "| Baseline | Median Inference ms/sample | Mean Parameters | Mean Estimated Model Memory KB |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in efficiency_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["baseline"]),
                        fmt(row["median_inference_ms"]),
                        fmt(row["mean_model_parameters"]),
                        fmt(row["mean_estimated_model_memory_kb"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- [ROC-AUC by class](figures/roc_auc_by_class.png)",
            "- [Calibration reliability](figures/calibration_reliability.png)",
            "- [Accuracy-coverage curves](figures/accuracy_coverage.png)",
            "",
            "## Caveat",
            "",
            "The current dataset has only five clean labels and five effective held-out users. Treat the intervals and calibration estimates as early diagnostics, not final claims.",
            "",
            "## Artifacts",
            "",
            "- `user_bootstrap_metric_ci.csv`",
            "- `per_user_metrics.csv`",
            "- `roc_auc_by_class.csv`",
            "- `calibration_bins.csv`",
            "- `accuracy_coverage.csv`",
            "- `efficiency_summary.csv`",
            "- `figures/*.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ci_cell(row: dict[str, Any]) -> str:
    return f"{fmt(row['value'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"


def build_figures(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    roc_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    try:
        plt = load_pyplot()
    except ImportError as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_auc(plt, roc_rows, figures_dir / "roc_auc_by_class.png")
    plot_calibration(plt, summary_rows, calibration_rows, figures_dir / "calibration_reliability.png")
    plot_accuracy_coverage(plt, coverage_rows, figures_dir / "accuracy_coverage.png")


def load_pyplot():
    matplotlib_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-airwriting"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans Bengali", "Noto Sans", "DejaVu Sans"],
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )
    return plt


def plot_roc_auc(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    baselines = sorted({row["baseline"] for row in rows})
    labels = sorted({row["label"] for row in rows})
    x = np.arange(len(labels))
    width = 0.18
    colors = ["#2f6f9f", "#d07c2c", "#4b8f54", "#9a4f8f"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for baseline_index, baseline in enumerate(baselines):
        values = []
        for label in labels:
            match = next(row for row in rows if row["baseline"] == baseline and row["label"] == label)
            value = match["roc_auc"]
            values.append(0.0 if math.isnan(float(value)) else float(value))
        offsets = x + (baseline_index - (len(baselines) - 1) / 2) * width
        ax.bar(offsets, values, width, label=baseline, color=colors[baseline_index % len(colors)], linewidth=0)
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("One-vs-Rest ROC-AUC by Class")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    plt,
    summary_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    baselines = sorted({row["baseline"] for row in rows})
    ece_by_baseline = {
        row["baseline"]: float(row["value"]) for row in summary_rows if row["metric"] == "ECE, 10 bins"
    }
    colors = ["#2f6f9f", "#d07c2c", "#4b8f54", "#9a4f8f"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.2, label="Perfect calibration")
    for baseline_index, baseline in enumerate(baselines):
        baseline_rows = [
            row
            for row in rows
            if row["baseline"] == baseline and int(row["count"]) > 0 and not math.isnan(float(row["accuracy"]))
        ]
        ax.plot(
            [float(row["avg_confidence"]) for row in baseline_rows],
            [float(row["accuracy"]) for row in baseline_rows],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[baseline_index % len(colors)],
            label=f"{baseline} (ECE {ece_by_baseline[baseline]:.2f})",
        )
    ax.set_title("Calibration Reliability")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.24)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_coverage(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    baselines = sorted({row["baseline"] for row in rows})
    colors = ["#2f6f9f", "#d07c2c", "#4b8f54", "#9a4f8f"]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for baseline_index, baseline in enumerate(baselines):
        baseline_rows = sorted(
            [row for row in rows if row["baseline"] == baseline],
            key=lambda row: float(row["coverage"]),
        )
        ax.plot(
            [float(row["coverage"]) for row in baseline_rows],
            [float(row["accuracy"]) for row in baseline_rows],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[baseline_index % len(colors)],
            label=baseline,
        )
    ax.set_title("Accuracy-Coverage Curves")
    ax.set_xlabel("Coverage after confidence filtering")
    ax.set_ylabel("Accuracy on retained samples")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.24)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def round_float(value: float) -> float:
    if math.isnan(float(value)):
        return math.nan
    return round(float(value), 6)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    float_value = float(value)
    if math.isnan(float_value):
        return "n/a"
    return f"{float_value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
