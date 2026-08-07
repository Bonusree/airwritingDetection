#!/usr/bin/env python3
"""Build a compact feature-modeling report and metric figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


BASELINES = (
    ("Summary Features", "02_summary_baseline"),
    ("Sequence 128 Flattened", "03_sequence_baseline"),
    ("Image Descriptors", "04_image_baseline"),
    ("Skeleton Landmarks", "05_skeleton_status"),
)

METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("top2_accuracy", "Top-2 Accuracy"),
)


def main() -> int:
    script_path = Path(__file__).resolve()
    feature_root = script_path.parents[1]
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_artifacts = load_baseline_artifacts(feature_root)
    figure_paths = build_figures(output_dir, baseline_artifacts)

    lines = [
        "# Procedure 1 Feature Modeling Report",
        "",
        "All reported test scores use outer user-held-out folds. Model choices are selected from inner validation folds only.",
        "",
        "## Aggregate Metrics",
        "",
        "| Baseline | Accuracy | Balanced Accuracy | Macro-F1 | Top-2 Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    summary_rows = []
    for artifact in baseline_artifacts:
        display_name = artifact["display_name"]
        aggregate = artifact["aggregate"]
        metrics = aggregate["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    fmt(metrics["accuracy"]["mean"]),
                    fmt(metrics["balanced_accuracy"]["mean"]),
                    fmt(metrics["macro_f1"]["mean"]),
                    fmt(metrics["top2_accuracy"]["mean"]),
                ]
            )
            + " |"
        )
        summary_rows.append(
            {
                "baseline": display_name,
                "accuracy_mean": metrics["accuracy"]["mean"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
                "macro_f1_mean": metrics["macro_f1"]["mean"],
                "top2_accuracy_mean": metrics["top2_accuracy"]["mean"],
                "outer_folds_with_unseen_test_labels": "|".join(
                    str(item) for item in aggregate["outer_folds_with_unseen_test_labels"]
                ),
            }
        )

    if figure_paths:
        lines.extend(
            [
                "",
                "## Evaluation Figures",
                "",
                f"- [Aggregate metric comparison]({figure_paths['aggregate_metric_comparison']})",
                f"- [Outer-fold metric stability]({figure_paths['outer_fold_metric_stability']})",
                "",
                "| Baseline | Confusion Matrix | Per-Class Precision/Recall/F1 |",
                "| --- | --- | --- |",
            ]
        )
        for artifact in baseline_artifacts:
            slug = artifact["slug"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        artifact["display_name"],
                        f"[PNG]({figure_paths[f'confusion_matrix_{slug}']})",
                        f"[PNG]({figure_paths[f'per_class_metrics_{slug}']})",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Important Caveat",
            "",
            "The current clean dataset has only five labels and uneven user/label coverage. Some held-out users contain labels that do not appear in the corresponding training users, so those outer folds are intentionally marked in the aggregate JSON files.",
            "",
            "## Artifacts",
            "",
            "- `01_feature_extraction/artifacts/summary_features.csv`",
            "- `01_feature_extraction/artifacts/sequence_features_128.npz`",
            "- `01_feature_extraction/artifacts/image_224_uint8.npz`",
            "- `02_summary_baseline/artifacts/outer_fold_metrics.csv`",
            "- `03_sequence_baseline/artifacts/outer_fold_metrics.csv`",
            "- `04_image_baseline/artifacts/outer_fold_metrics.csv`",
            "- `05_skeleton_status/artifacts/skeleton_landmarks_128.npz`",
            "- `05_skeleton_status/artifacts/skeleton_features.csv`",
            "- `05_skeleton_status/artifacts/outer_fold_metrics.csv`",
            "- `06_modeling_report/artifacts/figures/*.png`",
        ]
    )

    (output_dir / "feature_modeling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary_rows:
        with (output_dir / "aggregate_baseline_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote modeling report artifacts to {output_dir}")
    return 0


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def load_baseline_artifacts(feature_root: Path) -> list[dict[str, object]]:
    artifacts = []
    for display_name, folder in BASELINES:
        artifact_dir = feature_root / folder / "artifacts"
        aggregate_path = artifact_dir / "aggregate_metrics.json"
        confusion_path = artifact_dir / "confusion_matrix.csv"
        outer_metrics_path = artifact_dir / "outer_fold_metrics.csv"
        if not aggregate_path.exists() or not confusion_path.exists() or not outer_metrics_path.exists():
            continue

        labels, confusion = read_confusion_matrix(confusion_path)
        artifacts.append(
            {
                "display_name": display_name,
                "folder": folder,
                "slug": slugify(folder),
                "aggregate": json.loads(aggregate_path.read_text(encoding="utf-8")),
                "outer_metrics": read_csv_rows(outer_metrics_path),
                "labels": labels,
                "confusion": confusion,
            }
        )
    return artifacts


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        pred_columns = [name for name in (reader.fieldnames or []) if name.startswith("pred_")]
    labels = [row["true_label"] for row in rows]
    confusion = np.array([[int(row[column]) for column in pred_columns] for row in rows], dtype=np.int64)
    return labels, confusion


def slugify(value: str) -> str:
    return value.replace("_baseline", "").replace("_status", "").replace("_", "-")


def build_figures(output_dir: Path, baseline_artifacts: list[dict[str, object]]) -> dict[str, str]:
    if not baseline_artifacts:
        return {}
    try:
        plt = load_pyplot()
    except ImportError as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return {}

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    aggregate_path = figures_dir / "aggregate_metric_comparison.png"
    plot_aggregate_metric_comparison(plt, baseline_artifacts, aggregate_path)
    figure_paths["aggregate_metric_comparison"] = relative_figure_path(output_dir, aggregate_path)

    stability_path = figures_dir / "outer_fold_metric_stability.png"
    plot_outer_fold_metric_stability(plt, baseline_artifacts, stability_path)
    figure_paths["outer_fold_metric_stability"] = relative_figure_path(output_dir, stability_path)

    for artifact in baseline_artifacts:
        slug = str(artifact["slug"])
        confusion_path = figures_dir / f"confusion_matrix_{slug}.png"
        per_class_path = figures_dir / f"per_class_metrics_{slug}.png"
        plot_confusion_matrix(plt, artifact, confusion_path)
        plot_per_class_metrics(plt, artifact, per_class_path)
        figure_paths[f"confusion_matrix_{slug}"] = relative_figure_path(output_dir, confusion_path)
        figure_paths[f"per_class_metrics_{slug}"] = relative_figure_path(output_dir, per_class_path)

    return figure_paths


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
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )
    return plt


def plot_aggregate_metric_comparison(plt, baseline_artifacts: list[dict[str, object]], output_path: Path) -> None:
    names = [str(artifact["display_name"]) for artifact in baseline_artifacts]
    x = np.arange(len(names))
    width = 0.18
    colors = ["#2f6f9f", "#d07c2c", "#4b8f54", "#9a4f8f"]

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for metric_index, (metric_key, metric_label) in enumerate(METRICS):
        means = [
            float(artifact["aggregate"]["metrics"][metric_key]["mean"])  # type: ignore[index]
            for artifact in baseline_artifacts
        ]
        offsets = x + (metric_index - (len(METRICS) - 1) / 2) * width
        bars = ax.bar(
            offsets,
            means,
            width,
            label=metric_label,
            color=colors[metric_index],
            alpha=0.9,
            linewidth=0,
        )
        for bar, value in zip(bars, means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    ax.set_title("Aggregate Test Metrics by Baseline")
    ax.set_ylabel("Mean score across outer folds")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_outer_fold_metric_stability(plt, baseline_artifacts: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    colors = ["#2f6f9f", "#d07c2c", "#4b8f54", "#9a4f8f"]
    all_fold_ids = sorted(
        {
            int(row["outer_fold_id"])
            for artifact in baseline_artifacts
            for row in artifact["outer_metrics"]  # type: ignore[index]
        }
    )
    for axis, (metric_key, metric_label) in zip(axes.ravel(), METRICS, strict=True):
        for artifact_index, artifact in enumerate(baseline_artifacts):
            rows = sorted(
                artifact["outer_metrics"],  # type: ignore[arg-type]
                key=lambda row: int(row["outer_fold_id"]),
            )
            fold_ids = [int(row["outer_fold_id"]) for row in rows]
            values = [float(row[metric_key]) for row in rows]
            axis.plot(
                fold_ids,
                values,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[artifact_index % len(colors)],
                label=str(artifact["display_name"]),
            )
        axis.set_title(metric_label)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Outer fold")
        axis.set_xticks(all_fold_ids)
        axis.grid(alpha=0.24)

    axes[0, 0].set_ylabel("Test score")
    axes[1, 0].set_ylabel("Test score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Outer-Fold Metric Stability", y=0.99)
    fig.legend(handles, labels, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(plt, artifact: dict[str, object], output_path: Path) -> None:
    labels = artifact["labels"]  # type: ignore[assignment]
    confusion = artifact["confusion"]  # type: ignore[assignment]
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=np.float64), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(f"Confusion Matrix: {artifact['display_name']}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    max_value = float(normalized.max()) if normalized.size else 0.0
    threshold = max(0.45, max_value * 0.55)
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = int(confusion[row_index, col_index])
            color = "white" if normalized[row_index, col_index] >= threshold else "#222222"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized recall")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(plt, artifact: dict[str, object], output_path: Path) -> None:
    labels = artifact["labels"]  # type: ignore[assignment]
    confusion = artifact["confusion"]  # type: ignore[assignment]
    metric_rows = aggregate_per_class_metrics(confusion)
    x = np.arange(len(labels))
    width = 0.23
    series = (
        ("precision", "Precision", "#2f6f9f"),
        ("recall", "Recall", "#d07c2c"),
        ("f1", "F1", "#4b8f54"),
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for series_index, (metric_key, metric_label, color) in enumerate(series):
        values = [row[metric_key] for row in metric_rows]
        offsets = x + (series_index - 1) * width
        ax.bar(offsets, values, width, label=metric_label, color=color, alpha=0.9, linewidth=0)

    supports = [row["support"] for row in metric_rows]
    for index, support in enumerate(supports):
        ax.text(index, 1.025, f"n={support}", ha="center", va="bottom", fontsize=8, color="#444444")

    ax.set_title(f"Per-Class Metrics: {artifact['display_name']}")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def aggregate_per_class_metrics(confusion: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "support": int(confusion[index, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def relative_figure_path(output_dir: Path, figure_path: Path) -> str:
    return figure_path.relative_to(output_dir).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
