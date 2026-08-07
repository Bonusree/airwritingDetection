"""Small NumPy modeling utilities for procedure1 feature baselines."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


META_COLUMNS = {
    "sample_id",
    "source_sample_id",
    "user_id",
    "label",
    "role",
    "created_at",
    "device",
    "json_path",
    "image_path",
    "video_path",
    "outer_fold_id",
    "inner_fold_id",
    "augmented_sample_id",
}


@dataclass
class FeatureTable:
    X_by_id: dict[str, np.ndarray]
    y_by_id: dict[str, str]
    meta_by_id: dict[str, dict[str, str]]
    feature_names: list[str]
    labels: list[str]


@dataclass
class AugmentedTable:
    rows: list[dict[str, Any]]
    feature_names: list[str]


class StandardScaler:
    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


class RidgeClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: list[str]) -> "RidgeClassifier":
        self.classes_ = sorted(set(y))
        class_to_idx = {label: index for index, label in enumerate(self.classes_)}
        y_one_hot = np.zeros((len(y), len(self.classes_)), dtype=np.float64)
        for row_index, label in enumerate(y):
            y_one_hot[row_index, class_to_idx[label]] = 1.0

        X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        eye = np.eye(X_aug.shape[1], dtype=np.float64) * self.alpha
        eye[-1, -1] = 0.0
        self.weights_ = np.linalg.pinv(X_aug.T @ X_aug + eye) @ X_aug.T @ y_one_hot
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        return softmax(X_aug @ self.weights_)


class NearestCentroid:
    def fit(self, X: np.ndarray, y: list[str]) -> "NearestCentroid":
        self.classes_ = sorted(set(y))
        self.centroids_ = np.vstack([X[np.array(y) == label].mean(axis=0) for label in self.classes_])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(X, self.centroids_)
        return softmax(-distances)


class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: list[str]) -> "KNNClassifier":
        self.X_ = X
        self.y_ = np.array(y)
        self.classes_ = sorted(set(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(X, self.X_)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        class_to_idx = {label: index for index, label in enumerate(self.classes_)}
        k = min(self.k, self.X_.shape[0])
        for row_index in range(X.shape[0]):
            nearest = np.argpartition(distances[row_index], k - 1)[:k]
            weights = 1.0 / (distances[row_index, nearest] + 1e-9)
            for neighbor_index, weight in zip(nearest, weights, strict=True):
                proba[row_index, class_to_idx[self.y_[neighbor_index]]] += weight
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        return proba / row_sums


class GaussianNB:
    def fit(self, X: np.ndarray, y: list[str]) -> "GaussianNB":
        self.classes_ = sorted(set(y))
        self.means_ = []
        self.vars_ = []
        self.log_priors_ = []
        y_array = np.array(y)
        for label in self.classes_:
            X_class = X[y_array == label]
            self.means_.append(X_class.mean(axis=0))
            self.vars_.append(X_class.var(axis=0) + 1e-6)
            self.log_priors_.append(math.log(len(X_class) / len(X)))
        self.means_ = np.vstack(self.means_)
        self.vars_ = np.vstack(self.vars_)
        self.log_priors_ = np.array(self.log_priors_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = []
        for class_index in range(len(self.classes_)):
            mean = self.means_[class_index]
            var = self.vars_[class_index]
            log_likelihood = -0.5 * (np.log(2.0 * math.pi * var) + ((X - mean) ** 2) / var).sum(axis=1)
            log_probs.append(log_likelihood + self.log_priors_[class_index])
        return softmax(np.vstack(log_probs).T)


MODEL_CONFIGS = [
    {"model": "nearest_centroid"},
    {"model": "ridge", "alpha": 0.1},
    {"model": "ridge", "alpha": 1.0},
    {"model": "ridge", "alpha": 10.0},
    {"model": "knn", "k": 1},
    {"model": "knn", "k": 3},
    {"model": "knn", "k": 5},
    {"model": "gaussian_nb"},
]


def project_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[3]


def load_feature_csv(path: Path) -> FeatureTable:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    feature_names = [name for name in fieldnames if name not in META_COLUMNS]

    X_by_id: dict[str, np.ndarray] = {}
    y_by_id: dict[str, str] = {}
    meta_by_id: dict[str, dict[str, str]] = {}
    labels: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        values = np.array([float(row.get(name, 0.0) or 0.0) for name in feature_names], dtype=np.float64)
        X_by_id[sample_id] = values
        y_by_id[sample_id] = row["label"]
        meta_by_id[sample_id] = row
        labels.add(row["label"])
    return FeatureTable(X_by_id, y_by_id, meta_by_id, feature_names, sorted(labels))


def load_augmented_feature_csv(path: Path) -> AugmentedTable:
    if not path.exists():
        return AugmentedTable([], [])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    feature_names = [name for name in fieldnames if name not in META_COLUMNS]
    augmented_rows: list[dict[str, Any]] = []
    for row in rows:
        augmented_rows.append(
            {
                "outer_fold_id": int(row["outer_fold_id"]),
                "inner_fold_id": int(row["inner_fold_id"]),
                "sample_id": row["augmented_sample_id"],
                "source_sample_id": row["source_sample_id"],
                "user_id": row["user_id"],
                "label": row["label"],
                "X": np.array([float(row.get(name, 0.0) or 0.0) for name in feature_names], dtype=np.float64),
            }
        )
    return AugmentedTable(augmented_rows, feature_names)


def load_sequence_npz(path: Path) -> FeatureTable:
    data = np.load(path, allow_pickle=False)
    X = data["X"].astype(np.float64)
    sample_ids = data["sample_ids"].astype(str)
    labels = data["labels"].astype(str)
    users = data["users"].astype(str)
    X_flat = X.reshape((X.shape[0], -1))
    feature_names = [f"seq_{index:04d}" for index in range(X_flat.shape[1])]

    X_by_id = {sample_id: X_flat[index] for index, sample_id in enumerate(sample_ids)}
    y_by_id = {sample_id: labels[index] for index, sample_id in enumerate(sample_ids)}
    meta_by_id = {
        sample_id: {"sample_id": sample_id, "user_id": users[index], "label": labels[index]}
        for index, sample_id in enumerate(sample_ids)
    }
    return FeatureTable(X_by_id, y_by_id, meta_by_id, feature_names, sorted(set(labels.tolist())))


def load_augmented_sequence_npz(path: Path) -> AugmentedTable:
    if not path.exists():
        return AugmentedTable([], [])
    data = np.load(path, allow_pickle=False)
    X = data["X"].astype(np.float64)
    X_flat = X.reshape((X.shape[0], -1))
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(data["sample_ids"].astype(str)):
        rows.append(
            {
                "outer_fold_id": int(data["outer_fold_ids"][index]),
                "inner_fold_id": int(data["inner_fold_ids"][index]),
                "sample_id": sample_id,
                "source_sample_id": str(data["source_sample_ids"][index]),
                "user_id": str(data["users"][index]),
                "label": str(data["labels"][index]),
                "X": X_flat[index],
            }
        )
    feature_names = [f"seq_{index:04d}" for index in range(X_flat.shape[1])]
    return AugmentedTable(rows, feature_names)


def load_sample_splits(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_nested_cv(
    *,
    baseline_name: str,
    feature_table: FeatureTable,
    splits: list[dict[str, str]],
    output_dir: Path,
    augmented: AugmentedTable | None = None,
    model_configs: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(feature_table.labels)
    if augmented and augmented.rows:
        labels = sorted(set(labels) | {str(row["label"]) for row in augmented.rows})

    validation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    outer_metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confusion_total = np.zeros((len(labels), len(labels)), dtype=int)

    outer_fold_ids = sorted({int(row["outer_fold_id"]) for row in splits})
    configs = model_configs or MODEL_CONFIGS
    for outer_fold_id in outer_fold_ids:
        inner_fold_ids = sorted(
            {int(row["inner_fold_id"]) for row in splits if int(row["outer_fold_id"]) == outer_fold_id}
        )
        config_scores: list[dict[str, Any]] = []
        for config_index, config in enumerate(configs):
            inner_metrics = []
            for inner_fold_id in inner_fold_ids:
                train_ids = split_sample_ids(splits, outer_fold_id, inner_fold_id, "train")
                val_ids = split_sample_ids(splits, outer_fold_id, inner_fold_id, "val")
                X_train, y_train = collect_original_features(feature_table, train_ids)
                if augmented and augmented.rows:
                    X_aug, y_aug = collect_augmented_features(augmented, outer_fold_id, inner_fold_id)
                    if len(y_aug):
                        X_train = np.vstack([X_train, X_aug])
                        y_train = y_train + y_aug
                X_val, y_val = collect_original_features(feature_table, val_ids)
                metrics, _predicted, _proba = fit_predict_evaluate(config, X_train, y_train, X_val, y_val, labels)
                inner_metrics.append(metrics)
                validation_rows.append(
                    {
                        "baseline": baseline_name,
                        "outer_fold_id": outer_fold_id,
                        "inner_fold_id": inner_fold_id,
                        "config_index": config_index,
                        "config": config_name(config),
                        **metrics,
                    }
                )

            avg_macro_f1 = float(np.mean([row["macro_f1"] for row in inner_metrics])) if inner_metrics else 0.0
            avg_balanced_accuracy = (
                float(np.mean([row["balanced_accuracy"] for row in inner_metrics])) if inner_metrics else 0.0
            )
            config_scores.append(
                {
                    "config_index": config_index,
                    "config": config,
                    "config_name": config_name(config),
                    "avg_macro_f1": avg_macro_f1,
                    "avg_balanced_accuracy": avg_balanced_accuracy,
                }
            )

        selected = sorted(
            config_scores,
            key=lambda row: (-row["avg_macro_f1"], -row["avg_balanced_accuracy"], row["config_name"]),
        )[0]
        selected_rows.append(
            {
                "baseline": baseline_name,
                "outer_fold_id": outer_fold_id,
                "selected_config": selected["config_name"],
                "validation_macro_f1": round_float(selected["avg_macro_f1"]),
                "validation_balanced_accuracy": round_float(selected["avg_balanced_accuracy"]),
            }
        )

        development_ids = split_development_sample_ids(splits, outer_fold_id)
        test_ids = split_test_sample_ids(splits, outer_fold_id)
        X_train, y_train = collect_original_features(feature_table, development_ids)
        if augmented and augmented.rows:
            X_aug, y_aug = collect_augmented_features_for_outer(augmented, outer_fold_id)
            if len(y_aug):
                X_train = np.vstack([X_train, X_aug])
                y_train = y_train + y_aug
        X_test, y_test = collect_original_features(feature_table, test_ids)
        metrics, predicted, proba = fit_predict_evaluate(
            selected["config"],
            X_train,
            y_train,
            X_test,
            y_test,
            labels,
            measure_inference=True,
        )
        missing_test_labels = sorted(set(y_test) - set(y_train))
        outer_metric_rows.append(
            {
                "baseline": baseline_name,
                "outer_fold_id": outer_fold_id,
                "selected_config": selected["config_name"],
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "missing_test_labels_from_train": "|".join(missing_test_labels),
                **metrics,
            }
        )

        fold_confusion, fold_per_class = per_class_metrics(y_test, predicted, labels)
        confusion_total += fold_confusion
        for row in fold_per_class:
            per_class_rows.append({"baseline": baseline_name, "outer_fold_id": outer_fold_id, **row})

        for sample_id, true_label, predicted_label, row_proba in zip(test_ids, y_test, predicted, proba, strict=True):
            ranked = sorted(zip(labels, row_proba.tolist(), strict=True), key=lambda item: item[1], reverse=True)
            prediction_row: dict[str, Any] = {
                "baseline": baseline_name,
                "outer_fold_id": outer_fold_id,
                "sample_id": sample_id,
                "user_id": feature_table.meta_by_id[sample_id].get("user_id", ""),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "top1_score": round_float(ranked[0][1]) if ranked else 0.0,
                "top2_label": ranked[1][0] if len(ranked) > 1 else "",
                "top2_score": round_float(ranked[1][1]) if len(ranked) > 1 else "",
            }
            for label, score in zip(labels, row_proba.tolist(), strict=True):
                prediction_row[f"score_{label}"] = round_float(score)
            prediction_rows.append(prediction_row)

    write_csv(output_dir / "validation_scores.csv", validation_rows)
    write_csv(output_dir / "selected_models.csv", selected_rows)
    write_csv(output_dir / "outer_fold_metrics.csv", outer_metric_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "test_predictions.csv", prediction_rows)
    write_confusion_matrix(output_dir / "confusion_matrix.csv", confusion_total, labels)
    write_json(output_dir / "aggregate_metrics.json", aggregate_metrics(baseline_name, outer_metric_rows, labels))


def split_sample_ids(splits: list[dict[str, str]], outer_fold_id: int, inner_fold_id: int, role: str) -> list[str]:
    return [
        row["sample_id"]
        for row in splits
        if int(row["outer_fold_id"]) == outer_fold_id
        and int(row["inner_fold_id"]) == inner_fold_id
        and row["role"] == role
    ]


def split_development_sample_ids(splits: list[dict[str, str]], outer_fold_id: int) -> list[str]:
    return sorted(
        {
            row["sample_id"]
            for row in splits
            if int(row["outer_fold_id"]) == outer_fold_id and row["role"] != "test"
        }
    )


def split_test_sample_ids(splits: list[dict[str, str]], outer_fold_id: int) -> list[str]:
    return sorted(
        {
            row["sample_id"]
            for row in splits
            if int(row["outer_fold_id"]) == outer_fold_id and row["role"] == "test"
        }
    )


def collect_original_features(feature_table: FeatureTable, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    X = np.vstack([feature_table.X_by_id[sample_id] for sample_id in sample_ids])
    y = [feature_table.y_by_id[sample_id] for sample_id in sample_ids]
    return X, y


def collect_augmented_features(
    augmented: AugmentedTable,
    outer_fold_id: int,
    inner_fold_id: int,
) -> tuple[np.ndarray, list[str]]:
    rows = [
        row
        for row in augmented.rows
        if row["outer_fold_id"] == outer_fold_id and row["inner_fold_id"] == inner_fold_id
    ]
    if not rows:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack([row["X"] for row in rows]), [str(row["label"]) for row in rows]


def collect_augmented_features_for_outer(
    augmented: AugmentedTable,
    outer_fold_id: int,
) -> tuple[np.ndarray, list[str]]:
    rows = [row for row in augmented.rows if row["outer_fold_id"] == outer_fold_id]
    if not rows:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack([row["X"] for row in rows]), [str(row["label"]) for row in rows]


def fit_predict_evaluate(
    config: dict[str, Any],
    X_train: np.ndarray,
    y_train: list[str],
    X_eval: np.ndarray,
    y_eval: list[str],
    labels: list[str],
    measure_inference: bool = False,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    model = build_model(config).fit(X_train_scaled, y_train)
    model_proba = model.predict_proba(X_eval_scaled)
    proba = align_proba(model_proba, model.classes_, labels)
    predicted = [labels[index] for index in np.argmax(proba, axis=1)]
    metrics: dict[str, Any] = compute_metrics(y_eval, predicted, proba, labels)
    if measure_inference:
        metrics.update(inference_diagnostics(model, scaler, X_eval_scaled))
    return metrics, predicted, proba


def inference_diagnostics(model: Any, scaler: StandardScaler, X_eval_scaled: np.ndarray) -> dict[str, Any]:
    sample_latencies_ms = []
    for row in X_eval_scaled:
        start = time.perf_counter()
        model.predict_proba(row.reshape(1, -1))
        sample_latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "median_inference_ms": round_float(float(np.median(sample_latencies_ms))) if sample_latencies_ms else 0.0,
        "model_parameters": estimate_model_parameters(model),
        "estimated_model_memory_bytes": estimate_model_memory_bytes(model, scaler),
    }


def estimate_model_parameters(model: Any) -> int:
    total = 0
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += int(value.size)
        elif isinstance(value, list):
            total += sum(int(item.size) for item in value if isinstance(item, np.ndarray))
    return total


def estimate_model_memory_bytes(model: Any, scaler: StandardScaler) -> int:
    total = int(getattr(scaler, "mean_", np.array([], dtype=np.float64)).nbytes)
    total += int(getattr(scaler, "scale_", np.array([], dtype=np.float64)).nbytes)
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
        elif isinstance(value, list):
            total += sum(int(item.nbytes) for item in value if isinstance(item, np.ndarray))
    return total


def build_model(config: dict[str, Any]) -> Any:
    if config["model"] == "nearest_centroid":
        return NearestCentroid()
    if config["model"] == "ridge":
        return RidgeClassifier(alpha=float(config["alpha"]))
    if config["model"] == "knn":
        return KNNClassifier(k=int(config["k"]))
    if config["model"] == "gaussian_nb":
        return GaussianNB()
    raise ValueError(f"Unknown model config: {config}")


def config_name(config: dict[str, Any]) -> str:
    if config["model"] == "nearest_centroid":
        return "nearest_centroid"
    if config["model"] == "ridge":
        return f"ridge_alpha_{config['alpha']}"
    if config["model"] == "knn":
        return f"knn_k_{config['k']}"
    return str(config["model"])


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    proba: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    confusion, rows = per_class_metrics(y_true, y_pred, labels)
    supports = np.array([row["support"] for row in rows], dtype=np.float64)
    present = supports > 0
    f1_values = np.array([row["f1"] for row in rows], dtype=np.float64)
    recall_values = np.array([row["recall"] for row in rows], dtype=np.float64)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    top2 = top_k_accuracy(y_true, proba, labels, k=2)
    return {
        "accuracy": round_float(accuracy),
        "balanced_accuracy": round_float(float(recall_values[present].mean()) if present.any() else 0.0),
        "macro_f1": round_float(float(f1_values[present].mean()) if present.any() else 0.0),
        "top2_accuracy": round_float(top2),
    }


def per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    label_to_idx = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        confusion[label_to_idx[truth], label_to_idx[prediction]] += 1

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "support": support,
                "precision": round_float(precision),
                "recall": round_float(recall),
                "f1": round_float(f1),
            }
        )
    return confusion, rows


def top_k_accuracy(y_true: list[str], proba: np.ndarray, labels: list[str], k: int) -> float:
    if not y_true:
        return 0.0
    label_to_idx = {label: index for index, label in enumerate(labels)}
    top_indices = np.argsort(-proba, axis=1)[:, : min(k, len(labels))]
    hits = 0
    for row_index, label in enumerate(y_true):
        if label_to_idx[label] in top_indices[row_index]:
            hits += 1
    return hits / len(y_true)


def align_proba(proba: np.ndarray, model_labels: list[str], labels: list[str]) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    label_to_idx = {label: index for index, label in enumerate(labels)}
    for source_index, label in enumerate(model_labels):
        aligned[:, label_to_idx[label]] = proba[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    return aligned / row_sums


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sums = exp_scores.sum(axis=1, keepdims=True)
    sums[sums < 1e-12] = 1.0
    return exp_scores / sums


def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = (X * X).sum(axis=1, keepdims=True)
    Y_norm = (Y * Y).sum(axis=1, keepdims=True).T
    distances_sq = np.maximum(X_norm + Y_norm - 2.0 * X @ Y.T, 0.0)
    return np.sqrt(distances_sq)


def aggregate_metrics(baseline_name: str, outer_metric_rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "top2_accuracy")
    aggregate: dict[str, Any] = {
        "baseline": baseline_name,
        "outer_folds": len(outer_metric_rows),
        "labels": labels,
        "metrics": {},
    }
    for metric_name in metric_names:
        values = np.array([float(row[metric_name]) for row in outer_metric_rows], dtype=np.float64)
        aggregate["metrics"][metric_name] = {
            "mean": round_float(float(values.mean())) if len(values) else 0.0,
            "std": round_float(float(values.std(ddof=1))) if len(values) > 1 else 0.0,
            "min": round_float(float(values.min())) if len(values) else 0.0,
            "max": round_float(float(values.max())) if len(values) else 0.0,
        }
    aggregate["outer_folds_with_unseen_test_labels"] = [
        int(row["outer_fold_id"])
        for row in outer_metric_rows
        if row.get("missing_test_labels_from_train")
    ]
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion_matrix(path: Path, confusion: np.ndarray, labels: list[str]) -> None:
    rows = []
    for index, label in enumerate(labels):
        row = {"true_label": label}
        for pred_index, pred_label in enumerate(labels):
            row[f"pred_{pred_label}"] = int(confusion[index, pred_index])
        rows.append(row)
    write_csv(path, rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def round_float(value: float, places: int = 6) -> float:
    return round(float(value), places)
