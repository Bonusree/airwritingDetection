#!/usr/bin/env python3
"""Build trajectory, sequence, and image features for procedure1 modeling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TARGET_IMAGE_SIZE = 224
SEQUENCE_FEATURE_NAMES = [
    "x",
    "y",
    "stroke_id_norm",
    "delta_x",
    "delta_y",
    "sin_alpha",
    "cos_alpha",
    "sin_beta",
    "cos_beta",
    "same_stroke",
    "diff_stroke",
]


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    procedure_root = repo_root / "procedure1"
    source_dir = procedure_root / "artifacts"
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = load_jsonl(source_dir / "cleaned_trajectories.jsonl")
    augmented = load_jsonl(source_dir / "augmented_train_trajectories.jsonl")

    summary_rows = [summary_row(sample, augmented=False) for sample in cleaned]
    augmented_summary_rows = [summary_row(sample, augmented=True) for sample in augmented]
    write_csv(output_dir / "summary_features.csv", summary_rows)
    write_csv(output_dir / "augmented_summary_features.csv", augmented_summary_rows)

    write_sequence_npz(output_dir / "sequence_features_128.npz", cleaned, augmented=False)
    write_sequence_index(output_dir / "sequence_index.csv", cleaned, augmented=False)
    write_sequence_npz(output_dir / "augmented_sequence_features_128.npz", augmented, augmented=True)
    write_sequence_index(output_dir / "augmented_sequence_index.csv", augmented, augmented=True)

    image_rows, image_stack = build_image_features(repo_root, cleaned)
    write_csv(output_dir / "image_features.csv", image_rows)
    np.savez_compressed(
        output_dir / "image_224_uint8.npz",
        X=image_stack,
        sample_ids=np.array([row["sample_id"] for row in image_rows]),
        labels=np.array([row["label"] for row in image_rows]),
        users=np.array([row["user_id"] for row in image_rows]),
    )

    labels = sorted({sample["label"] for sample in cleaned})
    write_json(
        output_dir / "feature_summary.json",
        {
            "clean_samples": len(cleaned),
            "augmented_train_rows": len(augmented),
            "summary_feature_count": len([key for key in summary_rows[0] if key not in meta_columns()]) if summary_rows else 0,
            "sequence_shape": [len(cleaned), 128, len(SEQUENCE_FEATURE_NAMES)],
            "image_shape": list(image_stack.shape),
            "labels": labels,
            "sequence_feature_names": SEQUENCE_FEATURE_NAMES,
        },
    )

    print(f"Wrote feature artifacts to {output_dir}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Augmented summary rows: {len(augmented_summary_rows)}")
    print(f"Image rows: {len(image_rows)}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summary_row(sample: dict[str, Any], *, augmented: bool) -> dict[str, Any]:
    if augmented:
        trajectory = sample["trajectory"]
        points = trajectory["points_augmented_variable"]
        strokes = trajectory["stroke_ids_variable"]
        features_8d = trajectory["features_8d_variable"]
        metadata = {
            "sample_id": sample["augmented_sample_id"],
            "augmented_sample_id": sample["augmented_sample_id"],
            "source_sample_id": sample["source_sample_id"],
            "user_id": sample["user_id"],
            "label": sample["label"],
            "role": sample["role"],
            "outer_fold_id": sample["outer_fold_id"],
            "inner_fold_id": sample["inner_fold_id"],
        }
    else:
        trajectory = sample["trajectory"]
        points = trajectory["bbox_normalized_points"]
        strokes = trajectory["stroke_ids"]
        features_8d = trajectory["features_8d_bbox_normalized"]
        metadata = {
            "sample_id": sample["sample_id"],
            "user_id": sample["user_id"],
            "label": sample["label"],
            "created_at": sample["created_at"],
            "device": sample["device"],
            "json_path": sample["source"]["json_path"],
            "image_path": sample["source"]["image_path"],
            "video_path": sample["source"]["video_path"],
        }

    feature_values = trajectory_summary_features(points, strokes, features_8d)
    return {**metadata, **feature_values}


def trajectory_summary_features(
    points_raw: list[list[float]],
    strokes_raw: list[int],
    features_8d: list[list[float]],
) -> dict[str, float]:
    points = np.asarray(points_raw, dtype=np.float64)
    strokes = np.asarray(strokes_raw, dtype=np.int32)
    features = np.asarray(features_8d, dtype=np.float64)
    if len(points) == 0:
        return {}

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([], dtype=np.float64)
    same_stroke_steps = steps[strokes[1:] == strokes[:-1]] if len(points) > 1 else np.array([], dtype=np.float64)
    deltas = np.diff(points, axis=0) if len(points) > 1 else np.empty((0, 2), dtype=np.float64)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0]) if len(deltas) else np.array([], dtype=np.float64)
    angle_change = wrap_angle(np.diff(angles)) if len(angles) > 1 else np.array([], dtype=np.float64)
    curvature = np.abs(angle_change)
    acceleration = np.diff(steps) if len(steps) > 1 else np.array([], dtype=np.float64)
    stroke_lengths = np.array([np.sum(strokes == stroke_id) for stroke_id in sorted(set(strokes.tolist()))], dtype=np.float64)
    xs = points[:, 0]
    ys = points[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    path_length = float(steps.sum()) if len(steps) else 0.0
    straight_line = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    feature_values: dict[str, float] = {
        "num_points": float(len(points)),
        "num_strokes": float(len(stroke_lengths)),
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "bbox_aspect": width / (height + 1e-9),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "std_x": float(xs.std()),
        "std_y": float(ys.std()),
        "start_x": float(points[0, 0]),
        "start_y": float(points[0, 1]),
        "end_x": float(points[-1, 0]),
        "end_y": float(points[-1, 1]),
        "delta_x": float(points[-1, 0] - points[0, 0]),
        "delta_y": float(points[-1, 1] - points[0, 1]),
        "path_length": path_length,
        "straight_line_distance": straight_line,
        "straightness": straight_line / (path_length + 1e-9),
        "stroke_transition_count": float(max(0, len(stroke_lengths) - 1)),
        "longest_stroke_ratio": float(stroke_lengths.max() / len(points)) if len(stroke_lengths) else 0.0,
        "single_point_strokes": float(np.sum(stroke_lengths == 1)) if len(stroke_lengths) else 0.0,
        "dwell_step_ratio": float(np.mean(same_stroke_steps < 0.002)) if len(same_stroke_steps) else 0.0,
    }
    feature_values.update(prefix_stats("step", steps))
    feature_values.update(prefix_stats("same_stroke_step", same_stroke_steps))
    feature_values.update(prefix_stats("acceleration", acceleration))
    feature_values.update(prefix_stats("curvature", curvature))
    feature_values.update(prefix_stats("stroke_duration_points", stroke_lengths))
    if features.ndim == 2 and features.shape[1] == 8:
        for feature_index in range(8):
            feature_values.update(prefix_stats(f"feature8d_{feature_index}", features[:, feature_index]))
    return {key: round_float(value) for key, value in feature_values.items()}


def prefix_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def write_sequence_npz(path: Path, rows: list[dict[str, Any]], *, augmented: bool) -> None:
    sequences = []
    sample_ids = []
    source_sample_ids = []
    outer_fold_ids = []
    inner_fold_ids = []
    labels = []
    users = []
    for row in rows:
        sequence = sequence_features(row, augmented=augmented)
        sequences.append(sequence)
        if augmented:
            sample_ids.append(row["augmented_sample_id"])
            source_sample_ids.append(row["source_sample_id"])
            outer_fold_ids.append(int(row["outer_fold_id"]))
            inner_fold_ids.append(int(row["inner_fold_id"]))
        else:
            sample_ids.append(row["sample_id"])
        labels.append(row["label"])
        users.append(row["user_id"])

    payload = {
        "X": np.asarray(sequences, dtype=np.float32),
        "sample_ids": np.asarray(sample_ids),
        "labels": np.asarray(labels),
        "users": np.asarray(users),
    }
    if augmented:
        payload["source_sample_ids"] = np.asarray(source_sample_ids)
        payload["outer_fold_ids"] = np.asarray(outer_fold_ids, dtype=np.int32)
        payload["inner_fold_ids"] = np.asarray(inner_fold_ids, dtype=np.int32)
    np.savez_compressed(path, **payload)


def write_sequence_index(path: Path, rows: list[dict[str, Any]], *, augmented: bool) -> None:
    index_rows = []
    for row in rows:
        if augmented:
            index_rows.append(
                {
                    "sample_id": row["augmented_sample_id"],
                    "source_sample_id": row["source_sample_id"],
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "role": row["role"],
                    "outer_fold_id": row["outer_fold_id"],
                    "inner_fold_id": row["inner_fold_id"],
                }
            )
        else:
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "created_at": row["created_at"],
                    "device": row["device"],
                }
            )
    write_csv(path, index_rows)


def sequence_features(row: dict[str, Any], *, augmented: bool) -> np.ndarray:
    trajectory = row["trajectory"]
    resampled = trajectory["resampled_128"]
    points = np.asarray(resampled["points"], dtype=np.float64)
    strokes = np.asarray(resampled["stroke_ids"], dtype=np.float64)
    features_8d = np.asarray(resampled["features_8d"], dtype=np.float64)
    if len(points) != 128:
        raise ValueError(f"Expected 128 resampled points in {row.get('sample_id') or row.get('augmented_sample_id')}")
    max_stroke = max(float(strokes.max()), 1.0) if len(strokes) else 1.0
    stroke_norm = (strokes / max_stroke).reshape((-1, 1))
    return np.hstack([points, stroke_norm, features_8d])


def build_image_features(repo_root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    feature_rows = []
    images = []
    for row in rows:
        image_path = repo_root / row["source"]["image_path"]
        image = load_ink_image(image_path)
        images.append(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        feature_rows.append(
            {
                "sample_id": row["sample_id"],
                "user_id": row["user_id"],
                "label": row["label"],
                "created_at": row["created_at"],
                "device": row["device"],
                "image_path": row["source"]["image_path"],
                **image_descriptor_features(image),
            }
        )
    return feature_rows, np.stack(images, axis=0)


def load_ink_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    image = image.resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ink = alpha * (1.0 - luminance)
    if float(ink.max()) < 0.05 and float(alpha.max()) > 0.05:
        ink = alpha
    if float(ink.max()) < 0.05:
        ink = 1.0 - luminance
    return np.clip(ink, 0.0, 1.0)


def image_descriptor_features(image: np.ndarray) -> dict[str, float]:
    mask = image > 0.05
    rows, cols = np.indices(image.shape)
    total = float(image.sum())
    features: dict[str, float] = {
        "ink_density": float(mask.mean()),
        "ink_intensity_mean": float(image.mean()),
        "ink_intensity_std": float(image.std()),
        "ink_intensity_sum": total,
    }
    if total > 1e-9:
        cx = float((cols * image).sum() / total) / (image.shape[1] - 1)
        cy = float((rows * image).sum() / total) / (image.shape[0] - 1)
        sx = float(np.sqrt((((cols / (image.shape[1] - 1)) - cx) ** 2 * image).sum() / total))
        sy = float(np.sqrt((((rows / (image.shape[0] - 1)) - cy) ** 2 * image).sum() / total))
        ys, xs = np.where(mask)
        bbox_w = float((xs.max() - xs.min() + 1) / image.shape[1]) if len(xs) else 0.0
        bbox_h = float((ys.max() - ys.min() + 1) / image.shape[0]) if len(ys) else 0.0
    else:
        cx = cy = sx = sy = bbox_w = bbox_h = 0.0
    features.update(
        {
            "image_centroid_x": cx,
            "image_centroid_y": cy,
            "image_std_x": sx,
            "image_std_y": sy,
            "image_bbox_width": bbox_w,
            "image_bbox_height": bbox_h,
            "image_bbox_aspect": bbox_w / (bbox_h + 1e-9),
        }
    )

    for index, value in enumerate(np.array_split(image.sum(axis=0), 16)):
        features[f"proj_x_{index:02d}"] = float(value.sum() / (total + 1e-9))
    for index, value in enumerate(np.array_split(image.sum(axis=1), 16)):
        features[f"proj_y_{index:02d}"] = float(value.sum() / (total + 1e-9))
    for row_index, row_band in enumerate(np.array_split(image, 7, axis=0)):
        for col_index, zone in enumerate(np.array_split(row_band, 7, axis=1)):
            features[f"zone_{row_index}_{col_index}"] = float(zone.mean())
    return {key: round_float(value) for key, value in features.items()}


def meta_columns() -> set[str]:
    return {
        "sample_id",
        "augmented_sample_id",
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
    }


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
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def round_float(value: float, places: int = 6) -> float:
    return round(float(value), places)


if __name__ == "__main__":
    raise SystemExit(main())
