#!/usr/bin/env python3
"""Build trajectory, sequence, and image features for procedure2 modeling.

Reads the paper-faithful preprocessing artifacts from
`procedure2/preprocess/artifacts/` (see that folder's `preprocess.py`) rather
than recomputing anything from `output/` directly:

  * `cleaned_trajectories.jsonl`: native, variable-length trajectories with
    the causal 8-D feature encoding (delta_p, delta_q, sin/cos alpha, sin/cos
    beta, same_stroke, diff_stroke) -> summary_features.csv.
  * `resampled_256.jsonl`: arc-length resampled, fixed T=256 sequences ->
    sequence_features_256.npz.
  * `manifest.csv`: matched PNG paths -> image_features.csv, for the subset
    of accepted samples that actually have a trajectory PNG (26 of 172 do
    not; see `sample_warnings.csv`).

No train-only augmentation exists yet for procedure2 (unlike procedure1), so
there is no augmented counterpart to any of these three artifacts.
"""

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
    "delta_p",
    "delta_q",
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
    preprocess_dir = script_path.parents[2] / "preprocess" / "artifacts"
    output_dir = script_path.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_by_id = {row["sample_id"]: row for row in load_csv(preprocess_dir / "manifest.csv")}
    cleaned = load_jsonl(preprocess_dir / "cleaned_trajectories.jsonl")
    resampled = load_jsonl(preprocess_dir / "resampled_256.jsonl")

    summary_rows = [summary_row(sample, manifest_by_id) for sample in cleaned]
    write_csv(output_dir / "summary_features.csv", summary_rows)

    write_sequence_npz(output_dir / "sequence_features_256.npz", resampled)
    write_sequence_index(output_dir / "sequence_index.csv", resampled, manifest_by_id)

    image_rows, image_stack = build_image_features(repo_root, cleaned, manifest_by_id)
    write_csv(output_dir / "image_features.csv", image_rows)
    if image_rows:
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
            "summary_feature_count": len([key for key in summary_rows[0] if key not in meta_columns()]) if summary_rows else 0,
            "sequence_shape": [len(resampled), 256, len(SEQUENCE_FEATURE_NAMES)],
            "image_shape": list(image_stack.shape) if image_rows else [0, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE],
            "image_samples_with_png": len(image_rows),
            "image_samples_missing_png": len(cleaned) - len(image_rows),
            "labels": labels,
            "sequence_feature_names": SEQUENCE_FEATURE_NAMES,
        },
    )

    print(f"Wrote feature artifacts to {output_dir}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Sequence rows: {len(resampled)}")
    print(f"Image rows: {len(image_rows)} (missing PNG: {len(cleaned) - len(image_rows)})")
    return 0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summary_row(sample: dict[str, Any], manifest_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    manifest_row = manifest_by_id.get(sample["sample_id"], {})
    metadata = {
        "sample_id": sample["sample_id"],
        "user_id": sample["user_id"],
        "label": sample["label"],
        "created_at": manifest_row.get("created_at", ""),
        "json_path": manifest_row.get("json_path", ""),
        "image_path": manifest_row.get("image_path", ""),
        "video_path": manifest_row.get("video_path", ""),
    }
    feature_values = trajectory_summary_features(
        sample["points"], sample["stroke_ids"], sample["features_8d"], sample["feature_names"]
    )
    return {**metadata, **feature_values}


def trajectory_summary_features(
    points_raw: list[list[float]],
    strokes_raw: list[int],
    features_8d: list[list[float]],
    feature_names: list[str],
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
    stroke_ids_sorted = sorted(set(strokes.tolist()))
    stroke_lengths = np.array([np.sum(strokes == stroke_id) for stroke_id in stroke_ids_sorted], dtype=np.float64)
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
        # inter-stroke gap: number of stroke transitions relative to the trajectory,
        # a proxy for total dwell/gap time between strokes (frame-index units).
        "stroke_transition_count": float(max(0, len(stroke_lengths) - 1)),
        "longest_stroke_ratio": float(stroke_lengths.max() / len(points)) if len(stroke_lengths) else 0.0,
        "single_point_strokes": float(np.sum(stroke_lengths == 1)) if len(stroke_lengths) else 0.0,
        # dwell_step_ratio: fraction of within-stroke steps below a near-zero
        # displacement threshold, i.e. points where the pen dwelled in place.
        "dwell_step_ratio": float(np.mean(same_stroke_steps < 0.002)) if len(same_stroke_steps) else 0.0,
    }
    feature_values.update(prefix_stats("step", steps))
    feature_values.update(prefix_stats("same_stroke_step", same_stroke_steps))
    feature_values.update(prefix_stats("acceleration", acceleration))
    feature_values.update(prefix_stats("curvature", curvature))
    feature_values.update(prefix_stats("stroke_duration_points", stroke_lengths))
    if features.ndim == 2 and features.shape[1] == len(feature_names):
        for feature_index, feature_name in enumerate(feature_names):
            feature_values.update(prefix_stats(f"feature8d_{feature_name}", features[:, feature_index]))
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


def write_sequence_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    sequences = []
    sample_ids = []
    labels = []
    users = []
    for row in rows:
        sequences.append(sequence_features(row))
        sample_ids.append(row["sample_id"])
        labels.append(row["label"])
        users.append(row["user_id"])

    np.savez_compressed(
        path,
        X=np.asarray(sequences, dtype=np.float32),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels),
        users=np.asarray(users),
    )


def write_sequence_index(path: Path, rows: list[dict[str, Any]], manifest_by_id: dict[str, dict[str, str]]) -> None:
    index_rows = []
    for row in rows:
        manifest_row = manifest_by_id.get(row["sample_id"], {})
        index_rows.append(
            {
                "sample_id": row["sample_id"],
                "user_id": row["user_id"],
                "label": row["label"],
                "created_at": manifest_row.get("created_at", ""),
            }
        )
    write_csv(path, index_rows)


def sequence_features(row: dict[str, Any]) -> np.ndarray:
    points = np.asarray(row["points"], dtype=np.float64)
    strokes = np.asarray(row["stroke_ids"], dtype=np.float64)
    features_8d = np.asarray(row["features_8d"], dtype=np.float64)
    if len(points) != 256:
        raise ValueError(f"Expected 256 resampled points in {row.get('sample_id')}")
    max_stroke = max(float(strokes.max()), 1.0) if len(strokes) else 1.0
    stroke_norm = (strokes / max_stroke).reshape((-1, 1))
    return np.hstack([points, stroke_norm, features_8d])


def build_image_features(
    repo_root: Path,
    rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    feature_rows = []
    images = []
    for row in rows:
        manifest_row = manifest_by_id.get(row["sample_id"], {})
        image_path_str = manifest_row.get("image_path", "")
        if not image_path_str:
            continue
        image_path = repo_root / image_path_str
        if not image_path.exists():
            continue
        image = load_ink_image(image_path)
        images.append(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        feature_rows.append(
            {
                "sample_id": row["sample_id"],
                "user_id": row["user_id"],
                "label": row["label"],
                "created_at": manifest_row.get("created_at", ""),
                "image_path": image_path_str,
                **image_descriptor_features(image),
            }
        )
    if not feature_rows:
        return [], np.empty((0, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), dtype=np.uint8)
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
        "user_id",
        "label",
        "created_at",
        "json_path",
        "image_path",
        "video_path",
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
