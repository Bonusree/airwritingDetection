#!/usr/bin/env python3
"""Procedure 1 data readiness pipeline for the air-writing dataset.

This script implements the first two steps from the project plan:

1. Immediate preprocessing: inventory, manifest creation, triplet matching,
   JSON/trajectory validation, label checks, exclusions, and user-level folds.
2. Cleaning and augmentation: trajectory cleaning, bbox normalization,
   128-point resampling, duplicate detection, weighted-sampling tables, and
   fold-scoped train-only augmentation metadata/JSONL.

The implementation intentionally uses only Python's standard library so it can
run in this repository without installing the modeling stack first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BANGLA_VOWELS = tuple(
    chr(code)
    for code in (
        0x0985,
        0x0986,
        0x0987,
        0x0988,
        0x0989,
        0x098A,
        0x098B,
        0x098F,
        0x0990,
        0x0993,
        0x0994,
    )
)

VIDEO_EXTENSIONS = (".webm", ".mp4")
POINT_EPSILON = 1e-7
MIN_VALID_POINTS = 10
TARGET_POINTS = 128


@dataclass
class Candidate:
    json_path: Path
    path_user: str
    path_label: str
    data: dict[str, Any] | None = None
    sample_id: str = ""
    user_id: str = ""
    label: str = ""
    created_at: str = ""
    user_agent: str = ""
    device: str = ""
    image_path: Path | None = None
    video_path: Path | None = None
    raw_points: list[tuple[float, float]] = field(default_factory=list)
    raw_strokes: list[int] = field(default_factory=list)
    clean_points: list[tuple[float, float]] = field(default_factory=list)
    clean_strokes: list[int] = field(default_factory=list)
    bbox_points: list[tuple[float, float]] = field(default_factory=list)
    resampled_points: list[tuple[float, float]] = field(default_factory=list)
    resampled_strokes: list[int] = field(default_factory=list)
    clean_features: list[list[float]] = field(default_factory=list)
    bbox_features: list[list[float]] = field(default_factory=list)
    resampled_features: list[list[float]] = field(default_factory=list)
    normalization: dict[str, float] = field(default_factory=dict)
    cleaning: dict[str, int | float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    exclusions: list[tuple[str, str]] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    duplicate_of: str = ""
    duplicate_type: str = ""
    duplicate_distance: float | None = None
    test_fold_id: int | None = None

    @property
    def is_excluded(self) -> bool:
        return bool(self.exclusions)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(repo_root, input_dir, args)
    mark_duplicate_sample_ids(candidates)

    prelim_clean = [candidate for candidate in candidates if not candidate.is_excluded]
    mark_exact_duplicates(prelim_clean)
    prelim_clean = [candidate for candidate in candidates if not candidate.is_excluded]
    mark_near_duplicates(prelim_clean, args.near_duplicate_threshold)

    clean_samples = [candidate for candidate in candidates if not candidate.is_excluded]
    assign_user_folds(clean_samples)
    nested_splits = build_nested_split_assignments(clean_samples)

    rng = random.Random(args.seed)
    augmentations, video_plans = build_fold_scoped_augmentations(
        clean_samples,
        nested_splits,
        augmentations_per_sample=args.augmentations_per_sample,
        rng=rng,
    )
    weights = build_training_weights(clean_samples, nested_splits)

    write_outputs(
        repo_root=repo_root,
        input_dir=input_dir,
        output_dir=output_dir,
        candidates=candidates,
        clean_samples=clean_samples,
        nested_splits=nested_splits,
        augmentations=augmentations,
        video_plans=video_plans,
        weights=weights,
        args=args,
    )

    print_summary(output_dir, candidates, clean_samples, augmentations)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest, clean trajectories, assign user folds, and create fold-scoped augmentation artifacts."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--input-dir", default="output", help="Raw dataset directory relative to repo root.")
    parser.add_argument(
        "--output-dir",
        default="procedure1/artifacts",
        help="Artifact directory relative to repo root.",
    )
    parser.add_argument(
        "--augmentations-per-sample",
        type=int,
        default=3,
        help="Trajectory augmentation variants per source sample per nested train split.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.003,
        help="Mean DTW distance threshold for near duplicates within the same user and label.",
    )
    parser.add_argument(
        "--jump-abs-threshold",
        type=float,
        default=0.35,
        help="Absolute normalized step length considered a severe tracking jump.",
    )
    parser.add_argument(
        "--jump-median-factor",
        type=float,
        default=12.0,
        help="Median same-stroke step multiplier used with --jump-abs-threshold.",
    )
    parser.add_argument("--seed", type=int, default=20260802, help="Deterministic augmentation seed.")
    return parser.parse_args()


def load_candidates(repo_root: Path, input_dir: Path, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for json_path in sorted(input_dir.glob("*/*/*.json")):
        path_label = normalize_text(json_path.parent.name)
        path_user = normalize_text(json_path.parent.parent.name)
        candidate = Candidate(json_path=json_path, path_user=path_user, path_label=path_label)
        read_and_validate_candidate(candidate, repo_root, args)
        candidates.append(candidate)
    return candidates


def read_and_validate_candidate(candidate: Candidate, repo_root: Path, args: argparse.Namespace) -> None:
    try:
        with candidate.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - report exact file-level parse failure.
        candidate.sample_id = candidate.json_path.stem
        candidate.user_id = candidate.path_user
        candidate.label = candidate.path_label
        candidate.exclusions.append(("corrupt_json", str(exc)))
        return

    if not isinstance(data, dict):
        candidate.sample_id = candidate.json_path.stem
        candidate.user_id = candidate.path_user
        candidate.label = candidate.path_label
        candidate.exclusions.append(("corrupt_json", "top-level JSON is not an object"))
        return

    candidate.data = data
    candidate.sample_id = normalize_text(data.get("sample_id") or data.get("sampleId") or candidate.json_path.stem)
    candidate.user_id = normalize_text(data.get("username") or data.get("user_id") or candidate.path_user)
    candidate.label = normalize_text(data.get("label") or candidate.path_label)
    candidate.created_at = normalize_text(data.get("created_at") or data.get("createdAt") or "")
    candidate.user_agent = normalize_text(data.get("user_agent") or data.get("userAgent") or "")
    candidate.device = parse_device(candidate.user_agent)
    candidate.image_path = find_companion_file(candidate, ".png")
    candidate.video_path = find_companion_file(candidate, VIDEO_EXTENSIONS)
    candidate.hashes = file_hashes(candidate)

    required_fields = ("sample_id", "username", "label", "frame_size", "trajectory", "user_agent")
    missing = [name for name in required_fields if name not in data or data.get(name) in (None, "")]
    if missing:
        candidate.exclusions.append(("missing_required_fields", ",".join(missing)))

    if candidate.user_id != candidate.path_user:
        candidate.warnings.append(f"path_user_mismatch:{candidate.path_user}")
    if candidate.label != candidate.path_label:
        candidate.exclusions.append(
            ("path_label_mismatch", f"path={candidate.path_label};json={candidate.label}")
        )
    if candidate.label not in BANGLA_VOWELS:
        candidate.exclusions.append(("unknown_label", f"label={candidate.label}"))

    frame_size = data.get("frame_size")
    if not valid_frame_size(frame_size):
        candidate.exclusions.append(("invalid_frame_size", repr(frame_size)))

    if candidate.image_path is None:
        candidate.exclusions.append(("missing_png", "no trajectory PNG matched this JSON"))
    if candidate.video_path is None:
        candidate.warnings.append("missing_video")

    validate_and_clean_trajectory(candidate, args)


def validate_and_clean_trajectory(candidate: Candidate, args: argparse.Namespace) -> None:
    trajectory = candidate.data.get("trajectory") if candidate.data else None
    if not isinstance(trajectory, dict):
        candidate.exclusions.append(("invalid_trajectory", "trajectory is missing or not an object"))
        return

    points_raw = trajectory.get("points_normalized")
    strokes_raw = trajectory.get("stroke_ids")
    features_raw = trajectory.get("features_8d")
    num_points_raw = trajectory.get("num_points")

    if not isinstance(points_raw, list):
        candidate.exclusions.append(("invalid_trajectory", "trajectory.points_normalized is not a list"))
        return
    if not isinstance(strokes_raw, list):
        candidate.exclusions.append(("invalid_trajectory", "trajectory.stroke_ids is not a list"))
        return
    if not isinstance(features_raw, list):
        candidate.exclusions.append(("invalid_trajectory", "trajectory.features_8d is not a list"))
        return
    if isinstance(num_points_raw, int) and num_points_raw != len(points_raw):
        candidate.exclusions.append(
            ("num_points_mismatch", f"num_points={num_points_raw};points={len(points_raw)}")
        )
    if len(strokes_raw) != len(points_raw):
        candidate.exclusions.append(
            ("stroke_length_mismatch", f"strokes={len(strokes_raw)};points={len(points_raw)}")
        )
    if len(features_raw) != len(points_raw):
        candidate.exclusions.append(
            ("features_length_mismatch", f"features={len(features_raw)};points={len(points_raw)}")
        )

    if len(strokes_raw) != len(points_raw) or len(features_raw) != len(points_raw):
        return

    parsed_points: list[tuple[float, float]] = []
    parsed_strokes: list[int] = []
    dropped_invalid = 0
    invalid_feature_rows = 0
    out_of_bounds = 0

    for point, stroke, features in zip(points_raw, strokes_raw, features_raw, strict=True):
        parsed_point = parse_point(point)
        stroke_id = parse_stroke_id(stroke)
        feature_ok = valid_feature_row(features)
        if not feature_ok:
            invalid_feature_rows += 1
        if parsed_point is None or stroke_id is None or not feature_ok:
            dropped_invalid += 1
            continue
        x, y = parsed_point
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            dropped_invalid += 1
            out_of_bounds += 1
            continue
        parsed_points.append((x, y))
        parsed_strokes.append(stroke_id)

    candidate.raw_points = parsed_points
    candidate.raw_strokes = parsed_strokes
    if dropped_invalid:
        candidate.warnings.append(f"dropped_invalid_points:{dropped_invalid}")
    if invalid_feature_rows:
        candidate.warnings.append(f"invalid_feature_rows:{invalid_feature_rows}")
    if out_of_bounds:
        candidate.warnings.append(f"out_of_bounds_points:{out_of_bounds}")

    cleaned_points, cleaned_strokes, cleaning_stats = clean_points(
        parsed_points,
        parsed_strokes,
        jump_abs_threshold=args.jump_abs_threshold,
        jump_median_factor=args.jump_median_factor,
    )
    cleaning_stats["dropped_invalid"] = dropped_invalid
    candidate.clean_points = cleaned_points
    candidate.clean_strokes = cleaned_strokes
    candidate.cleaning = cleaning_stats

    if len(cleaned_points) < MIN_VALID_POINTS:
        candidate.exclusions.append(
            ("fewer_than_10_valid_points", f"valid_points={len(cleaned_points)}")
        )
        return
    if max_side(cleaned_points) <= POINT_EPSILON:
        candidate.exclusions.append(("zero_length_sample", "bounding box side length is zero"))
        return

    bbox_points, normalization = bbox_normalize(cleaned_points)
    candidate.bbox_points = bbox_points
    candidate.normalization = normalization
    candidate.clean_features = compute_features(cleaned_points, cleaned_strokes)
    candidate.bbox_features = compute_features(bbox_points, cleaned_strokes)
    candidate.resampled_points, candidate.resampled_strokes = resample_by_stroke(
        bbox_points,
        cleaned_strokes,
        TARGET_POINTS,
    )
    candidate.resampled_features = compute_features(candidate.resampled_points, candidate.resampled_strokes)
    candidate.hashes["trajectory_cleaned"] = trajectory_hash(candidate.bbox_points, candidate.clean_strokes)


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def parse_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def parse_stroke_id(value: Any) -> int | None:
    try:
        stroke_id = int(value)
    except (TypeError, ValueError):
        return None
    return stroke_id


def valid_feature_row(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 8:
        return False
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
    return True


def valid_frame_size(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        width = float(value[0])
        height = float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0


def parse_device(user_agent: str) -> str:
    ua = user_agent.lower()
    if not ua:
        return "unknown"
    if "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua:
        os_name = "iOS"
    elif "windows" in ua:
        os_name = "Windows"
    elif "mac os x" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "linux" in ua or "x11" in ua:
        os_name = "Linux"
    else:
        os_name = "UnknownOS"

    if "edg/" in ua:
        browser = "Edge"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "chrome/" in ua or "chromium/" in ua:
        browser = "Chrome"
    elif "safari/" in ua:
        browser = "Safari"
    else:
        browser = "UnknownBrowser"

    form = "mobile" if "mobile" in ua or "android" in ua else "desktop"
    return f"{os_name} {browser} {form}"


def find_companion_file(candidate: Candidate, extensions: str | tuple[str, ...]) -> Path | None:
    if isinstance(extensions, str):
        extensions = (extensions,)

    directory = candidate.json_path.parent
    media = candidate.data.get("media", {}) if candidate.data else {}
    media_names: list[str] = []
    if ".png" in extensions and isinstance(media, dict) and media.get("image"):
        media_names.append(str(media["image"]))
    if any(ext in VIDEO_EXTENSIONS for ext in extensions) and isinstance(media, dict) and media.get("video"):
        media_names.append(str(media["video"]))

    candidates: list[Path] = []
    for ext in extensions:
        candidates.append(candidate.json_path.with_suffix(ext))
        candidates.append(directory / f"{candidate.label}_{candidate.sample_id}{ext}")
        candidates.append(directory / f"{candidate.sample_id}{ext}")
    for name in media_names:
        candidates.append(directory / name)

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    for ext in extensions:
        matches = sorted(directory.glob(f"*{candidate.sample_id}*{ext}"))
        if matches:
            return matches[0]
    return None


def file_hashes(candidate: Candidate) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, path in (
        ("json", candidate.json_path),
        ("image", candidate.image_path),
        ("video", candidate.video_path),
    ):
        if path is None:
            continue
        hashes[key] = sha256_file(path)
    return hashes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_points(
    points: list[tuple[float, float]],
    strokes: list[int],
    *,
    jump_abs_threshold: float,
    jump_median_factor: float,
) -> tuple[list[tuple[float, float]], list[int], dict[str, int | float]]:
    if not points:
        return [], [], {
            "raw_points": 0,
            "clean_points": 0,
            "dropped_repeated": 0,
            "dropped_jumps": 0,
            "jump_threshold": jump_abs_threshold,
        }

    dedup_points: list[tuple[float, float]] = []
    dedup_strokes: list[int] = []
    dropped_repeated = 0
    prev_point: tuple[float, float] | None = None
    prev_stroke: int | None = None
    for point, stroke in zip(points, strokes, strict=True):
        if prev_point is not None and stroke == prev_stroke and euclidean(point, prev_point) <= POINT_EPSILON:
            dropped_repeated += 1
            continue
        dedup_points.append(point)
        dedup_strokes.append(stroke)
        prev_point = point
        prev_stroke = stroke

    same_stroke_steps = [
        euclidean(dedup_points[index], dedup_points[index - 1])
        for index in range(1, len(dedup_points))
        if dedup_strokes[index] == dedup_strokes[index - 1]
        and euclidean(dedup_points[index], dedup_points[index - 1]) > POINT_EPSILON
    ]
    if same_stroke_steps:
        median_step = statistics.median(same_stroke_steps)
        jump_threshold = max(jump_abs_threshold, median_step * jump_median_factor)
    else:
        median_step = 0.0
        jump_threshold = jump_abs_threshold

    jump_points: list[tuple[float, float]] = []
    jump_strokes: list[int] = []
    dropped_jumps = 0
    for point, stroke in zip(dedup_points, dedup_strokes, strict=True):
        if (
            jump_points
            and stroke == jump_strokes[-1]
            and euclidean(point, jump_points[-1]) > jump_threshold
        ):
            dropped_jumps += 1
            continue
        jump_points.append(point)
        jump_strokes.append(stroke)

    return jump_points, jump_strokes, {
        "raw_points": len(points),
        "clean_points": len(jump_points),
        "dropped_repeated": dropped_repeated,
        "dropped_jumps": dropped_jumps,
        "jump_threshold": round_float(jump_threshold),
        "median_step": round_float(median_step),
    }


def euclidean(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def max_side(points: list[tuple[float, float]]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def bbox_normalize(
    points: list[tuple[float, float]]
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    scale = max(width, height)
    normalized = [((x - center_x) / scale, (y - center_y) / scale) for x, y in points]
    return normalized, {
        "bbox_min_x": round_float(min_x),
        "bbox_max_x": round_float(max_x),
        "bbox_min_y": round_float(min_y),
        "bbox_max_y": round_float(max_y),
        "bbox_width": round_float(width),
        "bbox_height": round_float(height),
        "bbox_center_x": round_float(center_x),
        "bbox_center_y": round_float(center_y),
        "bbox_scale": round_float(scale),
    }


def compute_features(points: list[tuple[float, float]], strokes: list[int]) -> list[list[float]]:
    features: list[list[float]] = []
    for index, (x, y) in enumerate(points):
        dx = 0.0
        dy = 0.0
        sin_alpha = 0.0
        cos_alpha = 1.0
        sin_beta = 0.0
        cos_beta = 1.0

        if index > 0 and strokes[index] == strokes[index - 1]:
            prev_x, prev_y = points[index - 1]
            dx = x - prev_x
            dy = y - prev_y
            distance = math.hypot(dx, dy) + 1e-9
            cos_alpha = dx / distance
            sin_alpha = dy / distance

            if index >= 2 and strokes[index - 1] == strokes[index - 2]:
                prev_prev_x, prev_prev_y = points[index - 2]
                dx_prev = prev_x - prev_prev_x
                dy_prev = prev_y - prev_prev_y
                distance_prev = math.hypot(dx_prev, dy_prev) + 1e-9
                cos_alpha_prev = dx_prev / distance_prev
                sin_alpha_prev = dy_prev / distance_prev
                sin_beta = sin_alpha * cos_alpha_prev - cos_alpha * sin_alpha_prev
                cos_beta = cos_alpha * cos_alpha_prev + sin_alpha * sin_alpha_prev

        same_stroke = 1.0 if index < len(points) - 1 and strokes[index] == strokes[index + 1] else 0.0
        features.append(
            [
                round_float(dx),
                round_float(dy),
                round_float(sin_alpha),
                round_float(cos_alpha),
                round_float(sin_beta),
                round_float(cos_beta),
                same_stroke,
                1.0 - same_stroke,
            ]
        )
    return features


def resample_by_stroke(
    points: list[tuple[float, float]],
    strokes: list[int],
    target_points: int,
) -> tuple[list[tuple[float, float]], list[int]]:
    if not points:
        return [], []
    if len(points) == 1:
        return [points[0] for _ in range(target_points)], [strokes[0] for _ in range(target_points)]

    chunks = split_stroke_chunks(points, strokes)
    if len(chunks) >= target_points:
        sampled = []
        sampled_strokes = []
        for chunk in chunks[:target_points]:
            sampled.append(chunk["points"][0])
            sampled_strokes.append(chunk["stroke"])
        return sampled, sampled_strokes

    weights = [max(stroke_length(chunk["points"]), float(len(chunk["points"]))) for chunk in chunks]
    total_weight = sum(weights) or float(len(chunks))
    raw_allocations = [target_points * weight / total_weight for weight in weights]
    allocations = [max(1, int(math.floor(value))) for value in raw_allocations]

    while sum(allocations) > target_points:
        candidates = [
            (allocations[index] - raw_allocations[index], index)
            for index in range(len(allocations))
            if allocations[index] > 1
        ]
        if not candidates:
            break
        _, chosen_index = max(candidates)
        allocations[chosen_index] -= 1

    while sum(allocations) < target_points:
        candidates = [
            (raw_allocations[index] - allocations[index], weights[index], index)
            for index in range(len(allocations))
        ]
        _, _, chosen_index = max(candidates)
        allocations[chosen_index] += 1

    resampled_points: list[tuple[float, float]] = []
    resampled_strokes: list[int] = []
    for chunk, allocation in zip(chunks, allocations, strict=True):
        chunk_points = resample_chunk(chunk["points"], allocation)
        resampled_points.extend(chunk_points)
        resampled_strokes.extend([chunk["stroke"]] * len(chunk_points))

    if len(resampled_points) > target_points:
        resampled_points = resampled_points[:target_points]
        resampled_strokes = resampled_strokes[:target_points]
    while len(resampled_points) < target_points:
        resampled_points.append(resampled_points[-1])
        resampled_strokes.append(resampled_strokes[-1])

    return resampled_points, resampled_strokes


def split_stroke_chunks(
    points: list[tuple[float, float]],
    strokes: list[int],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_points: list[tuple[float, float]] = []
    current_stroke: int | None = None
    for point, stroke in zip(points, strokes, strict=True):
        if current_stroke is None or stroke == current_stroke:
            current_points.append(point)
            current_stroke = stroke
            continue
        chunks.append({"stroke": current_stroke, "points": current_points})
        current_points = [point]
        current_stroke = stroke
    if current_points and current_stroke is not None:
        chunks.append({"stroke": current_stroke, "points": current_points})
    return chunks


def stroke_length(points: list[tuple[float, float]]) -> float:
    return sum(euclidean(points[index], points[index - 1]) for index in range(1, len(points)))


def resample_chunk(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(count)]
    if count == 1:
        return [points[0]]

    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + euclidean(points[index], points[index - 1]))
    total = cumulative[-1]
    if total <= POINT_EPSILON:
        return [points[0] for _ in range(count)]

    targets = [total * index / (count - 1) for index in range(count)]
    resampled: list[tuple[float, float]] = []
    segment_index = 1
    for target in targets:
        while segment_index < len(cumulative) - 1 and cumulative[segment_index] < target:
            segment_index += 1
        prev_distance = cumulative[segment_index - 1]
        next_distance = cumulative[segment_index]
        ratio = 0.0 if next_distance == prev_distance else (target - prev_distance) / (next_distance - prev_distance)
        prev_point = points[segment_index - 1]
        next_point = points[segment_index]
        resampled.append(
            (
                prev_point[0] + (next_point[0] - prev_point[0]) * ratio,
                prev_point[1] + (next_point[1] - prev_point[1]) * ratio,
            )
        )
    return resampled


def trajectory_hash(points: list[tuple[float, float]], strokes: list[int]) -> str:
    payload = [
        [round_float(x), round_float(y), int(stroke)]
        for (x, y), stroke in zip(points, strokes, strict=True)
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def round_float(value: float, places: int = 6) -> float:
    return round(float(value), places)


def mark_duplicate_sample_ids(candidates: list[Candidate]) -> None:
    seen: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.json_path.as_posix()):
        if not candidate.sample_id:
            continue
        if candidate.sample_id in seen:
            candidate.duplicate_of = seen[candidate.sample_id].sample_id
            candidate.duplicate_type = "sample_id"
            candidate.exclusions.append(
                ("duplicate_sample_id", f"first_json={seen[candidate.sample_id].json_path.as_posix()}")
            )
            continue
        seen[candidate.sample_id] = candidate


def mark_exact_duplicates(candidates: list[Candidate]) -> None:
    seen: dict[tuple[str, str], Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.json_path.as_posix()):
        duplicate_key = None
        for hash_name in ("image", "video", "trajectory_cleaned"):
            digest = candidate.hashes.get(hash_name)
            if not digest:
                continue
            key = (hash_name, digest)
            if key in seen:
                duplicate_key = key
                break
        if duplicate_key is None:
            for hash_name in ("image", "video", "trajectory_cleaned"):
                digest = candidate.hashes.get(hash_name)
                if digest:
                    seen[(hash_name, digest)] = candidate
            continue

        first = seen[duplicate_key]
        candidate.duplicate_of = first.sample_id
        candidate.duplicate_type = f"exact_{duplicate_key[0]}_hash"
        candidate.exclusions.append(
            (
                "exact_duplicate",
                f"duplicate_of={first.sample_id};type={candidate.duplicate_type}",
            )
        )


def mark_near_duplicates(candidates: list[Candidate], threshold: float) -> None:
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.user_id, candidate.label), []).append(candidate)

    for group in groups.values():
        keepers: list[Candidate] = []
        for candidate in sorted(group, key=lambda item: (parse_datetime_key(item.created_at), item.json_path.as_posix())):
            duplicate_of: Candidate | None = None
            duplicate_distance: float | None = None
            for keeper in keepers:
                distance = dtw_mean_distance(candidate.resampled_points, keeper.resampled_points, threshold)
                if distance <= threshold:
                    duplicate_of = keeper
                    duplicate_distance = distance
                    break
            if duplicate_of is None:
                keepers.append(candidate)
                continue
            candidate.duplicate_of = duplicate_of.sample_id
            candidate.duplicate_type = "near_dtw"
            candidate.duplicate_distance = duplicate_distance
            candidate.exclusions.append(
                (
                    "near_duplicate",
                    f"duplicate_of={duplicate_of.sample_id};dtw_mean={round_float(duplicate_distance or 0.0)}",
                )
            )


def parse_datetime_key(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return value


def dtw_mean_distance(
    points_a: list[tuple[float, float]],
    points_b: list[tuple[float, float]],
    threshold: float,
) -> float:
    if not points_a or not points_b:
        return float("inf")

    n = len(points_a)
    m = len(points_b)
    max_allowed = threshold * (n + m)
    previous = [float("inf")] * (m + 1)
    current = [float("inf")] * (m + 1)
    previous[0] = 0.0

    for i in range(1, n + 1):
        current[0] = float("inf")
        row_min = float("inf")
        point_a = points_a[i - 1]
        for j in range(1, m + 1):
            cost = euclidean(point_a, points_b[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
            row_min = min(row_min, current[j])
        if row_min > max_allowed:
            return float("inf")
        previous, current = current, [float("inf")] * (m + 1)

    return previous[m] / (n + m)


def assign_user_folds(clean_samples: list[Candidate]) -> None:
    users = sorted({candidate.user_id for candidate in clean_samples})
    if not users:
        return
    for fold_id, user in enumerate(users):
        for candidate in clean_samples:
            if candidate.user_id == user:
                candidate.test_fold_id = fold_id


def build_nested_split_assignments(clean_samples: list[Candidate]) -> list[dict[str, Any]]:
    users = sorted({candidate.user_id for candidate in clean_samples})
    if not users:
        return []

    sample_counts_by_user: dict[str, int] = {}
    for candidate in clean_samples:
        sample_counts_by_user[candidate.user_id] = sample_counts_by_user.get(candidate.user_id, 0) + 1

    outer_split_policy = "leave_one_user_out" if len(users) >= 5 else f"grouped_{len(users)}_fold"
    rows: list[dict[str, Any]] = []
    for outer_fold_id, test_user in enumerate(users):
        development_users = [user for user in users if user != test_user]
        inner_fold_count = min(3, len(development_users))
        inner_split_policy = f"grouped_{inner_fold_count}_fold_inner_validation"
        validation_groups = assign_balanced_user_groups(development_users, sample_counts_by_user, inner_fold_count)

        for inner_fold_id, validation_users in enumerate(validation_groups):
            validation_set = set(validation_users)
            for user in users:
                if user == test_user:
                    role = "test"
                elif user in validation_set:
                    role = "val"
                else:
                    role = "train"
                rows.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "inner_fold_id": inner_fold_id,
                        "user_id": user,
                        "role": role,
                        "outer_split_policy": outer_split_policy,
                        "inner_split_policy": inner_split_policy,
                    }
                )
    return rows


def assign_balanced_user_groups(
    users: list[str],
    sample_counts_by_user: dict[str, int],
    group_count: int,
) -> list[list[str]]:
    if group_count <= 0:
        return []
    groups: list[list[str]] = [[] for _ in range(group_count)]
    group_sizes = [0 for _ in range(group_count)]
    for user in sorted(users, key=lambda item: (-sample_counts_by_user.get(item, 0), item)):
        group_index = min(range(group_count), key=lambda index: (group_sizes[index], index))
        groups[group_index].append(user)
        group_sizes[group_index] += sample_counts_by_user.get(user, 0)
    return [sorted(group) for group in groups]


def build_fold_scoped_augmentations(
    clean_samples: list[Candidate],
    nested_splits: list[dict[str, Any]],
    *,
    augmentations_per_sample: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if augmentations_per_sample <= 0:
        return [], []

    samples_by_user: dict[str, list[Candidate]] = {}
    for candidate in clean_samples:
        samples_by_user.setdefault(candidate.user_id, []).append(candidate)

    augmentations: list[dict[str, Any]] = []
    video_plans: list[dict[str, Any]] = []

    train_assignments = [row for row in nested_splits if row["role"] == "train"]
    for assignment in train_assignments:
        outer_fold_id = int(assignment["outer_fold_id"])
        inner_fold_id = int(assignment["inner_fold_id"])
        for candidate in samples_by_user.get(str(assignment["user_id"]), []):
            for variant_index in range(augmentations_per_sample):
                params = sample_augmentation_params(rng)
                augmented_points, augmented_strokes = augment_trajectory(
                    candidate.bbox_points,
                    candidate.clean_strokes,
                    params,
                    rng,
                )
                if len(augmented_points) < MIN_VALID_POINTS:
                    augmented_points = candidate.bbox_points
                    augmented_strokes = candidate.clean_strokes
                resampled_points, resampled_strokes = resample_by_stroke(
                    augmented_points,
                    augmented_strokes,
                    TARGET_POINTS,
                )
                augmented_id = (
                    f"{candidate.sample_id}__outer{outer_fold_id:02d}"
                    f"__inner{inner_fold_id:02d}__aug{variant_index + 1:02d}"
                )
                row = {
                    "augmented_sample_id": augmented_id,
                    "source_sample_id": candidate.sample_id,
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "role": "train",
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "trajectory": {
                        "points_augmented_variable": round_points(augmented_points),
                        "stroke_ids_variable": augmented_strokes,
                        "features_8d_variable": compute_features(augmented_points, augmented_strokes),
                        "resampled_128": {
                            "points": round_points(resampled_points),
                            "stroke_ids": resampled_strokes,
                            "features_8d": compute_features(resampled_points, resampled_strokes),
                        },
                    },
                    "augmentation": params,
                }
                augmentations.append(row)
                video_plans.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "inner_fold_id": inner_fold_id,
                        "augmented_sample_id": augmented_id,
                        "source_sample_id": candidate.sample_id,
                        "user_id": candidate.user_id,
                        "label": candidate.label,
                        "source_video_path": candidate.video_path,
                        "brightness": params["video_brightness"],
                        "contrast": params["video_contrast"],
                        "motion_blur_kernel": params["video_motion_blur_kernel"],
                        "frame_dropout_rate": params["video_frame_dropout_rate"],
                        "crop_translation_x": params["video_crop_translation_x"],
                        "crop_translation_y": params["video_crop_translation_y"],
                    }
                )

    return augmentations, video_plans


def sample_augmentation_params(rng: random.Random) -> dict[str, float | int]:
    blur_kernel = rng.choice([3, 5])
    return {
        "rotation_deg": round_float(rng.uniform(-7.0, 7.0)),
        "scale": round_float(rng.uniform(0.90, 1.10)),
        "translation_x": round_float(rng.uniform(-0.05, 0.05)),
        "translation_y": round_float(rng.uniform(-0.05, 0.05)),
        "jitter_sigma": round_float(rng.uniform(0.003, 0.008)),
        "point_dropout_rate": round_float(rng.uniform(0.03, 0.08)),
        "temporal_warp": round_float(rng.uniform(0.80, 1.25)),
        "keypoint_jitter_sigma": 0.005,
        "keypoint_temporal_mask_rate": round_float(rng.uniform(0.05, 0.15)),
        "keypoint_rotation_deg": round_float(rng.uniform(-7.0, 7.0)),
        "video_brightness": round_float(rng.uniform(0.85, 1.15)),
        "video_contrast": round_float(rng.uniform(0.85, 1.15)),
        "video_motion_blur_kernel": blur_kernel,
        "video_frame_dropout_rate": round_float(rng.uniform(0.0, 0.10)),
        "video_crop_translation_x": round_float(rng.uniform(-0.05, 0.05)),
        "video_crop_translation_y": round_float(rng.uniform(-0.05, 0.05)),
    }


def augment_trajectory(
    points: list[tuple[float, float]],
    strokes: list[int],
    params: dict[str, float | int],
    rng: random.Random,
) -> tuple[list[tuple[float, float]], list[int]]:
    if not points:
        return [], []

    warped_length = max(MIN_VALID_POINTS, int(round(len(points) * float(params["temporal_warp"]))))
    warped_points, warped_strokes = resample_by_index(points, strokes, warped_length)

    angle = math.radians(float(params["rotation_deg"]))
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    scale = float(params["scale"])
    tx = float(params["translation_x"])
    ty = float(params["translation_y"])
    jitter_sigma = float(params["jitter_sigma"])

    transformed: list[tuple[float, float]] = []
    for x, y in warped_points:
        xr = (x * cos_angle - y * sin_angle) * scale + tx
        yr = (x * sin_angle + y * cos_angle) * scale + ty
        transformed.append((xr + rng.gauss(0.0, jitter_sigma), yr + rng.gauss(0.0, jitter_sigma)))

    keep = dropout_mask_by_stroke(warped_strokes, float(params["point_dropout_rate"]), rng)
    dropped_points = [point for point, keep_point in zip(transformed, keep, strict=True) if keep_point]
    dropped_strokes = [stroke for stroke, keep_point in zip(warped_strokes, keep, strict=True) if keep_point]
    return dropped_points, dropped_strokes


def resample_by_index(
    points: list[tuple[float, float]],
    strokes: list[int],
    target_length: int,
) -> tuple[list[tuple[float, float]], list[int]]:
    if not points:
        return [], []
    if len(points) == 1:
        return [points[0] for _ in range(target_length)], [strokes[0] for _ in range(target_length)]
    if target_length <= 1:
        return [points[0]], [strokes[0]]

    output_points: list[tuple[float, float]] = []
    output_strokes: list[int] = []
    last_index = len(points) - 1
    for output_index in range(target_length):
        position = output_index * last_index / (target_length - 1)
        lower = int(math.floor(position))
        upper = min(last_index, lower + 1)
        ratio = position - lower
        if strokes[lower] != strokes[upper]:
            chosen = lower if ratio < 0.5 else upper
            output_points.append(points[chosen])
            output_strokes.append(strokes[chosen])
            continue
        x = points[lower][0] + (points[upper][0] - points[lower][0]) * ratio
        y = points[lower][1] + (points[upper][1] - points[lower][1]) * ratio
        output_points.append((x, y))
        output_strokes.append(strokes[lower])
    return output_points, output_strokes


def dropout_mask_by_stroke(strokes: list[int], rate: float, rng: random.Random) -> list[bool]:
    keep = [True] * len(strokes)
    if len(strokes) <= MIN_VALID_POINTS:
        return keep

    first_by_stroke: set[int] = set()
    last_by_stroke: set[int] = set()
    for index, stroke in enumerate(strokes):
        if index == 0 or stroke != strokes[index - 1]:
            first_by_stroke.add(index)
        if index == len(strokes) - 1 or stroke != strokes[index + 1]:
            last_by_stroke.add(index)

    protected = first_by_stroke | last_by_stroke | {0, len(strokes) - 1}
    for index in range(len(strokes)):
        if index in protected:
            continue
        keep[index] = rng.random() >= rate

    if sum(keep) < MIN_VALID_POINTS:
        kept_indices = {index for index, value in enumerate(keep) if value}
        missing = MIN_VALID_POINTS - len(kept_indices)
        available = [index for index in range(len(strokes)) if index not in kept_indices]
        rng.shuffle(available)
        for index in available[:missing]:
            keep[index] = True
    return keep


def build_training_weights(
    clean_samples: list[Candidate],
    nested_splits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_user: dict[str, list[Candidate]] = {}
    for candidate in clean_samples:
        samples_by_user.setdefault(candidate.user_id, []).append(candidate)

    rows: list[dict[str, Any]] = []
    train_keys = sorted(
        {
            (int(row["outer_fold_id"]), int(row["inner_fold_id"]))
            for row in nested_splits
            if row["role"] == "train"
        }
    )
    for outer_fold_id, inner_fold_id in train_keys:
        train_users = {
            str(row["user_id"])
            for row in nested_splits
            if int(row["outer_fold_id"]) == outer_fold_id
            and int(row["inner_fold_id"]) == inner_fold_id
            and row["role"] == "train"
        }
        train_samples = [
            candidate
            for user in sorted(train_users)
            for candidate in samples_by_user.get(user, [])
        ]
        class_counts: dict[str, int] = {}
        user_counts: dict[str, int] = {}
        for candidate in train_samples:
            class_counts[candidate.label] = class_counts.get(candidate.label, 0) + 1
            user_counts[candidate.user_id] = user_counts.get(candidate.user_id, 0) + 1
        raw_weights = []
        for candidate in train_samples:
            class_count = class_counts[candidate.label]
            user_count = user_counts[candidate.user_id]
            weight = 1.0 / math.sqrt(class_count * user_count)
            raw_weights.append(weight)
        mean_weight = statistics.mean(raw_weights) if raw_weights else 1.0
        for candidate, raw_weight in zip(train_samples, raw_weights, strict=True):
            rows.append(
                {
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "sample_id": candidate.sample_id,
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "class_count_in_train": class_counts[candidate.label],
                    "user_count_in_train": user_counts[candidate.user_id],
                    "sample_weight": round_float(raw_weight / mean_weight),
                }
            )
    return rows


def write_outputs(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    candidates: list[Candidate],
    clean_samples: list[Candidate],
    nested_splits: list[dict[str, Any]],
    augmentations: list[dict[str, Any]],
    video_plans: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    write_manifest(repo_root, output_dir, candidates)
    write_exclusions(repo_root, output_dir, candidates)
    write_inventory(repo_root, input_dir, output_dir, candidates, clean_samples, augmentations)
    write_folds(repo_root, output_dir, clean_samples, nested_splits)
    write_clean_manifest(repo_root, output_dir, clean_samples)
    write_cleaned_jsonl(repo_root, output_dir, clean_samples)
    write_augmentations(repo_root, output_dir, augmentations, video_plans)
    write_training_weights(output_dir, weights)
    write_run_summary(output_dir, candidates, clean_samples, nested_splits, augmentations, args)


def write_manifest(repo_root: Path, output_dir: Path, candidates: list[Candidate]) -> None:
    rows = [
        {
            "sample_id": candidate.sample_id,
            "user_id": candidate.user_id,
            "label": candidate.label,
            "json_path": relpath(candidate.json_path, repo_root),
            "image_path": relpath(candidate.image_path, repo_root),
            "video_path": relpath(candidate.video_path, repo_root),
            "device": candidate.device,
            "created_at": candidate.created_at,
        }
        for candidate in candidates
    ]
    write_csv_file(
        output_dir / "manifest.csv",
        rows,
        ("sample_id", "user_id", "label", "json_path", "image_path", "video_path", "device", "created_at"),
    )


def write_exclusions(repo_root: Path, output_dir: Path, candidates: list[Candidate]) -> None:
    exclusion_rows = []
    warning_rows = []
    duplicate_rows = []
    for candidate in candidates:
        for reason, details in candidate.exclusions:
            exclusion_rows.append(
                {
                    "sample_id": candidate.sample_id,
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "json_path": relpath(candidate.json_path, repo_root),
                    "reason": reason,
                    "details": details,
                }
            )
        for warning in candidate.warnings:
            warning_rows.append(
                {
                    "sample_id": candidate.sample_id,
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "json_path": relpath(candidate.json_path, repo_root),
                    "warning": warning,
                }
            )
        if candidate.duplicate_type:
            duplicate_rows.append(
                {
                    "sample_id": candidate.sample_id,
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "duplicate_of": candidate.duplicate_of,
                    "duplicate_type": candidate.duplicate_type,
                    "distance": "" if candidate.duplicate_distance is None else round_float(candidate.duplicate_distance),
                    "json_path": relpath(candidate.json_path, repo_root),
                }
            )

    write_csv_file(
        output_dir / "excluded_samples.csv",
        exclusion_rows,
        ("sample_id", "user_id", "label", "json_path", "reason", "details"),
    )
    write_csv_file(
        output_dir / "sample_warnings.csv",
        warning_rows,
        ("sample_id", "user_id", "label", "json_path", "warning"),
    )
    write_csv_file(
        output_dir / "duplicates_report.csv",
        duplicate_rows,
        ("sample_id", "user_id", "label", "duplicate_of", "duplicate_type", "distance", "json_path"),
    )


def write_inventory(
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    candidates: list[Candidate],
    clean_samples: list[Candidate],
    augmentations: list[dict[str, Any]],
) -> None:
    inventory: dict[tuple[str, str, str], int] = {}
    for path in sorted(input_dir.glob("*/*/*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext not in {"json", "png", "webm", "mp4"}:
            continue
        user = normalize_text(path.parent.parent.name)
        label = normalize_text(path.parent.name)
        inventory[(user, label, ext)] = inventory.get((user, label, ext), 0) + 1

    inventory_rows = [
        {"user_id": user, "label": label, "extension": ext, "count": count}
        for (user, label, ext), count in sorted(inventory.items())
    ]
    write_csv_file(output_dir / "inventory_counts.csv", inventory_rows, ("user_id", "label", "extension", "count"))

    manifest_counts: dict[tuple[str, str], int] = {}
    clean_counts: dict[tuple[str, str], int] = {}
    augmented_counts: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        manifest_counts[(candidate.user_id, candidate.label)] = manifest_counts.get((candidate.user_id, candidate.label), 0) + 1
    for candidate in clean_samples:
        clean_counts[(candidate.user_id, candidate.label)] = clean_counts.get((candidate.user_id, candidate.label), 0) + 1
    for row in augmentations:
        key = (str(row["user_id"]), str(row["label"]))
        augmented_counts[key] = augmented_counts.get(key, 0) + 1

    users = sorted({key[0] for key in inventory} | {candidate.user_id for candidate in candidates})
    labels = sorted({key[1] for key in inventory} | set(BANGLA_VOWELS))
    class_rows = []
    for user in users:
        for label in labels:
            json_count = inventory.get((user, label, "json"), 0)
            png_count = inventory.get((user, label, "png"), 0)
            webm_count = inventory.get((user, label, "webm"), 0)
            mp4_count = inventory.get((user, label, "mp4"), 0)
            manifest_count = manifest_counts.get((user, label), 0)
            clean_count = clean_counts.get((user, label), 0)
            augmented_count = augmented_counts.get((user, label), 0)
            if any((json_count, png_count, webm_count, mp4_count, manifest_count, clean_count, augmented_count)):
                class_rows.append(
                    {
                        "user_id": user,
                        "label": label,
                        "json_count": json_count,
                        "png_count": png_count,
                        "video_count": webm_count + mp4_count,
                        "manifest_samples": manifest_count,
                        "clean_samples": clean_count,
                        "augmented_train_rows": augmented_count,
                    }
                )
    write_csv_file(
        output_dir / "class_user_counts.csv",
        class_rows,
        (
            "user_id",
            "label",
            "json_count",
            "png_count",
            "video_count",
            "manifest_samples",
            "clean_samples",
            "augmented_train_rows",
        ),
    )


def write_folds(
    repo_root: Path,
    output_dir: Path,
    clean_samples: list[Candidate],
    nested_splits: list[dict[str, Any]],
) -> None:
    users = sorted({candidate.user_id for candidate in clean_samples})
    split_policy = "leave_one_user_out" if len(users) >= 5 else f"grouped_{len(users)}_fold"
    rows = []
    for fold_id, test_user in enumerate(users):
        for user in users:
            rows.append(
                {
                    "fold_id": fold_id,
                    "user_id": user,
                    "role": "test" if user == test_user else "train",
                    "split_policy": split_policy,
                }
            )
    write_csv_file(output_dir / "fold_assignments.csv", rows, ("fold_id", "user_id", "role", "split_policy"))

    write_csv_file(
        output_dir / "nested_fold_assignments.csv",
        nested_splits,
        (
            "outer_fold_id",
            "inner_fold_id",
            "user_id",
            "role",
            "outer_split_policy",
            "inner_split_policy",
        ),
    )

    split_lookup = {
        (int(row["outer_fold_id"]), int(row["inner_fold_id"]), str(row["user_id"])): str(row["role"])
        for row in nested_splits
    }
    split_keys = sorted({(int(row["outer_fold_id"]), int(row["inner_fold_id"])) for row in nested_splits})
    sample_rows = []
    for outer_fold_id, inner_fold_id in split_keys:
        for candidate in clean_samples:
            sample_rows.append(
                {
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "sample_id": candidate.sample_id,
                    "user_id": candidate.user_id,
                    "label": candidate.label,
                    "role": split_lookup[(outer_fold_id, inner_fold_id, candidate.user_id)],
                    "json_path": relpath(candidate.json_path, repo_root),
                    "image_path": relpath(candidate.image_path, repo_root),
                    "video_path": relpath(candidate.video_path, repo_root),
                }
            )
    write_csv_file(
        output_dir / "sample_splits.csv",
        sample_rows,
        (
            "outer_fold_id",
            "inner_fold_id",
            "sample_id",
            "user_id",
            "label",
            "role",
            "json_path",
            "image_path",
            "video_path",
        ),
    )


def write_clean_manifest(repo_root: Path, output_dir: Path, clean_samples: list[Candidate]) -> None:
    rows = []
    for candidate in clean_samples:
        rows.append(
            {
                "sample_id": candidate.sample_id,
                "user_id": candidate.user_id,
                "label": candidate.label,
                "json_path": relpath(candidate.json_path, repo_root),
                "image_path": relpath(candidate.image_path, repo_root),
                "video_path": relpath(candidate.video_path, repo_root),
                "device": candidate.device,
                "created_at": candidate.created_at,
                "raw_points": candidate.cleaning.get("raw_points", ""),
                "clean_points": candidate.cleaning.get("clean_points", ""),
                "dropped_invalid": candidate.cleaning.get("dropped_invalid", 0),
                "dropped_repeated": candidate.cleaning.get("dropped_repeated", 0),
                "dropped_jumps": candidate.cleaning.get("dropped_jumps", 0),
                "num_strokes": len(set(candidate.clean_strokes)),
                "bbox_width": candidate.normalization.get("bbox_width", ""),
                "bbox_height": candidate.normalization.get("bbox_height", ""),
                "bbox_scale": candidate.normalization.get("bbox_scale", ""),
                "test_fold_id": candidate.test_fold_id,
                "warnings": ";".join(candidate.warnings),
            }
        )
    write_csv_file(
        output_dir / "clean_manifest.csv",
        rows,
        (
            "sample_id",
            "user_id",
            "label",
            "json_path",
            "image_path",
            "video_path",
            "device",
            "created_at",
            "raw_points",
            "clean_points",
            "dropped_invalid",
            "dropped_repeated",
            "dropped_jumps",
            "num_strokes",
            "bbox_width",
            "bbox_height",
            "bbox_scale",
            "test_fold_id",
            "warnings",
        ),
    )


def write_cleaned_jsonl(repo_root: Path, output_dir: Path, clean_samples: list[Candidate]) -> None:
    path = output_dir / "cleaned_trajectories.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for candidate in clean_samples:
            payload = {
                "sample_id": candidate.sample_id,
                "user_id": candidate.user_id,
                "label": candidate.label,
                "created_at": candidate.created_at,
                "device": candidate.device,
                "source": {
                    "json_path": relpath(candidate.json_path, repo_root),
                    "image_path": relpath(candidate.image_path, repo_root),
                    "video_path": relpath(candidate.video_path, repo_root),
                },
                "fold": {"test_fold_id": candidate.test_fold_id},
                "cleaning": candidate.cleaning,
                "normalization": candidate.normalization,
                "trajectory": {
                    "points_cleaned": round_points(candidate.clean_points),
                    "stroke_ids": candidate.clean_strokes,
                    "features_8d_cleaned": candidate.clean_features,
                    "bbox_normalized_points": round_points(candidate.bbox_points),
                    "features_8d_bbox_normalized": candidate.bbox_features,
                    "resampled_128": {
                        "points": round_points(candidate.resampled_points),
                        "stroke_ids": candidate.resampled_strokes,
                        "features_8d": candidate.resampled_features,
                    },
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_augmentations(
    repo_root: Path,
    output_dir: Path,
    augmentations: list[dict[str, Any]],
    video_plans: list[dict[str, Any]],
) -> None:
    jsonl_path = output_dir / "augmented_train_trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in augmentations:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest_rows = []
    for row in augmentations:
        params = row["augmentation"]
        manifest_rows.append(
            {
                "outer_fold_id": row["outer_fold_id"],
                "inner_fold_id": row["inner_fold_id"],
                "augmented_sample_id": row["augmented_sample_id"],
                "source_sample_id": row["source_sample_id"],
                "role": row["role"],
                "user_id": row["user_id"],
                "label": row["label"],
                "rotation_deg": params["rotation_deg"],
                "scale": params["scale"],
                "translation_x": params["translation_x"],
                "translation_y": params["translation_y"],
                "jitter_sigma": params["jitter_sigma"],
                "point_dropout_rate": params["point_dropout_rate"],
                "temporal_warp": params["temporal_warp"],
            }
        )
    write_csv_file(
        output_dir / "augmented_manifest.csv",
        manifest_rows,
        (
            "outer_fold_id",
            "inner_fold_id",
            "augmented_sample_id",
            "source_sample_id",
            "role",
            "user_id",
            "label",
            "rotation_deg",
            "scale",
            "translation_x",
            "translation_y",
            "jitter_sigma",
            "point_dropout_rate",
            "temporal_warp",
        ),
    )

    video_rows = [
        {
            **row,
            "source_video_path": relpath(row["source_video_path"], repo_root),
        }
        for row in video_plans
    ]
    write_csv_file(
        output_dir / "video_augmentation_plan.csv",
        video_rows,
        (
            "outer_fold_id",
            "inner_fold_id",
            "augmented_sample_id",
            "source_sample_id",
            "user_id",
            "label",
            "source_video_path",
            "brightness",
            "contrast",
            "motion_blur_kernel",
            "frame_dropout_rate",
            "crop_translation_x",
            "crop_translation_y",
        ),
    )


def write_training_weights(output_dir: Path, weights: list[dict[str, Any]]) -> None:
    write_csv_file(
        output_dir / "training_sample_weights.csv",
        weights,
        (
            "outer_fold_id",
            "inner_fold_id",
            "sample_id",
            "user_id",
            "label",
            "class_count_in_train",
            "user_count_in_train",
            "sample_weight",
        ),
    )


def write_run_summary(
    output_dir: Path,
    candidates: list[Candidate],
    clean_samples: list[Candidate],
    nested_splits: list[dict[str, Any]],
    augmentations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    reason_counts: dict[str, int] = {}
    for candidate in candidates:
        for reason, _details in candidate.exclusions:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    users = sorted({candidate.user_id for candidate in clean_samples})
    labels = sorted({candidate.label for candidate in clean_samples})
    inner_counts_by_outer: dict[int, set[int]] = {}
    sample_counts_by_user: dict[str, int] = {}
    sample_role_rows = {"train": 0, "val": 0, "test": 0}
    for candidate in clean_samples:
        sample_counts_by_user[candidate.user_id] = sample_counts_by_user.get(candidate.user_id, 0) + 1
    for row in nested_splits:
        outer_fold_id = int(row["outer_fold_id"])
        inner_fold_id = int(row["inner_fold_id"])
        role = str(row["role"])
        user_id = str(row["user_id"])
        inner_counts_by_outer.setdefault(outer_fold_id, set()).add(inner_fold_id)
        sample_role_rows[role] = sample_role_rows.get(role, 0) + sample_counts_by_user.get(user_id, 0)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir,
        "total_json_samples": len(candidates),
        "clean_samples": len(clean_samples),
        "excluded_samples": len({candidate.sample_id for candidate in candidates if candidate.is_excluded}),
        "exclusion_reasons": reason_counts,
        "users": users,
        "labels": labels,
        "split_policy": "leave_one_user_out" if len(users) >= 5 else f"grouped_{len(users)}_fold",
        "outer_fold_count": len(inner_counts_by_outer),
        "inner_validation_folds_per_outer": {
            str(outer_fold_id): len(inner_fold_ids)
            for outer_fold_id, inner_fold_ids in sorted(inner_counts_by_outer.items())
        },
        "nested_user_split_rows": len(nested_splits),
        "sample_split_rows": sum(sample_role_rows.values()),
        "sample_role_rows": sample_role_rows,
        "augmentations_per_sample": args.augmentations_per_sample,
        "augmented_train_rows": len(augmentations),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "jump_abs_threshold": args.jump_abs_threshold,
        "jump_median_factor": args.jump_median_factor,
        "notes": [
            "Outer folds hold out one full user as test.",
            "Inner folds choose validation users only from the outer-fold training users.",
            "Augmentation rows are generated only for train roles after train/validation/test assignment.",
            "Video augmentation is recorded as deterministic per-row parameters; raw video files are not rewritten.",
        ],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv_file(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def relpath(path: Path | str | None, repo_root: Path) -> str:
    if path is None or path == "":
        return ""
    path = Path(path)
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def round_points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[round_float(x), round_float(y)] for x, y in points]


def print_summary(
    output_dir: Path,
    candidates: list[Candidate],
    clean_samples: list[Candidate],
    augmentations: list[dict[str, Any]],
) -> None:
    excluded_count = len({candidate.sample_id for candidate in candidates if candidate.is_excluded})
    print(f"Wrote procedure artifacts to {output_dir}")
    print(f"JSON samples: {len(candidates)}")
    print(f"Clean samples: {len(clean_samples)}")
    print(f"Excluded samples: {excluded_count}")
    print(f"Augmented train rows: {len(augmentations)}")


if __name__ == "__main__":
    raise SystemExit(main())
